"""
Training script for FULL VIDEO cut selection model

1 VIDEO = 1 SAMPLE (no sequence splitting)
Applies 90s minimum constraint PER VIDEO
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cut_selection.datasets.cut_dataset_enhanced_fullvideo import EnhancedCutSelectionDatasetFullVideo, collate_fn_fullvideo
from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel
from src.cut_selection.utils.losses import CombinedCutSelectionLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント対応
try:
    import japanize_matplotlib
    logger.info("✅ japanize_matplotlib loaded")
except ImportError:
    logger.warning("⚠️  japanize_matplotlib not installed")


def set_seed(seed: int):
    """完全な再現性のためにシードを固定"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 Random seed set to {seed} for reproducibility")


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, max_grad_norm=1.0):
    """Train for one epoch with full videos"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    num_videos = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        temporal = batch['temporal'].to(device)
        active_labels = batch['active'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(audio, visual, temporal, padding_mask=padding_mask)
                active_logits = outputs['active']
                loss, loss_dict = criterion(active_logits, active_labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(audio, visual, temporal, padding_mask=padding_mask)
            active_logits = outputs['active']
            loss, loss_dict = criterion(active_logits, active_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_ce_loss += loss_dict['ce_loss']
        total_tv_loss += loss_dict['tv_loss']
        num_videos += audio.shape[0]
    
    return {
        'total_loss': total_loss / num_videos,
        'ce_loss': total_ce_loss / num_videos,
        'tv_loss': total_tv_loss / num_videos
    }


def validate(model, dataloader, criterion, device, config=None):
    """Validate the model with full videos - apply 90s constraint PER VIDEO"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    
    all_predictions = []
    all_labels = []
    all_confidence_scores = []
    all_video_names = []
    all_original_lengths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            temporal = batch['temporal'].to(device)
            active_labels = batch['active'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            outputs = model(audio, visual, temporal, padding_mask=padding_mask)
            active_logits = outputs['active']
            
            loss, loss_dict = criterion(active_logits, active_labels)
            total_loss += loss_dict['total_loss']
            total_ce_loss += loss_dict['ce_loss']
            total_tv_loss += loss_dict['tv_loss']
            
            predictions = torch.argmax(active_logits, dim=-1)
            probs = torch.softmax(active_logits, dim=-1)
            confidence_scores = probs[..., 1] - probs[..., 0]
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(active_labels.cpu().numpy())
            all_confidence_scores.append(confidence_scores.cpu().numpy())
            all_video_names.extend(batch['video_names'])
            all_original_lengths.extend(batch['original_lengths'])
    
    # Calculate average losses
    num_videos = len(all_video_names)
    avg_total_loss = total_loss / num_videos
    avg_ce_loss = total_ce_loss / num_videos
    avg_tv_loss = total_tv_loss / num_videos
    
    # Get duration constraint parameters
    max_total_duration = config.get('max_total_duration', 180.0) if config else 180.0
    hard_max_duration = config.get('hard_max_duration', 200.0) if config else 200.0
    min_total_duration = config.get('min_total_duration', 90.0) if config else 90.0
    fps = config.get('inference_fps', 10.0) if config else 10.0
    min_clip_duration = config.get('min_clip_duration', 3.0) if config else 3.0
    
    def extract_clips_from_predictions(predictions, fps=10.0, min_duration=3.0):
        """Extract clips from binary predictions"""
        clips = []
        in_clip = False
        start_idx = 0
        
        for i, active in enumerate(predictions):
            if active == 1 and not in_clip:
                start_idx = i
                in_clip = True
            elif active == 0 and in_clip:
                end_idx = i
                start_time = start_idx / fps
                end_time = end_idx / fps
                if (end_time - start_time) >= min_duration:
                    clips.append((start_time, end_time))
                in_clip = False
        
        # Handle last clip
        if in_clip:
            end_idx = len(predictions)
            start_time = start_idx / fps
            end_time = end_idx / fps
            if (end_time - start_time) >= min_duration:
                clips.append((start_time, end_time))
        
        return clips
    
    # Process each video separately and apply 90s constraint PER VIDEO
    video_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_pred_duration = 0.0
    
    for video_idx in range(num_videos):
        video_name = all_video_names[video_idx]
        original_length = all_original_lengths[video_idx]
        
        # Extract data for this video (remove padding)
        # Flatten if needed (batch dimension should be 1)
        video_predictions_raw = all_predictions[video_idx]
        video_labels_raw = all_labels[video_idx]
        video_confidence_raw = all_confidence_scores[video_idx]
        
        # Handle batch dimension
        if video_predictions_raw.ndim > 1:
            video_predictions_raw = video_predictions_raw.flatten()
        if video_labels_raw.ndim > 1:
            video_labels_raw = video_labels_raw.flatten()
        if video_confidence_raw.ndim > 1:
            video_confidence_raw = video_confidence_raw.flatten()
        
        video_predictions = video_predictions_raw[:original_length]
        video_labels = video_labels_raw[:original_length]
        video_confidence = video_confidence_raw[:original_length]
        
        # Calculate video duration
        video_duration = original_length / fps
        
        # Apply 90s constraint per video
        # If video < 90s, accept all frames (B案)
        if video_duration < min_total_duration:
            logger.debug(f"Video {video_name}: {video_duration:.1f}s < {min_total_duration:.1f}s → Accept all")
            optimal_predictions = np.ones_like(video_labels, dtype=int)
        else:
            # Find threshold that maximizes Recall within 90-200s constraint
            from sklearn.metrics import precision_recall_curve
            
            precisions, recalls, thresholds = precision_recall_curve(video_labels, video_confidence)
            
            # Pre-calculate approximate durations
            approx_durations = []
            for thresh in thresholds:
                candidate_predictions = (video_confidence >= thresh).astype(int)
                active_frames = np.sum(candidate_predictions)
                approx_duration = active_frames / fps
                approx_durations.append(approx_duration)
            
            approx_durations = np.array(approx_durations)
            
            # Find threshold that maximizes Recall within constraints
            best_threshold = thresholds[0] if len(thresholds) > 0 else 0.0
            best_recall = 0.0
            
            for i, (prec, rec, thresh, approx_dur) in enumerate(zip(precisions, recalls, thresholds, approx_durations)):
                # Hard constraints
                if approx_dur < min_total_duration * 0.95:
                    continue
                if approx_dur > hard_max_duration * 1.05:
                    continue
                
                # Maximize Recall
                if rec > best_recall:
                    best_recall = rec
                    best_threshold = thresh
            
            # If no threshold satisfies constraints, use lowest threshold
            if best_recall == 0.0:
                best_threshold = thresholds[0] if len(thresholds) > 0 else 0.0
            
            optimal_predictions = (video_confidence >= best_threshold).astype(int)
        
        # Extract clips and calculate duration
        clips = extract_clips_from_predictions(optimal_predictions, fps, min_clip_duration)
        pred_duration = sum(e - s for s, e in clips)
        
        # Calculate metrics for this video
        tp = np.sum((optimal_predictions == 1) & (video_labels == 1))
        fp = np.sum((optimal_predictions == 1) & (video_labels == 0))
        fn = np.sum((optimal_predictions == 0) & (video_labels == 1))
        tn = np.sum((optimal_predictions == 0) & (video_labels == 0))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        total_pred_duration += pred_duration
        
        video_metrics.append({
            'video_name': video_name,
            'duration': video_duration,
            'pred_duration': pred_duration,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    # Calculate overall metrics
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    
    # Average predicted duration per video
    avg_pred_duration = total_pred_duration / num_videos
    
    metrics = {
        'loss': avg_total_loss,
        'ce_loss': avg_ce_loss,
        'tv_loss': avg_tv_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'avg_pred_duration': avg_pred_duration,
        'video_metrics': video_metrics
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    random_state = config.get('random_state', 42)
    set_seed(random_state)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not config.get('cpu', False) else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Load datasets
    logger.info(f"\nLoading training data from {config['train_data_path']}")
    train_dataset = EnhancedCutSelectionDatasetFullVideo(config['train_data_path'])
    
    logger.info(f"\nLoading validation data from {config['val_data_path']}")
    val_dataset = EnhancedCutSelectionDatasetFullVideo(config['val_data_path'])
    
    # Create data loaders (batch_size=1 for variable length videos)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_fullvideo,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_fullvideo,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    model = EnhancedCutSelectionModel(
        audio_features=config['audio_features'],
        visual_features=config['visual_features'],
        temporal_features=config.get('temporal_features', 6),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # Calculate class weights from training data
    logger.info("Calculating class weights from training data...")
    train_active_count = 0
    train_inactive_count = 0
    
    for item in train_dataset:
        active_labels = item['active'].numpy()
        train_active_count += np.sum(active_labels == 1)
        train_inactive_count += np.sum(active_labels == 0)
    
    total_train_samples = train_active_count + train_inactive_count
    base_weight_active = total_train_samples / (2 * train_active_count) if train_active_count > 0 else 1.0
    base_weight_inactive = total_train_samples / (2 * train_inactive_count) if train_inactive_count > 0 else 1.0
    
    weight_active = base_weight_active * 3.0
    weight_inactive = base_weight_inactive * 1.0
    class_weights = torch.tensor([weight_inactive, weight_active], device=device)
    
    logger.info(f"  Train Active: {train_active_count:,} ({train_active_count/total_train_samples*100:.2f}%)")
    logger.info(f"  Train Inactive: {train_inactive_count:,} ({train_inactive_count/total_train_samples*100:.2f}%)")
    logger.info(f"  Class weights: [inactive={weight_inactive:.4f}, active={weight_active:.4f}]")
    
    # Loss and optimizer
    criterion = CombinedCutSelectionLoss(
        class_weights=class_weights,
        tv_weight=config.get('tv_weight', 0.0),
        label_smoothing=config.get('label_smoothing', 0.0),
        use_focal=config.get('use_focal_loss', False),
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        target_duration=config.get('max_total_duration', 180.0),
        duration_penalty_weight=config.get('duration_penalty_weight', 0.5),
        recall_reward_weight=config.get('recall_reward_weight', 5.0),
        fps=config.get('inference_fps', 10.0),
        min_clip_duration=config.get('min_clip_duration', 3.0)
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Training setup
    best_val_f1 = 0.0
    patience_counter = 0
    use_amp = config.get('use_amp', False) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    if scaler:
        logger.info("📊 Mixed Precision Training enabled")
    
    # Training history for plotting
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_recall': [],
        'val_precision': [],
        'val_accuracy': [],
        'val_specificity': [],
        'avg_pred_duration': []
    }
    
    # Training loop
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting training for {config['num_epochs']} epochs")
    logger.info(f"{'='*80}\n")
    
    for epoch in range(config['num_epochs']):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, criterion, device, scaler, max_grad_norm)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                   f"Train Loss: {train_losses['total_loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val F1: {val_metrics['f1']:.4f}, "
                   f"Val Recall: {val_metrics['recall']:.4f}, "
                   f"Avg Pred Duration: {val_metrics['avg_pred_duration']:.1f}s")
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['avg_pred_duration'].append(val_metrics['avg_pred_duration'])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, model_path)
            logger.info(f"✅ Best model saved (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 15):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Plot training history
    logger.info("\n📊 Generating training plots...")
    plot_training_history(history, checkpoint_dir, config)
    
    logger.info(f"\n✅ Training complete! Best F1: {best_val_f1:.4f}")
    logger.info(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")


def plot_training_history(history, checkpoint_dir, config):
    """Plot training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Full Video Training History', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. F1 Score
    ax = axes[0, 1]
    ax.plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2, marker='o')
    best_f1 = max(history['val_f1'])
    best_epoch = epochs[history['val_f1'].index(best_f1)]
    ax.axhline(y=best_f1, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_f1:.4f} (Epoch {best_epoch})')
    ax.set_title('F1 Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Recall
    ax = axes[0, 2]
    ax.plot(epochs, history['val_recall'], 'r-', label='Val Recall', linewidth=2, marker='s')
    best_recall = max(history['val_recall'])
    best_recall_epoch = epochs[history['val_recall'].index(best_recall)]
    ax.axhline(y=best_recall, color='orange', linestyle='--', alpha=0.5, label=f'Best: {best_recall:.4f} (Epoch {best_recall_epoch})')
    ax.set_title('Recall (採用の正確性)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Precision & Specificity
    ax = axes[1, 0]
    ax.plot(epochs, history['val_precision'], 'b-', label='Precision', linewidth=2, marker='o')
    ax.plot(epochs, history['val_specificity'], 'orange', label='Specificity', linewidth=2, marker='^')
    ax.set_title('Precision & Specificity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Accuracy
    ax = axes[1, 1]
    ax.plot(epochs, history['val_accuracy'], 'purple', label='Val Accuracy', linewidth=2, marker='d')
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Average Predicted Duration
    ax = axes[1, 2]
    ax.plot(epochs, history['avg_pred_duration'], 'green', label='Avg Pred Duration', linewidth=2, marker='o')
    
    # Add constraint lines
    min_duration = config.get('min_total_duration', 90.0)
    target_duration = config.get('max_total_duration', 180.0)
    hard_max_duration = config.get('hard_max_duration', 200.0)
    
    ax.axhline(y=min_duration, color='red', linestyle='--', alpha=0.5, label=f'Min: {min_duration:.0f}s')
    ax.axhline(y=target_duration, color='blue', linestyle='--', alpha=0.5, label=f'Target: {target_duration:.0f}s')
    ax.axhline(y=hard_max_duration, color='orange', linestyle='--', alpha=0.5, label=f'Hard Max: {hard_max_duration:.0f}s')
    
    # Color code the duration line
    for i in range(len(epochs) - 1):
        dur = history['avg_pred_duration'][i]
        if dur < min_duration:
            color = 'red'
        elif dur <= target_duration:
            color = 'green'
        elif dur <= hard_max_duration:
            color = 'orange'
        else:
            color = 'red'
        ax.plot(epochs[i:i+2], history['avg_pred_duration'][i:i+2], color=color, linewidth=3)
    
    ax.set_title('Average Predicted Duration (Per Video)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Duration (seconds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = checkpoint_dir / 'training_history.png'
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Training plot saved: {plot_path}")
    
    # Save history as CSV
    import pandas as pd
    df = pd.DataFrame(history)
    csv_path = checkpoint_dir / 'training_history.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"📊 Training history CSV saved: {csv_path}")
    
    plt.close(fig)


if __name__ == "__main__":
    main()
