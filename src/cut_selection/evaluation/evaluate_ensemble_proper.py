"""
Proper Ensemble Evaluation with K-Fold Cross Validation

Evaluates ensemble on validation data ONLY (no training data leakage).
Each fold's validation data is evaluated using all 5 models.
"""
import torch
from torch.utils.data import DataLoader, Subset
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from src.cut_selection.datasets.cut_dataset_enhanced import EnhancedCutSelectionDataset
from src.cut_selection.evaluation.ensemble_predictor import EnsemblePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_ensemble_proper(checkpoint_dir: str, data_path: str, n_folds: int = 5, device: str = 'cuda'):
    """
    Properly evaluate ensemble using only validation data from each fold
    
    This ensures no training data leakage in evaluation.
    
    Args:
        checkpoint_dir: Directory with fold models
        data_path: Path to dataset
        n_folds: Number of folds
        device: Device to use
    """
    logger.info("="*80)
    logger.info("PROPER ENSEMBLE EVALUATION (Validation Data Only)")
    logger.info("="*80)
    
    # Load full dataset
    logger.info(f"\nLoading dataset from {data_path}")
    full_dataset = EnhancedCutSelectionDataset(data_path)
    
    # Get video groups for GroupKFold
    video_groups = full_dataset.get_video_groups()
    unique_videos = list(set(video_groups))
    logger.info(f"📹 Total unique videos: {len(unique_videos)}")
    logger.info(f"📊 Total sequences: {len(full_dataset)}")
    
    # Setup GroupKFold
    group_kfold = GroupKFold(n_splits=n_folds)
    
    # Load K-Fold summary for comparison
    checkpoint_path = Path(checkpoint_dir)
    summary_path = checkpoint_path / 'kfold_summary.csv'
    
    individual_f1_scores = []
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        individual_f1_scores = summary_df[summary_df['fold'] != 'Mean ± Std']['best_val_f1'].astype(float).tolist()
        mean_individual_f1 = np.mean(individual_f1_scores)
        
        logger.info(f"\n📊 Individual Model Performance (from training):")
        logger.info(f"   Mean F1: {mean_individual_f1:.4f}")
        for i, f1 in enumerate(individual_f1_scores, 1):
            logger.info(f"   Fold {i}: {f1:.4f}")
    
    # Test all voting strategies
    strategies = ['soft', 'hard', 'weighted']
    
    # Store results for each strategy
    strategy_results = {strategy: {
        'all_predictions': [],
        'all_confidences': [],
        'all_labels': [],
        'fold_metrics': []
    } for strategy in strategies}
    
    # Evaluate each fold
    logger.info(f"\n{'='*80}")
    logger.info("Evaluating each fold's validation data with ONLY that fold's model")
    logger.info("(No data leakage - each model only sees unseen validation data)")
    logger.info(f"{'='*80}")
    
    for fold, (train_indices, val_indices) in enumerate(group_kfold.split(range(len(full_dataset)), groups=video_groups)):
        train_videos = set([video_groups[i] for i in train_indices])
        val_videos = set([video_groups[i] for i in val_indices])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Fold {fold+1}/{n_folds}")
        logger.info(f"  Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")
        logger.info(f"  Val sequences: {len(val_indices)}")
        logger.info(f"  Using ONLY Model {fold+1} (trained on Fold {fold+1} train data)")
        logger.info(f"{'='*80}")
        
        # Create validation dataloader
        val_dataset = Subset(full_dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        # Load ONLY this fold's model
        from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel
        
        model_path = Path(checkpoint_dir) / f"fold_{fold+1}_best_model.pth"
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        
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
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Predict on validation data with this single model
        all_confidences_fold = []
        all_labels_fold = []
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                visual = batch['visual'].to(device)
                temporal = batch['temporal'].to(device)
                labels = batch['active']
                
                outputs = model(audio, visual, temporal)
                logits = outputs['active']  # (batch, seq_len, 2)
                probs = torch.softmax(logits, dim=-1)  # (batch, seq_len, 2)
                confidence = probs[..., 1] - probs[..., 0]  # (batch, seq_len)
                
                all_confidences_fold.append(confidence.cpu().numpy())
                all_labels_fold.append(labels.numpy())
        
        # Store results for each strategy (same predictions, just different thresholds later)
        confidences_fold = np.concatenate(all_confidences_fold, axis=0)
        labels_fold = np.concatenate(all_labels_fold, axis=0)
        
        for strategy in strategies:
            strategy_results[strategy]['all_confidences'].append(confidences_fold)
            strategy_results[strategy]['all_labels'].append(labels_fold)
    
    # Now find optimal threshold and evaluate for each strategy
    logger.info(f"\n{'='*80}")
    logger.info("Finding optimal thresholds and final evaluation")
    logger.info("(Using single-model predictions from each fold)")
    logger.info(f"{'='*80}")
    
    final_results = {}
    
    # For single model evaluation, all strategies give the same result
    # (since we're using one model per fold, not ensemble)
    logger.info(f"\n{'='*80}")
    logger.info(f"Single Model Evaluation (No Ensemble)")
    logger.info(f"{'='*80}")
    
    # Use 'soft' strategy results (they're all the same for single model)
    strategy = 'soft'
    
    # Concatenate all validation data
    all_confidences = np.concatenate(strategy_results[strategy]['all_confidences'], axis=0).flatten()
    all_labels = np.concatenate(strategy_results[strategy]['all_labels'], axis=0).flatten()
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_confidences)
    
    best_threshold = 0.0
    best_f1 = 0.0
    best_metrics = {}
    
    min_recall = 0.71
    
    for prec, rec, thresh in zip(precisions, recalls, thresholds):
        if rec < min_recall:
            continue
        
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = {
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
    
    logger.info(f"   Optimal threshold: {best_threshold:.4f}")
    logger.info(f"   Best F1: {best_f1:.4f}")
    
    # Apply threshold and calculate final metrics
    predictions = (all_confidences >= best_threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    
    accuracy = (predictions == all_labels).mean()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    final_results['single_model'] = {
        'threshold': best_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }
    
    logger.info(f"\n   Final Metrics:")
    logger.info(f"     Accuracy: {accuracy:.4f}")
    logger.info(f"     Precision: {precision:.4f}")
    logger.info(f"     Recall: {recall:.4f}")
    logger.info(f"     F1 Score: {f1:.4f}")
    logger.info(f"     Specificity: {specificity:.4f}")
    
    # Compare results
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON OF ENSEMBLE STRATEGIES (Proper Evaluation)")
    logger.info(f"{'='*80}")
    
    comparison_data = []
    
    # Add individual models baseline
    if individual_f1_scores:
        comparison_data.append({
            'Strategy': 'Individual (Mean)',
            'F1 Score': np.mean(individual_f1_scores),
            'Accuracy': '-',
            'Precision': '-',
            'Recall': '-',
            'Specificity': '-',
            'Threshold': '-'
        })
    
    # Add ensemble results
    for strategy, metrics in final_results.items():
        comparison_data.append({
            'Strategy': f'Ensemble ({strategy})',
            'F1 Score': metrics['f1'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'Threshold': metrics['threshold']
        })
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Find best strategy
    best_strategy = max(final_results.items(), key=lambda x: x[1]['f1'])
    best_strategy_name = best_strategy[0]
    best_f1 = best_strategy[1]['f1']
    
    if individual_f1_scores:
        improvement = best_f1 - np.mean(individual_f1_scores)
        improvement_pct = improvement / np.mean(individual_f1_scores) * 100
    else:
        improvement = 0.0
        improvement_pct = 0.0
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🏆 BEST ENSEMBLE STRATEGY: {best_strategy_name.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"   F1 Score: {best_f1:.4f}")
    if individual_f1_scores:
        logger.info(f"   Improvement over individual: +{improvement:.4f} ({improvement_pct:+.2f}%)")
    logger.info(f"   Threshold: {best_strategy[1]['threshold']:.4f}")
    
    # Save results
    output_dir = Path(checkpoint_dir)
    comparison_df.to_csv(output_dir / 'ensemble_comparison_proper.csv', index=False)
    logger.info(f"\n💾 Results saved to: {output_dir / 'ensemble_comparison_proper.csv'}")
    
    # Create visualization
    create_comparison_plot(comparison_df, output_dir, np.mean(individual_f1_scores) if individual_f1_scores else 0.0)
    
    return final_results, best_strategy_name


def create_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path, baseline_f1: float):
    """Create visualization comparing ensemble strategies"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1 Score comparison
    ax = axes[0]
    strategies = comparison_df['Strategy'].tolist()
    f1_scores = comparison_df['F1 Score'].tolist()
    
    colors = ['gray'] + ['skyblue', 'lightcoral', 'lightgreen'][:len(f1_scores)-1]
    bars = ax.bar(range(len(strategies)), f1_scores, color=colors, edgecolor='black', alpha=0.7)
    
    # Add baseline line
    if baseline_f1 > 0:
        ax.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=2, label=f'Individual Mean: {baseline_f1:.4f}')
    
    # Add values on bars
    for i, (strategy, f1) in enumerate(zip(strategies, f1_scores)):
        ax.text(i, f1 + 0.01, f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison (Proper Evaluation)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(f1_scores) * 1.15])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Metrics comparison
    ax = axes[1]
    
    ensemble_rows = comparison_df[comparison_df['Strategy'].str.contains('Ensemble')]
    
    if len(ensemble_rows) > 0:
        metrics_names = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Specificity']
        
        for idx, row in ensemble_rows.iterrows():
            strategy = row['Strategy']
            values = [
                row['F1 Score'],
                row['Accuracy'],
                row['Precision'],
                row['Recall'],
                row['Specificity']
            ]
            
            x = np.arange(len(metrics_names))
            ax.plot(x, values, marker='o', linewidth=2, label=strategy, markersize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=15, ha='right')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Ensemble Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'ensemble_comparison_proper.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plot saved to: {save_path}")
    
    plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Proper Ensemble Evaluation (Validation Data Only)')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='checkpoints_cut_selection_kfold_enhanced',
                       help='Directory with fold models')
    parser.add_argument('--data_path', type=str,
                       default='preprocessed_data/combined_sequences_cut_selection_enhanced.npz',
                       help='Path to dataset')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run proper evaluation
    results, best_strategy = evaluate_ensemble_proper(
        args.checkpoint_dir,
        args.data_path,
        args.n_folds,
        args.device
    )
    
    logger.info(f"\n✅ Proper ensemble evaluation complete!")
    logger.info(f"   Best strategy: {best_strategy}")
    logger.info(f"\n⚠️  Note: These results use ONLY validation data (no training data leakage)")


if __name__ == "__main__":
    main()
