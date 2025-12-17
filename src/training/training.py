"""
Training pipeline for Multi-Track Transformer
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from src.model.model import MultiTrackTransformer
from src.model.loss import MultiTrackLoss, GradientClipper, prepare_targets_from_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Training pipeline with logging and checkpointing"""
    
    def __init__(
        self,
        model: MultiTrackTransformer,
        loss_fn: MultiTrackLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_clipper: Optional[GradientClipper] = None,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize training pipeline
        
        Args:
            model: Multi-Track Transformer model
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler
            gradient_clipper: Optional gradient clipper
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clipper = gradient_clipper
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info(f"TrainingPipeline initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")
        logger.info(f"  Model parameters: {model.count_parameters():,}")

    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        loss_components = {
            'active': 0.0,
            'asset': 0.0,
            'scale': 0.0,
            'position': 0.0,
            'rotation': 0.0,
            'crop': 0.0
        }
        num_batches = 0
        
        # Modality utilization tracking
        modality_stats = {
            'total_samples': 0,
            'audio_available': 0,
            'visual_available': 0,
            'both_available': 0
        }
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Check if batch contains multimodal data
            is_multimodal = 'audio' in batch and 'visual' in batch
            
            if is_multimodal:
                # Multimodal forward pass
                audio = batch['audio'].to(self.device)
                visual = batch['visual'].to(self.device)
                track = batch['track'].to(self.device)
                padding_mask = batch['padding_mask'].to(self.device)
                modality_mask = batch.get('modality_mask', None)
                if modality_mask is not None:
                    modality_mask = modality_mask.to(self.device)
                
                # Track modality utilization
                batch_size = audio.shape[0]
                modality_stats['total_samples'] += batch_size
                if modality_mask is not None:
                    # Count samples with audio/visual available
                    audio_avail = modality_mask[:, :, 0].any(dim=1).sum().item()
                    visual_avail = modality_mask[:, :, 1].any(dim=1).sum().item()
                    both_avail = (modality_mask[:, :, 0].any(dim=1) & modality_mask[:, :, 1].any(dim=1)).sum().item()
                    
                    modality_stats['audio_available'] += audio_avail
                    modality_stats['visual_available'] += visual_avail
                    modality_stats['both_available'] += both_avail
                
                # Prepare targets from track data
                targets = prepare_targets_from_input(track)
                
                # Forward pass
                predictions = self.model(audio, visual, track, padding_mask, modality_mask)
            else:
                # Track-only forward pass (backward compatibility)
                sequences = batch['sequences'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # Prepare targets from input sequences
                targets = prepare_targets_from_input(sequences)
                
                # Forward pass
                predictions = self.model(sequences, masks)
                padding_mask = masks
            
            # Compute loss
            losses = self.loss_fn(predictions, targets, padding_mask)
            
            # Check for NaN or Inf
            if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                logger.warning(f"⚠️  NaN/Inf detected in batch {batch_idx}! Skipping...")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            if self.gradient_clipper is not None:
                grad_norm = self.gradient_clipper.clip(self.model)
                
                # Check for gradient explosion
                if grad_norm > 100.0:
                    logger.warning(f"⚠️  Large gradient norm detected: {grad_norm:.2f}")
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total'].item()
            for key in loss_components:
                loss_components[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_components = {k: v / num_batches for k, v in loss_components.items()} if num_batches > 0 else loss_components
        
        # Log modality utilization statistics
        if modality_stats['total_samples'] > 0:
            audio_pct = 100.0 * modality_stats['audio_available'] / modality_stats['total_samples']
            visual_pct = 100.0 * modality_stats['visual_available'] / modality_stats['total_samples']
            both_pct = 100.0 * modality_stats['both_available'] / modality_stats['total_samples']
            
            logger.info(f"\n📊 Modality Utilization (Epoch {epoch} Train):")
            logger.info(f"  Audio available: {audio_pct:.1f}% ({modality_stats['audio_available']}/{modality_stats['total_samples']})")
            logger.info(f"  Visual available: {visual_pct:.1f}% ({modality_stats['visual_available']}/{modality_stats['total_samples']})")
            logger.info(f"  Both available: {both_pct:.1f}% ({modality_stats['both_available']}/{modality_stats['total_samples']})")
        
        return {
            'total_loss': avg_loss,
            **avg_components,
            'modality_stats': modality_stats
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {
            'active': 0.0,
            'asset': 0.0,
            'scale': 0.0,
            'position': 0.0,
            'rotation': 0.0,
            'crop': 0.0
        }
        num_batches = 0
        
        # Modality utilization tracking
        modality_stats = {
            'total_samples': 0,
            'audio_available': 0,
            'visual_available': 0,
            'both_available': 0
        }
        
        # Progress bar
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                # Check if batch contains multimodal data
                is_multimodal = 'audio' in batch and 'visual' in batch
                
                if is_multimodal:
                    # Multimodal forward pass
                    audio = batch['audio'].to(self.device)
                    visual = batch['visual'].to(self.device)
                    track = batch['track'].to(self.device)
                    padding_mask = batch['padding_mask'].to(self.device)
                    modality_mask = batch.get('modality_mask', None)
                    if modality_mask is not None:
                        modality_mask = modality_mask.to(self.device)
                    
                    # Track modality utilization
                    batch_size = audio.shape[0]
                    modality_stats['total_samples'] += batch_size
                    if modality_mask is not None:
                        # Count samples with audio/visual available
                        audio_avail = modality_mask[:, :, 0].any(dim=1).sum().item()
                        visual_avail = modality_mask[:, :, 1].any(dim=1).sum().item()
                        both_avail = (modality_mask[:, :, 0].any(dim=1) & modality_mask[:, :, 1].any(dim=1)).sum().item()
                        
                        modality_stats['audio_available'] += audio_avail
                        modality_stats['visual_available'] += visual_avail
                        modality_stats['both_available'] += both_avail
                    
                    # Prepare targets from track data
                    targets = prepare_targets_from_input(track)
                    
                    # Forward pass
                    predictions = self.model(audio, visual, track, padding_mask, modality_mask)
                else:
                    # Track-only forward pass (backward compatibility)
                    sequences = batch['sequences'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    
                    # Prepare targets
                    targets = prepare_targets_from_input(sequences)
                    
                    # Forward pass
                    predictions = self.model(sequences, masks)
                    padding_mask = masks
                
                # Compute loss
                losses = self.loss_fn(predictions, targets, padding_mask)
                
                # Check for NaN or Inf
                if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                    logger.warning(f"⚠️  NaN/Inf detected in validation! Skipping batch...")
                    continue
                
                # Accumulate losses
                total_loss += losses['total'].item()
                for key in loss_components:
                    loss_components[key] += losses[key].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': losses['total'].item()})
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_components = {k: v / num_batches for k, v in loss_components.items()} if num_batches > 0 else loss_components
        
        # Log modality utilization statistics
        if modality_stats['total_samples'] > 0:
            audio_pct = 100.0 * modality_stats['audio_available'] / modality_stats['total_samples']
            visual_pct = 100.0 * modality_stats['visual_available'] / modality_stats['total_samples']
            both_pct = 100.0 * modality_stats['both_available'] / modality_stats['total_samples']
            
            logger.info(f"\n📊 Modality Utilization (Epoch {epoch} Val):")
            logger.info(f"  Audio available: {audio_pct:.1f}% ({modality_stats['audio_available']}/{modality_stats['total_samples']})")
            logger.info(f"  Visual available: {visual_pct:.1f}% ({modality_stats['visual_available']}/{modality_stats['total_samples']})")
            logger.info(f"  Both available: {both_pct:.1f}% ({modality_stats['both_available']}/{modality_stats['total_samples']})")
        
        return {
            'total_loss': avg_loss,
            **avg_components,
            'modality_stats': modality_stats
        }

    
    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': {
                'audio_features': self.model.audio_features,
                'visual_features': self.model.visual_features,
                'track_features': self.model.track_features,
                'd_model': self.model.d_model,
                'nhead': self.model.nhead,
                'num_layers': self.model.num_layers,
                'enable_multimodal': self.model.enable_multimodal,
                'fusion_type': self.model.fusion_type,
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        logger.info(f"✅ Checkpoint loaded from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_every: int = 5,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs (None = disabled)
        """
        logger.info("\n" + "="*80)
        logger.info("Starting Training")
        logger.info("="*80)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Device: {self.device}")
        logger.info("="*80 + "\n")
        
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # Check for best model
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                self.best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch summary
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{num_epochs} Summary (Time: {epoch_time:.2f}s)")
            logger.info(f"{'='*80}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            logger.info(f"\nTrain Loss: {train_metrics['total_loss']:.4f}")
            logger.info(f"  - Active: {train_metrics['active']:.4f}")
            logger.info(f"  - Asset: {train_metrics['asset']:.4f}")
            logger.info(f"  - Scale: {train_metrics['scale']:.4f}")
            logger.info(f"  - Position: {train_metrics['position']:.4f}")
            logger.info(f"  - Rotation: {train_metrics['rotation']:.4f}")
            logger.info(f"  - Crop: {train_metrics['crop']:.4f}")
            logger.info(f"\nVal Loss: {val_metrics['total_loss']:.4f}")
            logger.info(f"  - Active: {val_metrics['active']:.4f}")
            logger.info(f"  - Asset: {val_metrics['asset']:.4f}")
            logger.info(f"  - Scale: {val_metrics['scale']:.4f}")
            logger.info(f"  - Position: {val_metrics['position']:.4f}")
            logger.info(f"  - Rotation: {val_metrics['rotation']:.4f}")
            logger.info(f"  - Crop: {val_metrics['crop']:.4f}")
            
            if is_best:
                logger.info(f"\n🏆 New best model! (Val Loss: {val_metrics['total_loss']:.4f})")
            else:
                logger.info(f"\nBest Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
            
            logger.info(f"{'='*80}\n")
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best)
            
            # Early stopping
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                logger.info(f"\n⚠️  Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
        
        logger.info("\n" + "="*80)
        logger.info("Training Complete!")
        logger.info("="*80)
        logger.info(f"Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
        logger.info(f"Final Val Loss: {val_metrics['total_loss']:.4f}")
        logger.info("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    import yaml
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Multi-Track Transformer')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loading config from: {args.config}")
    
    from src.model.model import create_model
    from src.model.loss import create_optimizer, create_scheduler
    from src.training.multimodal_dataset import create_multimodal_dataloaders
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() and not config.get('cpu', False) else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(
        input_features=config.get('track_features', 240),
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1),
        num_tracks=config.get('num_tracks', 20),
        max_asset_classes=config.get('max_asset_classes', 10),
        enable_multimodal=config.get('enable_multimodal', False),
        audio_features=config.get('audio_features', 17),
        visual_features=config.get('visual_features', 522),
        fusion_type=config.get('fusion_type', 'gated')
    )
    
    # Create loss function
    loss_fn = MultiTrackLoss(
        active_weight=config.get('active_weight', 1.0),
        asset_weight=config.get('asset_weight', 1.0),
        scale_weight=config.get('scale_weight', 1.0),
        position_weight=config.get('position_weight', 1.0),
        rotation_weight=config.get('rotation_weight', 1.0),
        crop_weight=config.get('crop_weight', 1.0),
        ignore_inactive=config.get('ignore_inactive', True)
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model, 
        learning_rate=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-5),
        optimizer_type=config.get('optimizer', 'adam')
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer, 
        scheduler_type=config.get('scheduler', 'cosine'),
        num_epochs=config.get('num_epochs', 100),
        warmup_epochs=config.get('warmup_epochs', 2),
        min_lr=config.get('min_lr', 1e-6)
    )
    
    # Create gradient clipper
    gradient_clipper = GradientClipper(max_norm=config.get('grad_clip', 1.0))
    
    # Create training pipeline
    pipeline = TrainingPipeline(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clipper=gradient_clipper,
        device=device,
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints')
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_multimodal_dataloaders(
        train_npz=config.get('train_data', 'preprocessed_data/train_sequences.npz'),
        val_npz=config.get('val_data', 'preprocessed_data/val_sequences.npz'),
        features_dir=config.get('features_dir', 'data/processed/input_features'),
        batch_size=config.get('batch_size', 8),
        num_workers=config.get('num_workers', 0)
    )
    
    # Train
    logger.info(f"Starting training for {config.get('num_epochs', 100)} epochs...")
    pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 100),
        save_every=config.get('save_every', 5)
    )
    
    logger.info("\n✅ Training complete!")
