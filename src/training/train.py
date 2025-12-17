"""
Main training script for Multi-Track Transformer
"""
import argparse
import yaml
import torch
import logging
from pathlib import Path

from src.model.model import create_model, MultimodalTransformer
from src.model.loss import MultiTrackLoss, create_optimizer, create_scheduler, GradientClipper
from src.training.dataset import create_dataloaders
from src.training.multimodal_dataset import create_multimodal_dataloaders
from src.training.training import TrainingPipeline
from src.model.model_persistence import save_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function"""
    
    # Load configuration if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    else:
        config = vars(args)
    
    logger.info("\n" + "="*80)
    logger.info("Multi-Track Transformer Training")
    logger.info("="*80)
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*80 + "\n")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not config.get('cpu', False) else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    
    # Check if multimodal mode is enabled
    enable_multimodal = config.get('enable_multimodal', False)
    
    if enable_multimodal:
        logger.info("Multimodal mode enabled - loading video features")
        logger.info(f"  Features directory: {config.get('features_dir', 'input_features')}")
        logger.info(f"  Fusion type: {config.get('fusion_type', 'gated')}")
        logger.info(f"  Audio features: {config.get('audio_features', 17)}")
        logger.info(f"  Visual features: {config.get('visual_features', 522)}")
        logger.info(f"  Speech text embedding: 6 dimensions")
        logger.info(f"  Telop text embedding: 6 dimensions")
        
        train_loader, val_loader = create_multimodal_dataloaders(
            train_npz=config['train_data'],
            val_npz=config['val_data'],
            features_dir=config.get('features_dir', 'input_features'),
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0),
            enable_multimodal=True
        )
    else:
        logger.info("Track-only mode - using existing dataloaders")
        train_loader, val_loader = create_dataloaders(
            train_npz=config['train_data'],
            val_npz=config['val_data'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0)
        )
    
    # Create model
    logger.info("\nCreating model...")
    
    if enable_multimodal:
        logger.info("Creating multimodal transformer model")
        model = MultimodalTransformer(
            audio_features=config.get('audio_features', 17),
            visual_features=config.get('visual_features', 522),
            track_features=config.get('track_features', 180),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config.get('dropout', 0.1),
            num_tracks=config.get('num_tracks', 20),
            max_asset_classes=config.get('max_asset_classes', 10),
            enable_multimodal=True,
            fusion_type=config.get('fusion_type', 'gated')
        )
        num_params = model.count_parameters()
        logger.info(f"Model created with {num_params:,} trainable parameters")
    else:
        logger.info("Creating track-only transformer model")
        model = create_model(
            input_features=config.get('input_features', 180),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config.get('dropout', 0.1),
            num_tracks=config.get('num_tracks', 20),
            max_asset_classes=config.get('max_asset_classes', 10)
        )
    
    # Create loss function
    logger.info("\nCreating loss function...")
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
    logger.info("\nCreating optimizer...")
    optimizer = create_optimizer(
        model=model,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5),
        optimizer_type=config.get('optimizer', 'adam')
    )
    
    # Create scheduler
    logger.info("\nCreating scheduler...")
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_type=config.get('scheduler', 'cosine'),
        num_epochs=config['num_epochs'],
        warmup_epochs=config.get('warmup_epochs', 5),
        min_lr=config.get('min_lr', 1e-6)
    )
    
    # Create gradient clipper
    gradient_clipper = GradientClipper(
        max_norm=config.get('grad_clip', 1.0)
    )
    
    # Create training pipeline
    logger.info("\nCreating training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clipper=gradient_clipper,
        device=device,
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints')
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if config.get('resume'):
        logger.info(f"\nResuming from checkpoint: {config['resume']}")
        start_epoch = pipeline.load_checkpoint(config['resume'])
    
    # Train
    logger.info("\nStarting training...")
    pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        save_every=config.get('save_every', 5),
        early_stopping_patience=config.get('early_stopping_patience')
    )
    
    # Save final model
    logger.info("\nSaving final model...")
    save_model(
        model=model,
        save_path=Path(config.get('checkpoint_dir', 'checkpoints')) / 'final_model.pth',
        metadata={
            'config': config,
            'best_val_loss': pipeline.best_val_loss,
            'best_epoch': pipeline.best_epoch
        },
        optimizer=optimizer,
        metrics=pipeline.history
    )
    
    logger.info("\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Track Transformer")
    
    # Data
    parser.add_argument('--train_data', type=str, default='preprocessed_data/train_sequences.npz',
                       help='Path to training data')
    parser.add_argument('--val_data', type=str, default='preprocessed_data/val_sequences.npz',
                       help='Path to validation data')
    parser.add_argument('--features_dir', type=str, default='input_features',
                       help='Directory containing video feature CSV files')
    
    # Multimodal settings
    parser.add_argument('--enable_multimodal', type=lambda x: x.lower() == 'true', default=None,
                       help='Enable multimodal training with video features')
    parser.add_argument('--fusion_type', type=str, default='gated',
                       choices=['concat', 'add', 'gated', 'attention'],
                       help='Modality fusion strategy')
    parser.add_argument('--audio_features', type=int, default=None,
                       help='Number of audio features')
    parser.add_argument('--visual_features', type=int, default=None,
                       help='Number of visual features')
    parser.add_argument('--track_features', type=int, default=None,
                       help='Number of track features')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                       help='Number of encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Scheduler type')
    
    # Loss weights
    parser.add_argument('--active_weight', type=float, default=1.0,
                       help='Weight for active loss')
    parser.add_argument('--asset_weight', type=float, default=1.0,
                       help='Weight for asset loss')
    parser.add_argument('--scale_weight', type=float, default=1.0,
                       help='Weight for scale loss')
    parser.add_argument('--position_weight', type=float, default=1.0,
                       help='Weight for position loss')
    parser.add_argument('--rotation_weight', type=float, default=1.0,
                       help='Weight for rotation loss')
    parser.add_argument('--crop_weight', type=float, default=1.0,
                       help='Weight for crop loss')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU training')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Early stopping patience (None = disabled)')
    
    args = parser.parse_args()
    
    main(args)
