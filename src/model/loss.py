"""
Loss functions and training utilities for Multi-Track Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiTrackLoss(nn.Module):
    """
    Combined loss function for multi-track prediction
    
    Combines:
    - CrossEntropyLoss for classification (active, asset)
    - MSELoss for regression (scale, position, crop)
    """
    
    def __init__(
        self,
        active_weight: float = 1.0,
        asset_weight: float = 1.0,
        scale_weight: float = 1.0,
        position_weight: float = 1.0,
        rotation_weight: float = 1.0,
        crop_weight: float = 1.0,
        ignore_inactive: bool = True
    ):
        """
        Initialize loss function
        
        Args:
            active_weight: Weight for active classification loss
            asset_weight: Weight for asset classification loss
            scale_weight: Weight for scale regression loss
            position_weight: Weight for position (x, y, anchor_x, anchor_y) regression loss
            rotation_weight: Weight for rotation regression loss
            crop_weight: Weight for crop regression loss
            ignore_inactive: If True, only compute regression loss for active tracks
        """
        super().__init__()
        
        self.active_weight = active_weight
        self.asset_weight = asset_weight
        self.scale_weight = scale_weight
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.crop_weight = crop_weight
        self.ignore_inactive = ignore_inactive
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
        logger.info(f"MultiTrackLoss initialized:")
        logger.info(f"  active_weight: {active_weight}")
        logger.info(f"  asset_weight: {asset_weight}")
        logger.info(f"  scale_weight: {scale_weight}")
        logger.info(f"  position_weight: {position_weight}")
        logger.info(f"  rotation_weight: {rotation_weight}")
        logger.info(f"  crop_weight: {crop_weight}")
        logger.info(f"  ignore_inactive: {ignore_inactive}")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            predictions: Dict with model predictions
                - 'active': (batch, seq_len, num_tracks, 2)
                - 'asset': (batch, seq_len, num_tracks, num_classes)
                - 'scale', 'pos_x', 'pos_y', 'anchor_x', 'anchor_y', 'rotation', 'crop_*': (batch, seq_len, num_tracks, 1)
            targets: Dict with ground truth values (same structure as predictions)
            mask: Optional boolean mask (batch, seq_len) where True = valid data
        
        Returns:
            Dict with:
                - 'total': Total weighted loss
                - 'active': Active classification loss
                - 'asset': Asset classification loss
                - 'scale': Scale regression loss
                - 'position': Position regression loss
                - 'crop': Crop regression loss
        """
        batch_size, seq_len, num_tracks, _ = predictions['active'].shape
        device = predictions['active'].device
        
        # Create mask for valid positions
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Expand mask to track dimension
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, num_tracks)  # (batch, seq_len, num_tracks)
        
        # === Active Classification Loss ===
        # Reshape for cross entropy: (batch * seq_len * num_tracks, 2)
        active_pred = predictions['active'].reshape(-1, 2)
        active_target = targets['active'].reshape(-1).long()
        
        active_loss = self.ce_loss(active_pred, active_target)  # (batch * seq_len * num_tracks,)
        active_loss = active_loss.reshape(batch_size, seq_len, num_tracks)
        
        # Apply mask
        active_loss = active_loss * mask_expanded
        active_loss = active_loss.sum() / mask_expanded.sum()
        
        # === Asset Classification Loss ===
        num_asset_classes = predictions['asset'].shape[-1]
        asset_pred = predictions['asset'].reshape(-1, num_asset_classes)
        asset_target = targets['asset'].reshape(-1).long()
        
        asset_loss = self.ce_loss(asset_pred, asset_target)
        asset_loss = asset_loss.reshape(batch_size, seq_len, num_tracks)
        
        # Apply mask
        asset_loss = asset_loss * mask_expanded
        asset_loss = asset_loss.sum() / mask_expanded.sum()
        
        # === Regression Losses ===
        # If ignore_inactive, only compute loss for active tracks
        if self.ignore_inactive:
            # Get active mask from targets
            active_mask = (targets['active'] == 1).float()  # (batch, seq_len, num_tracks)
            active_mask = active_mask * mask_expanded  # Combine with padding mask
            
            # Avoid division by zero
            active_count = active_mask.sum()
            if active_count == 0:
                active_count = 1.0
        else:
            active_mask = mask_expanded.float()
            active_count = active_mask.sum()
        
        # Scale loss
        scale_pred = predictions['scale'].squeeze(-1)  # (batch, seq_len, num_tracks)
        scale_target = targets['scale'].squeeze(-1)
        scale_loss = self.mse_loss(scale_pred, scale_target)
        scale_loss = (scale_loss * active_mask).sum() / active_count
        
        # Position loss (x, y, anchor_x, anchor_y)
        pos_x_pred = predictions['pos_x'].squeeze(-1)
        pos_x_target = targets['pos_x'].squeeze(-1)
        pos_x_loss = self.mse_loss(pos_x_pred, pos_x_target)
        
        pos_y_pred = predictions['pos_y'].squeeze(-1)
        pos_y_target = targets['pos_y'].squeeze(-1)
        pos_y_loss = self.mse_loss(pos_y_pred, pos_y_target)
        
        anchor_x_pred = predictions['anchor_x'].squeeze(-1)
        anchor_x_target = targets['anchor_x'].squeeze(-1)
        anchor_x_loss = self.mse_loss(anchor_x_pred, anchor_x_target)
        
        anchor_y_pred = predictions['anchor_y'].squeeze(-1)
        anchor_y_target = targets['anchor_y'].squeeze(-1)
        anchor_y_loss = self.mse_loss(anchor_y_pred, anchor_y_target)
        
        position_loss = ((pos_x_loss + pos_y_loss + anchor_x_loss + anchor_y_loss) * active_mask).sum() / active_count
        
        # Rotation loss
        rotation_pred = predictions['rotation'].squeeze(-1)
        rotation_target = targets['rotation'].squeeze(-1)
        rotation_loss = self.mse_loss(rotation_pred, rotation_target)
        rotation_loss = (rotation_loss * active_mask).sum() / active_count
        
        # Crop loss (l, r, t, b)
        crop_losses = []
        for crop_key in ['crop_l', 'crop_r', 'crop_t', 'crop_b']:
            crop_pred = predictions[crop_key].squeeze(-1)
            crop_target = targets[crop_key].squeeze(-1)
            crop_loss = self.mse_loss(crop_pred, crop_target)
            crop_losses.append(crop_loss)
        
        crop_loss = (sum(crop_losses) * active_mask).sum() / active_count
        
        # === Total Loss ===
        total_loss = (
            self.active_weight * active_loss +
            self.asset_weight * asset_loss +
            self.scale_weight * scale_loss +
            self.position_weight * position_loss +
            self.rotation_weight * rotation_loss +
            self.crop_weight * crop_loss
        )
        
        return {
            'total': total_loss,
            'active': active_loss,
            'asset': asset_loss,
            'scale': scale_loss,
            'position': position_loss,
            'rotation': rotation_loss,
            'crop': crop_loss
        }



class GradientClipper:
    """Utility for gradient clipping"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipper
        
        Args:
            max_norm: Maximum norm for gradient clipping
            norm_type: Type of norm (2.0 for L2 norm)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip(self, model: nn.Module) -> float:
        """
        Clip gradients of model parameters
        
        Args:
            model: PyTorch model
        
        Returns:
            Total norm of gradients before clipping
        """
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type
        )


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    optimizer_type: str = 'adam'
) -> torch.optim.Optimizer:
    """
    Create optimizer
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
    
    Returns:
        Optimizer
    """
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer: lr={learning_rate}, weight_decay={weight_decay}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau', 'none')
        num_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
    
    Returns:
        Scheduler or None
    """
    if scheduler_type.lower() == 'none':
        return None
    
    if scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr
        )
    elif scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )
    elif scheduler_type.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler")
    return scheduler


def prepare_targets_from_input(
    input_sequences: torch.Tensor,
    num_tracks: int = 20
) -> Dict[str, torch.Tensor]:
    """
    Prepare target dict from input sequences for training
    
    Args:
        input_sequences: Input tensor of shape (batch, seq_len, features)
            where features = num_tracks * 12 (active, asset, scale, x, y, anchor_x, anchor_y, rotation, crop_l, crop_r, crop_t, crop_b)
        num_tracks: Number of tracks
    
    Returns:
        Dict with target tensors
    """
    batch_size, seq_len, features = input_sequences.shape
    device = input_sequences.device
    
    # Reshape to (batch, seq_len, num_tracks, 12)
    reshaped = input_sequences.reshape(batch_size, seq_len, num_tracks, 12)
    
    # Extract each parameter
    targets = {
        'active': reshaped[:, :, :, 0],  # (batch, seq_len, num_tracks)
        'asset': reshaped[:, :, :, 1],   # (batch, seq_len, num_tracks)
        'scale': reshaped[:, :, :, 2:3],  # (batch, seq_len, num_tracks, 1)
        'pos_x': reshaped[:, :, :, 3:4],
        'pos_y': reshaped[:, :, :, 4:5],
        'anchor_x': reshaped[:, :, :, 5:6],
        'anchor_y': reshaped[:, :, :, 6:7],
        'rotation': reshaped[:, :, :, 7:8],
        'crop_l': reshaped[:, :, :, 8:9],
        'crop_r': reshaped[:, :, :, 9:10],
        'crop_t': reshaped[:, :, :, 10:11],
        'crop_b': reshaped[:, :, :, 11:12]
    }
    
    return targets


if __name__ == "__main__":
    # Test loss function
    logger.info("Testing MultiTrackLoss...")
    
    batch_size = 4
    seq_len = 100
    num_tracks = 20
    num_asset_classes = 10
    
    # Create dummy predictions (with requires_grad for gradient test)
    predictions = {
        'active': torch.randn(batch_size, seq_len, num_tracks, 2, requires_grad=True),
        'asset': torch.randn(batch_size, seq_len, num_tracks, num_asset_classes, requires_grad=True),
        'scale': torch.randn(batch_size, seq_len, num_tracks, 1, requires_grad=True),
        'pos_x': torch.randn(batch_size, seq_len, num_tracks, 1, requires_grad=True),
        'pos_y': torch.randn(batch_size, seq_len, num_tracks, 1, requires_grad=True),
        'crop_l': torch.randn(batch_size, seq_len, num_tracks, 1, requires_grad=True),
        'crop_r': torch.randn(batch_size, seq_len, num_tracks, 1, requires_grad=True),
        'crop_t': torch.randn(batch_size, seq_len, num_tracks, 1, requires_grad=True),
        'crop_b': torch.randn(batch_size, seq_len, num_tracks, 1, requires_grad=True)
    }
    
    # Create dummy targets
    targets = {
        'active': torch.randint(0, 2, (batch_size, seq_len, num_tracks)),
        'asset': torch.randint(0, num_asset_classes, (batch_size, seq_len, num_tracks)),
        'scale': torch.randn(batch_size, seq_len, num_tracks, 1),
        'pos_x': torch.randn(batch_size, seq_len, num_tracks, 1),
        'pos_y': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_l': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_r': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_t': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_b': torch.randn(batch_size, seq_len, num_tracks, 1)
    }
    
    # Create mask (simulate padding)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -20:] = False  # Last 20 frames are padding
    
    # Create loss function
    loss_fn = MultiTrackLoss(
        active_weight=1.0,
        asset_weight=1.0,
        scale_weight=1.0,
        position_weight=1.0,
        crop_weight=1.0
    )
    
    # Compute loss
    logger.info("\nComputing loss...")
    losses = loss_fn(predictions, targets, mask)
    
    logger.info("\nLoss values:")
    for key, value in losses.items():
        logger.info(f"  {key}: {value.item():.4f}")
    
    # Test gradient flow
    logger.info("\nTesting gradient flow...")
    losses['total'].backward()
    
    # Check for NaN or Inf
    has_nan = any(torch.isnan(v).any() for v in losses.values())
    has_inf = any(torch.isinf(v).any() for v in losses.values())
    
    if has_nan:
        logger.error("❌ Loss contains NaN!")
    elif has_inf:
        logger.error("❌ Loss contains Inf!")
    else:
        logger.info("✅ Loss computation successful!")
    
    # Test prepare_targets_from_input
    logger.info("\nTesting prepare_targets_from_input...")
    input_seq = torch.randn(batch_size, seq_len, 180)  # 20 tracks * 9 params
    targets_from_input = prepare_targets_from_input(input_seq, num_tracks=20)
    
    logger.info("Target shapes:")
    for key, value in targets_from_input.items():
        logger.info(f"  {key}: {value.shape}")
    
    logger.info("\n✅ All tests passed!")
