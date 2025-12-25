"""
Custom Loss Functions for Cut Selection

Includes losses for temporal smoothness and class imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for temporal smoothness
    
    Penalizes rapid changes in predictions over time.
    This prevents "chattering" (flickering between 0 and 1).
    
    Example:
        Good:  [0, 0, 0, 1, 1, 1, 0, 0]  (smooth transitions)
        Bad:   [0, 1, 0, 1, 0, 1, 0, 1]  (chattering)
    """
    
    def __init__(self, weight: float = 0.1):
        """
        Initialize Total Variation Loss
        
        Args:
            weight: Weight for TV loss (0.01-0.5 recommended)
                   Higher = smoother predictions, but may miss short clips
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate total variation loss
        
        Args:
            predictions: Predicted probabilities (batch, seq_len, 2)
                        or logits (batch, seq_len, 2)
        
        Returns:
            TV loss (scalar)
        """
        # Get probability of active class (class 1)
        if predictions.size(-1) == 2:
            # If logits, convert to probabilities
            probs = F.softmax(predictions, dim=-1)[..., 1]  # (batch, seq_len)
        else:
            probs = predictions
        
        # Calculate differences between adjacent timesteps
        # diff[t] = |prob[t+1] - prob[t]|
        diff = torch.abs(probs[:, 1:] - probs[:, :-1])  # (batch, seq_len-1)
        
        # Sum over time and average over batch
        tv_loss = diff.mean()
        
        return self.weight * tv_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal Consistency Loss
    
    Encourages predictions to be consistent within a local window.
    Similar to TV loss but uses a window-based approach.
    """
    
    def __init__(self, window_size: int = 5, weight: float = 0.1):
        """
        Initialize Temporal Consistency Loss
        
        Args:
            window_size: Size of local window (frames)
            weight: Weight for consistency loss
        """
        super().__init__()
        self.window_size = window_size
        self.weight = weight
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate temporal consistency loss
        
        Args:
            predictions: Predicted probabilities (batch, seq_len, 2)
        
        Returns:
            Consistency loss (scalar)
        """
        # Get probability of active class
        if predictions.size(-1) == 2:
            probs = F.softmax(predictions, dim=-1)[..., 1]  # (batch, seq_len)
        else:
            probs = predictions
        
        batch_size, seq_len = probs.shape
        
        # Calculate variance within sliding windows
        total_variance = 0.0
        num_windows = 0
        
        for i in range(0, seq_len - self.window_size + 1):
            window = probs[:, i:i+self.window_size]  # (batch, window_size)
            variance = window.var(dim=1).mean()  # Average variance across batch
            total_variance += variance
            num_windows += 1
        
        if num_windows > 0:
            avg_variance = total_variance / num_windows
        else:
            avg_variance = 0.0
        
        return self.weight * avg_variance


class CombinedCutSelectionLoss(nn.Module):
    """
    Combined loss for cut selection training
    
    Combines:
    1. CrossEntropy loss (for classification)
    2. Total Variation loss (for temporal smoothness)
    3. Adoption Rate Penalty (to prevent over-prediction of active class)
    """
    
    def __init__(
        self,
        class_weights: torch.Tensor = None,
        tv_weight: float = 0.1,
        label_smoothing: float = 0.0,
        use_focal: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        target_adoption_rate: float = 0.23,
        adoption_penalty_weight: float = 0.5
    ):
        """
        Initialize combined loss
        
        Args:
            class_weights: Weights for class imbalance [inactive, active]
            tv_weight: Weight for total variation loss (0.01-0.5)
            label_smoothing: Label smoothing factor (0.0-0.2)
            use_focal: Use Focal Loss instead of CrossEntropy
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            target_adoption_rate: Target adoption rate (e.g., 0.23 for 23%)
            adoption_penalty_weight: Weight for adoption rate penalty
        """
        super().__init__()
        
        self.use_focal = use_focal
        self.target_adoption_rate = target_adoption_rate
        self.adoption_penalty_weight = adoption_penalty_weight
        
        if use_focal:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing
            )
        
        self.tv_loss = TotalVariationLoss(weight=tv_weight)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Calculate combined loss
        
        Args:
            predictions: Predicted logits (batch, seq_len, 2)
            targets: Ground truth labels (batch, seq_len)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses for logging
        """
        # Classification loss
        batch_size, seq_len, _ = predictions.shape
        predictions_flat = predictions.view(-1, 2)
        targets_flat = targets.view(-1).long()
        
        ce_loss = self.ce_loss(predictions_flat, targets_flat)
        
        # Temporal smoothness loss
        tv_loss = self.tv_loss(predictions)
        
        # Adoption rate penalty with progressive scaling
        # Calculate predicted adoption rate
        probs = F.softmax(predictions, dim=-1)
        pred_active_prob = probs[..., 1]  # (batch, seq_len)
        pred_adoption_rate = pred_active_prob.mean()
        
        # Progressive penalty based on adoption rate
        # 0-23%: No penalty
        # 23-50%: Linear penalty
        # 50-80%: Quadratic penalty (stronger)
        # 80-100%: Exponential penalty (very strong)
        
        if pred_adoption_rate <= self.target_adoption_rate:
            adoption_penalty = torch.tensor(0.0, device=predictions.device)
        elif pred_adoption_rate <= 0.5:
            # Linear penalty for 23-50%
            excess = pred_adoption_rate - self.target_adoption_rate
            adoption_penalty = self.adoption_penalty_weight * excess
        elif pred_adoption_rate <= 0.8:
            # Quadratic penalty for 50-80%
            excess = pred_adoption_rate - self.target_adoption_rate
            adoption_penalty = self.adoption_penalty_weight * (excess ** 2) * 5.0
        else:
            # Exponential penalty for 80-100% (very strong)
            excess = pred_adoption_rate - self.target_adoption_rate
            adoption_penalty = self.adoption_penalty_weight * (excess ** 3) * 20.0
        
        # Total loss
        total_loss = ce_loss + tv_loss + adoption_penalty
        
        # Return loss components for logging
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'tv_loss': tv_loss.item(),
            'adoption_penalty': adoption_penalty.item(),
            'pred_adoption_rate': pred_adoption_rate.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Focuses training on hard examples by down-weighting easy examples.
    Alternative to weighted CrossEntropy.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for class imbalance (0-1)
            gamma: Focusing parameter (0-5, typically 2)
                  Higher gamma = more focus on hard examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss
        
        Args:
            predictions: Predicted logits (batch*seq_len, 2)
            targets: Ground truth labels (batch*seq_len,)
        
        Returns:
            Focal loss (scalar)
        """
        # Get probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Get probability of true class
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        pt = (probs * targets_one_hot).sum(dim=-1)  # Probability of true class
        
        # Calculate focal loss
        focal_weight = (1 - pt) ** self.gamma
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
