"""
Time Series Data Augmentation for Cut Selection

Implements various augmentation techniques for time series data:
- Gaussian noise injection
- Time shifting
- Magnitude scaling
- Time warping
"""
import torch
import numpy as np
from typing import Tuple, Optional


class TimeSeriesAugmentation:
    """
    Time series augmentation for training data
    
    Applies random augmentations to audio, visual, and temporal features
    to improve model generalization.
    """
    
    def __init__(
        self,
        noise_std: float = 0.01,
        shift_range: int = 5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        augment_prob: float = 0.5
    ):
        """
        Initialize augmentation parameters
        
        Args:
            noise_std: Standard deviation for Gaussian noise
            shift_range: Maximum frames to shift (±shift_range)
            scale_range: Range for magnitude scaling (min, max)
            augment_prob: Probability of applying each augmentation
        """
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.augment_prob = augment_prob
    
    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to features
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            Noisy tensor
        """
        if np.random.rand() < self.augment_prob:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x
    
    def time_shift(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shift sequence in time (circular shift)
        
        Args:
            x: Input tensor (batch, seq_len, features)
            labels: Label tensor (batch, seq_len)
        
        Returns:
            Shifted tensor and labels
        """
        if np.random.rand() < self.augment_prob:
            batch_size, seq_len, _ = x.shape
            shift = np.random.randint(-self.shift_range, self.shift_range + 1)
            
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=1)
                labels = torch.roll(labels, shifts=shift, dims=1)
        
        return x, labels
    
    def magnitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale magnitude of features
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            Scaled tensor
        """
        if np.random.rand() < self.augment_prob:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            return x * scale
        return x
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping (speed up/slow down locally)
        
        Args:
            x: Input tensor (batch, seq_len, features)
        
        Returns:
            Warped tensor
        """
        if np.random.rand() < self.augment_prob:
            batch_size, seq_len, features = x.shape
            
            # Random warp factor (0.8 to 1.2)
            warp_factor = np.random.uniform(0.8, 1.2)
            
            # Create warped indices
            original_indices = torch.arange(seq_len, dtype=torch.float32)
            warped_indices = original_indices * warp_factor
            warped_indices = torch.clamp(warped_indices, 0, seq_len - 1).long()
            
            # Apply warping
            x_warped = x[:, warped_indices, :]
            
            return x_warped
        
        return x
    
    def augment(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        temporal: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply full augmentation pipeline
        
        Args:
            audio: Audio features (batch, seq_len, audio_dim)
            visual: Visual features (batch, seq_len, visual_dim)
            temporal: Temporal features (batch, seq_len, temporal_dim)
            labels: Labels (batch, seq_len)
        
        Returns:
            Augmented audio, visual, temporal, and labels
        """
        # Apply same time shift to all modalities and labels
        audio, labels = self.time_shift(audio, labels)
        visual, _ = self.time_shift(visual, labels)
        temporal, _ = self.time_shift(temporal, labels)
        
        # Apply noise independently to each modality
        audio = self.add_gaussian_noise(audio)
        visual = self.add_gaussian_noise(visual)
        temporal = self.add_gaussian_noise(temporal)
        
        # Apply magnitude scaling independently
        audio = self.magnitude_scale(audio)
        visual = self.magnitude_scale(visual)
        temporal = self.magnitude_scale(temporal)
        
        return audio, visual, temporal, labels


if __name__ == "__main__":
    # Test augmentation
    print("Testing TimeSeriesAugmentation...")
    
    batch_size = 4
    seq_len = 1000
    audio_dim = 235
    visual_dim = 543
    temporal_dim = 6
    
    # Create dummy data
    audio = torch.randn(batch_size, seq_len, audio_dim)
    visual = torch.randn(batch_size, seq_len, visual_dim)
    temporal = torch.randn(batch_size, seq_len, temporal_dim)
    labels = torch.randint(0, 2, (batch_size, seq_len))
    
    # Create augmenter
    augmenter = TimeSeriesAugmentation(
        noise_std=0.01,
        shift_range=5,
        scale_range=(0.9, 1.1),
        augment_prob=0.5
    )
    
    # Apply augmentation
    audio_aug, visual_aug, temporal_aug, labels_aug = augmenter.augment(
        audio, visual, temporal, labels
    )
    
    print(f"Original audio shape: {audio.shape}")
    print(f"Augmented audio shape: {audio_aug.shape}")
    print(f"Original labels shape: {labels.shape}")
    print(f"Augmented labels shape: {labels_aug.shape}")
    
    # Check that shapes are preserved
    assert audio_aug.shape == audio.shape
    assert visual_aug.shape == visual.shape
    assert temporal_aug.shape == temporal.shape
    assert labels_aug.shape == labels.shape
    
    print("✅ TimeSeriesAugmentation test passed!")
