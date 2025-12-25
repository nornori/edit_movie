"""
Enhanced Cut Selection Dataset V2 with Data Augmentation

Adds time series augmentation during training for better generalization.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from src.cut_selection.time_series_augmentation import TimeSeriesAugmentation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedCutSelectionDatasetV2(Dataset):
    """
    Dataset for enhanced cut selection with data augmentation
    
    Loads audio, visual, and temporal features with active labels.
    Applies time series augmentation during training.
    """
    
    def __init__(
        self,
        data_path: str,
        augment: bool = False,
        augment_prob: float = 0.5,
        noise_std: float = 0.01,
        shift_range: int = 5,
        scale_range: tuple = (0.9, 1.1)
    ):
        """
        Initialize dataset
        
        Args:
            data_path: Path to .npz file with combined sequences
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying each augmentation
            noise_std: Standard deviation for Gaussian noise
            shift_range: Maximum frames to shift
            scale_range: Range for magnitude scaling
        """
        self.data_path = Path(data_path)
        self.augment = augment
        
        logger.info(f"Loading enhanced cut selection data from {data_path}")
        
        # Load data
        data = np.load(data_path)
        
        self.audio = torch.from_numpy(data['audio']).float()  # (N, seq_len, audio_dim)
        self.visual = torch.from_numpy(data['visual']).float()  # (N, seq_len, visual_dim)
        self.temporal = torch.from_numpy(data['temporal']).float()  # (N, seq_len, temporal_dim)
        self.active = torch.from_numpy(data['active']).long()  # (N, seq_len)
        self.video_names = data['video_names']  # (N,)
        
        # Initialize augmenter if needed
        if self.augment:
            self.augmenter = TimeSeriesAugmentation(
                noise_std=noise_std,
                shift_range=shift_range,
                scale_range=scale_range,
                augment_prob=augment_prob
            )
            logger.info(f"✅ Data augmentation enabled:")
            logger.info(f"   Noise std: {noise_std}")
            logger.info(f"   Shift range: ±{shift_range} frames")
            logger.info(f"   Scale range: {scale_range}")
            logger.info(f"   Augment prob: {augment_prob}")
        else:
            self.augmenter = None
        
        # Dataset info
        self.num_sequences = len(self.audio)
        self.seq_len = self.audio.shape[1]
        self.audio_dim = self.audio.shape[2]
        self.visual_dim = self.visual.shape[2]
        self.temporal_dim = self.temporal.shape[2]
        
        # Calculate statistics
        total_frames = self.num_sequences * self.seq_len
        active_frames = (self.active == 1).sum().item()
        inactive_frames = (self.active == 0).sum().item()
        
        logger.info(f"EnhancedCutSelectionDatasetV2 initialized:")
        logger.info(f"  Total sequences: {self.num_sequences}")
        logger.info(f"  Unique videos: {len(np.unique(self.video_names))}")
        logger.info(f"  Sequence length: {self.seq_len}")
        logger.info(f"  Audio dimensions: {self.audio_dim}")
        logger.info(f"  Visual dimensions: {self.visual_dim}")
        logger.info(f"  Temporal dimensions: {self.temporal_dim}")
        logger.info(f"  Total input dimensions: {self.audio_dim + self.visual_dim + self.temporal_dim}")
        logger.info(f"  Active samples: {active_frames} ({active_frames/total_frames*100:.2f}%)")
        logger.info(f"  Inactive samples: {inactive_frames} ({inactive_frames/total_frames*100:.2f}%)")
        logger.info(f"  Augmentation: {'ENABLED' if self.augment else 'DISABLED'}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Get a single sequence with optional augmentation
        
        Returns:
            dict with:
                - audio: (seq_len, audio_dim)
                - visual: (seq_len, visual_dim)
                - temporal: (seq_len, temporal_dim)
                - active: (seq_len,)
                - video_name: str
        """
        audio = self.audio[idx]  # (seq_len, audio_dim)
        visual = self.visual[idx]  # (seq_len, visual_dim)
        temporal = self.temporal[idx]  # (seq_len, temporal_dim)
        active = self.active[idx]  # (seq_len,)
        video_name = self.video_names[idx]
        
        # Apply augmentation if enabled (only during training)
        if self.augment and self.augmenter is not None:
            # Add batch dimension for augmenter
            audio_batch = audio.unsqueeze(0)
            visual_batch = visual.unsqueeze(0)
            temporal_batch = temporal.unsqueeze(0)
            active_batch = active.unsqueeze(0)
            
            # Apply augmentation
            audio_batch, visual_batch, temporal_batch, active_batch = self.augmenter.augment(
                audio_batch, visual_batch, temporal_batch, active_batch
            )
            
            # Remove batch dimension
            audio = audio_batch.squeeze(0)
            visual = visual_batch.squeeze(0)
            temporal = temporal_batch.squeeze(0)
            active = active_batch.squeeze(0)
        
        return {
            'audio': audio,
            'visual': visual,
            'temporal': temporal,
            'active': active,
            'video_name': video_name
        }
    
    def get_video_groups(self):
        """
        Get video group indices for GroupKFold
        
        Returns:
            Array of video indices (one per sequence)
        """
        # Create mapping from video name to index
        unique_videos = np.unique(self.video_names)
        video_to_idx = {video: idx for idx, video in enumerate(unique_videos)}
        
        # Map each sequence to its video index
        groups = np.array([video_to_idx[video] for video in self.video_names])
        
        return groups


if __name__ == "__main__":
    # Test dataset
    print("Testing EnhancedCutSelectionDatasetV2...")
    
    data_path = "preprocessed_data/combined_sequences_cut_selection_enhanced.npz"
    
    # Test without augmentation
    dataset_no_aug = EnhancedCutSelectionDatasetV2(data_path, augment=False)
    sample = dataset_no_aug[0]
    
    print(f"\nSample without augmentation:")
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  Visual shape: {sample['visual'].shape}")
    print(f"  Temporal shape: {sample['temporal'].shape}")
    print(f"  Active shape: {sample['active'].shape}")
    print(f"  Video name: {sample['video_name']}")
    
    # Test with augmentation
    dataset_aug = EnhancedCutSelectionDatasetV2(
        data_path,
        augment=True,
        augment_prob=1.0,  # Always augment for testing
        noise_std=0.01,
        shift_range=5,
        scale_range=(0.9, 1.1)
    )
    sample_aug = dataset_aug[0]
    
    print(f"\nSample with augmentation:")
    print(f"  Audio shape: {sample_aug['audio'].shape}")
    print(f"  Visual shape: {sample_aug['visual'].shape}")
    print(f"  Temporal shape: {sample_aug['temporal'].shape}")
    print(f"  Active shape: {sample_aug['active'].shape}")
    
    # Check that augmentation changed the data
    audio_diff = torch.abs(sample['audio'] - sample_aug['audio']).mean().item()
    print(f"\nAudio difference (should be > 0): {audio_diff:.6f}")
    
    # Test video groups
    groups = dataset_no_aug.get_video_groups()
    print(f"\nVideo groups shape: {groups.shape}")
    print(f"Unique videos: {len(np.unique(groups))}")
    
    print("\n✅ EnhancedCutSelectionDatasetV2 test passed!")
