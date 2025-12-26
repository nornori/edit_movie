"""
Enhanced Dataset for cut selection training with FULL VIDEOS (variable length)

Loads audio + visual + temporal features with active labels from source videos.
1 VIDEO = 1 SAMPLE (no sequence splitting)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnhancedCutSelectionDatasetFullVideo(Dataset):
    """
    Enhanced Dataset for cut selection with FULL VIDEOS (variable length)
    
    Input: Audio + Visual + Temporal features from source videos
    Output: Active labels (binary: 0=not used, 1=used in edit)
    
    Each sample is a complete video (variable length)
    """
    
    def __init__(self, data_npz: str):
        """
        Initialize dataset
        
        Args:
            data_npz: Path to .npz file with full video data
                     (created by scripts/create_cut_selection_data_enhanced_fullvideo.py)
        """
        self.data_npz = data_npz
        
        logger.info(f"Loading full video data from {data_npz}")
        data = np.load(data_npz, allow_pickle=True)
        
        # Load videos (each is a numpy array with different length)
        self.audio_features = data['audio']  # List of (seq_len, audio_dim) arrays
        self.visual_features = data['visual']  # List of (seq_len, visual_dim) arrays
        self.temporal_features = data.get('temporal', None)  # List of (seq_len, temporal_dim) arrays
        self.active_labels = data['active']  # List of (seq_len,) arrays
        self.video_names = data['video_names']  # List of video names
        
        self.num_videos = len(self.audio_features)
        
        # Get feature dimensions from first video
        self.audio_dim = self.audio_features[0].shape[1]
        self.visual_dim = self.visual_features[0].shape[1]
        
        if self.temporal_features is not None:
            self.temporal_dim = self.temporal_features[0].shape[1]
            self.has_temporal = True
        else:
            self.temporal_dim = 0
            self.has_temporal = False
            logger.warning("No temporal features found in data.")
        
        # Calculate video length statistics
        video_lengths = [len(audio) for audio in self.audio_features]
        
        logger.info(f"EnhancedCutSelectionDatasetFullVideo initialized:")
        logger.info(f"  Total videos: {self.num_videos}")
        logger.info(f"  Video lengths: min={min(video_lengths)}, max={max(video_lengths)}, mean={np.mean(video_lengths):.1f}")
        logger.info(f"  Audio dimensions: {self.audio_dim}")
        logger.info(f"  Visual dimensions: {self.visual_dim}")
        logger.info(f"  Temporal dimensions: {self.temporal_dim}")
        logger.info(f"  Total input dimensions: {self.audio_dim + self.visual_dim + self.temporal_dim}")
        
        # Calculate class balance
        total_frames = sum(len(labels) for labels in self.active_labels)
        active_frames = sum(np.sum(labels == 1) for labels in self.active_labels)
        logger.info(f"  Active frames: {active_frames} / {total_frames} ({active_frames/total_frames*100:.2f}%)")
    
    def __len__(self):
        return self.num_videos
    
    def __getitem__(self, idx):
        """
        Get a single video
        
        Returns:
            dict with:
                - audio: (seq_len, audio_dim)
                - visual: (seq_len, visual_dim)
                - temporal: (seq_len, temporal_dim)
                - active: (seq_len,) - binary labels (long type for CrossEntropyLoss)
                - video_name: str - video name for tracking
        """
        result = {
            'audio': torch.from_numpy(self.audio_features[idx]).float(),
            'visual': torch.from_numpy(self.visual_features[idx]).float(),
            'active': torch.from_numpy(self.active_labels[idx]).long(),  # Must be long for CE loss
            'video_name': self.video_names[idx]
        }
        
        # Add temporal features if available
        if self.has_temporal:
            result['temporal'] = torch.from_numpy(self.temporal_features[idx]).float()
        else:
            # Fallback: create zero tensor
            seq_len = len(self.audio_features[idx])
            result['temporal'] = torch.zeros(seq_len, 7).float()
        
        return result
    
    def get_video_names(self):
        """Get video names for tracking"""
        return self.video_names


def collate_fn_fullvideo(batch):
    """
    Collate function for variable-length videos
    
    Pads all videos in batch to the same length (longest video in batch)
    
    Args:
        batch: List of dicts from __getitem__
    
    Returns:
        dict with:
            - audio: (batch, max_seq_len, audio_dim)
            - visual: (batch, max_seq_len, visual_dim)
            - temporal: (batch, max_seq_len, temporal_dim)
            - active: (batch, max_seq_len)
            - padding_mask: (batch, max_seq_len) - True for padding positions
            - video_names: List of video names
            - original_lengths: List of original sequence lengths
    """
    # Get max length in this batch
    max_len = max(item['audio'].shape[0] for item in batch)
    
    batch_size = len(batch)
    audio_dim = batch[0]['audio'].shape[1]
    visual_dim = batch[0]['visual'].shape[1]
    temporal_dim = batch[0]['temporal'].shape[1]
    
    # Initialize padded tensors
    audio_padded = torch.zeros(batch_size, max_len, audio_dim)
    visual_padded = torch.zeros(batch_size, max_len, visual_dim)
    temporal_padded = torch.zeros(batch_size, max_len, temporal_dim)
    active_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padding
    
    video_names = []
    original_lengths = []
    
    for i, item in enumerate(batch):
        seq_len = item['audio'].shape[0]
        
        # Copy data
        audio_padded[i, :seq_len] = item['audio']
        visual_padded[i, :seq_len] = item['visual']
        temporal_padded[i, :seq_len] = item['temporal']
        active_padded[i, :seq_len] = item['active']
        
        # Mark non-padding positions as False
        padding_mask[i, :seq_len] = False
        
        video_names.append(item['video_name'])
        original_lengths.append(seq_len)
    
    return {
        'audio': audio_padded,
        'visual': visual_padded,
        'temporal': temporal_padded,
        'active': active_padded,
        'padding_mask': padding_mask,
        'video_names': video_names,
        'original_lengths': original_lengths
    }
