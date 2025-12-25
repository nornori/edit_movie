"""
Enhanced Dataset for cut selection training with temporal features

Loads audio + visual + temporal features with active labels from source videos.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnhancedCutSelectionDataset(Dataset):
    """
    Enhanced Dataset for cut selection (Stage 1) with temporal features
    
    Input: Audio + Visual + Temporal features from source videos
    Output: Active labels (binary: 0=not used, 1=used in edit)
    """
    
    def __init__(self, sequences_npz: str):
        """
        Initialize dataset
        
        Args:
            sequences_npz: Path to .npz file with enhanced cut selection sequences
                          (created by scripts/create_cut_selection_data_enhanced.py)
        """
        self.sequences_npz = sequences_npz
        
        logger.info(f"Loading enhanced cut selection data from {sequences_npz}")
        data = np.load(sequences_npz, allow_pickle=True)
        
        # Load sequences
        self.audio_features = data['audio']  # (N, seq_len, audio_dim)
        self.visual_features = data['visual']  # (N, seq_len, visual_dim)
        self.temporal_features = data.get('temporal', None)  # (N, seq_len, temporal_dim) - optional
        self.active_labels = data['active']  # (N, seq_len) - binary labels
        
        # Load video names (for GroupKFold)
        if 'video_names' in data:
            self.video_names = data['video_names']
        else:
            # Fallback: assign unique ID to each sequence
            logger.warning("video_names not found in data. Using sequence indices as groups.")
            self.video_names = [f"seq_{i}" for i in range(len(self.audio_features))]
        
        self.num_sequences = len(self.audio_features)
        self.seq_len = self.audio_features.shape[1]
        self.audio_dim = self.audio_features.shape[2]
        self.visual_dim = self.visual_features.shape[2]
        
        # Handle temporal features
        if self.temporal_features is not None:
            self.temporal_dim = self.temporal_features.shape[2]
            self.has_temporal = True
        else:
            self.temporal_dim = 0
            self.has_temporal = False
            logger.warning("No temporal features found in data. Will use zeros as fallback.")
        
        logger.info(f"EnhancedCutSelectionDataset initialized:")
        logger.info(f"  Total sequences: {self.num_sequences}")
        logger.info(f"  Unique videos: {len(set(self.video_names))}")
        logger.info(f"  Sequence length: {self.seq_len}")
        logger.info(f"  Audio dimensions: {self.audio_dim}")
        logger.info(f"  Visual dimensions: {self.visual_dim}")
        logger.info(f"  Temporal dimensions: {self.temporal_dim}")
        logger.info(f"  Total input dimensions: {self.audio_dim + self.visual_dim + self.temporal_dim}")
        
        # Calculate class balance
        total_samples = self.active_labels.size
        active_count = np.sum(self.active_labels == 1)
        inactive_count = np.sum(self.active_labels == 0)
        logger.info(f"  Active samples: {active_count} ({active_count/total_samples*100:.2f}%)")
        logger.info(f"  Inactive samples: {inactive_count} ({inactive_count/total_samples*100:.2f}%)")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Get a single sequence
        
        Returns:
            dict with:
                - audio: (seq_len, audio_dim)
                - visual: (seq_len, visual_dim)
                - temporal: (seq_len, temporal_dim)
                - active: (seq_len,) - binary labels (long type for CrossEntropyLoss)
        """
        result = {
            'audio': torch.from_numpy(self.audio_features[idx]).float(),
            'visual': torch.from_numpy(self.visual_features[idx]).float(),
            'active': torch.from_numpy(self.active_labels[idx]).long()  # Must be long for CE loss
        }
        
        # Add temporal features if available
        if self.has_temporal:
            result['temporal'] = torch.from_numpy(self.temporal_features[idx]).float()
        else:
            # Fallback: create zero tensor with shape (seq_len, 7)
            result['temporal'] = torch.zeros(self.seq_len, 7).float()
        
        return result
    
    def get_video_groups(self):
        """Get video names for GroupKFold"""
        return self.video_names
