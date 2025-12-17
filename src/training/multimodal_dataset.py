"""
Multimodal Dataset Loader for Multi-Track Transformer

This module provides a PyTorch Dataset that loads and aligns audio, visual,
and track features for multimodal video editing prediction.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.utils.feature_alignment import FeatureAligner
from src.training.multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor
from src.data_preparation.text_embedding import SimpleTextEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for multimodal video editing data
    
    Loads audio features, visual features, and track sequences,
    aligns them by timestamp, and returns synchronized batches.
    """
    
    def __init__(
        self,
        sequences_npz: str,
        features_dir: str,
        audio_preprocessor: Optional[AudioFeaturePreprocessor] = None,
        visual_preprocessor: Optional[VisualFeaturePreprocessor] = None,
        enable_multimodal: bool = True,
        tolerance: float = 0.05,
        use_text_embedding: bool = True
    ):
        """
        Initialize MultimodalDataset
        
        Args:
            sequences_npz: Path to .npz file with track sequences
            features_dir: Directory containing audio and visual feature CSV files
            audio_preprocessor: Optional fitted AudioFeaturePreprocessor
            visual_preprocessor: Optional fitted VisualFeaturePreprocessor
            enable_multimodal: Whether to load multimodal features (fallback to track-only if False)
            tolerance: Timestamp alignment tolerance in seconds
            use_text_embedding: Whether to use text embedding for text_word column
        """
        self.features_dir = Path(features_dir)
        self.use_text_embedding = use_text_embedding
        
        # Initialize text embedder if enabled
        if use_text_embedding:
            self.text_embedder = SimpleTextEmbedder()
            logger.info(f"Text embedding enabled: {self.text_embedder.embedding_dim} dimensions")
        else:
            self.text_embedder = None
        self.enable_multimodal = enable_multimodal
        self.audio_preprocessor = audio_preprocessor
        self.visual_preprocessor = visual_preprocessor
        
        # Load track sequences
        logger.info(f"Loading track sequences from {sequences_npz}")
        data = np.load(sequences_npz)
        self.track_sequences = data['sequences']  # (N, seq_len, 180)
        self.masks = data['masks']  # (N, seq_len)
        self.video_ids = data.get('video_ids', None)
        self.source_video_names = data.get('source_video_names', None)
        
        # Initialize feature aligner
        self.aligner = FeatureAligner(tolerance=tolerance)
        
        # Lazy loading: cache for loaded features
        self._audio_cache = {}
        self._visual_cache = {}
        
        # Track statistics
        self.stats = {
            'total_videos': len(self.track_sequences),
            'audio_available': 0,
            'visual_available': 0,
            'both_available': 0,
            'track_only': 0
        }
        
        # Check feature availability
        if self.enable_multimodal:
            self._check_feature_availability()
        
        logger.info(f"MultimodalDataset initialized:")
        logger.info(f"  Total videos: {self.stats['total_videos']}")
        logger.info(f"  Audio available: {self.stats['audio_available']}")
        logger.info(f"  Visual available: {self.stats['visual_available']}")
        logger.info(f"  Both available: {self.stats['both_available']}")
        logger.info(f"  Track only: {self.stats['track_only']}")
        logger.info(f"  Enable multimodal: {self.enable_multimodal}")
    
    def _check_feature_availability(self):
        """Check which videos have audio and visual features available"""
        for idx in range(len(self.track_sequences)):
            video_id = self._get_video_id(idx)
            
            # Remove chunk suffix if present (e.g., "_chunk0", "_chunk1")
            base_video_id = video_id.rsplit('_chunk', 1)[0] if '_chunk' in video_id else video_id
            
            # Check for unified features file
            features_path = self.features_dir / f"{base_video_id}_features.csv"
            
            if features_path.exists():
                # Unified file contains both audio and visual features
                self.stats['audio_available'] += 1
                self.stats['visual_available'] += 1
                self.stats['both_available'] += 1
            else:
                self.stats['track_only'] += 1
    
    def _get_video_id(self, idx: int) -> str:
        """Get video ID for a given index"""
        if self.video_ids is not None:
            return str(self.video_ids[idx])
        elif self.source_video_names is not None:
            return str(self.source_video_names[idx])
        else:
            return f"video_{idx}"
    
    def _load_audio_features(self, video_id: str) -> Optional[pd.DataFrame]:
        """
        Load audio features for a video
        
        Args:
            video_id: Video identifier
        
        Returns:
            DataFrame with audio features or None if not available
        """
        # Check cache
        if video_id in self._audio_cache:
            return self._audio_cache[video_id]
        
        # Remove chunk suffix if present
        base_video_id = video_id.rsplit('_chunk', 1)[0] if '_chunk' in video_id else video_id
        
        audio_path = self.features_dir / f"{base_video_id}_features.csv"
        
        if not audio_path.exists():
            self._audio_cache[video_id] = None
            return None
        
        try:
            df = pd.read_csv(audio_path)
            
            # Verify required columns
            required_cols = ['time', 'audio_energy_rms', 'audio_is_speaking', 
                           'silence_duration_ms', 'text_is_active']
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                logger.warning(f"{video_id}: Missing audio columns: {missing_cols}")
                self._audio_cache[video_id] = None
                return None
            
            # Add text embedding if enabled and text_word column exists
            if self.use_text_embedding and 'text_word' in df.columns:
                text_embeddings = self.text_embedder.embed_series(df['text_word'])
                # Add embedding columns for speech text
                for i in range(self.text_embedder.embedding_dim):
                    df[f'speech_emb_{i}'] = text_embeddings[:, i]
            
            # Add telop embedding if enabled and telop_text column exists
            if self.use_text_embedding and 'telop_text' in df.columns:
                telop_embeddings = self.text_embedder.embed_series(df['telop_text'])
                # Add embedding columns for telop text
                for i in range(self.text_embedder.embedding_dim):
                    df[f'telop_emb_{i}'] = telop_embeddings[:, i]
            
            self._audio_cache[video_id] = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to load audio features for {video_id}: {e}")
            self._audio_cache[video_id] = None
            return None
    
    def _load_visual_features(self, video_id: str) -> Optional[pd.DataFrame]:
        """
        Load visual features for a video
        
        Args:
            video_id: Video identifier
        
        Returns:
            DataFrame with visual features or None if not available
        """
        # Check cache
        if video_id in self._visual_cache:
            return self._visual_cache[video_id]
        
        # Remove chunk suffix if present
        base_video_id = video_id.rsplit('_chunk', 1)[0] if '_chunk' in video_id else video_id
        
        # Visual features are in the same unified file as audio features
        features_path = self.features_dir / f"{base_video_id}_features.csv"
        
        if not features_path.exists():
            self._visual_cache[video_id] = None
            return None
        
        try:
            df = pd.read_csv(features_path)
            
            # Verify required columns
            required_cols = ['time', 'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
                           'face_count', 'face_center_x', 'face_center_y', 'face_size',
                           'face_mouth_open', 'face_eyebrow_raise']
            
            missing_cols = [c for c in required_cols if c not in df.columns]
            
            if missing_cols:
                logger.warning(f"{video_id}: Missing visual columns: {missing_cols}")
                self._visual_cache[video_id] = None
                return None
            
            # Check for CLIP features
            clip_cols = [f'clip_{i}' for i in range(512)]
            missing_clip = [c for c in clip_cols if c not in df.columns]
            
            if len(missing_clip) > 0:
                logger.warning(f"{video_id}: Missing {len(missing_clip)} CLIP features")
                self._visual_cache[video_id] = None
                return None
            
            self._visual_cache[video_id] = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to load visual features for {video_id}: {e}")
            self._visual_cache[video_id] = None
            return None
    
    def __len__(self) -> int:
        return len(self.track_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            Dict with keys:
            - 'audio': FloatTensor of shape (seq_len, 4) or None
            - 'visual': FloatTensor of shape (seq_len, 522) or None
            - 'track': FloatTensor of shape (seq_len, 180)
            - 'targets': FloatTensor of shape (seq_len, 20, 9)
            - 'padding_mask': BoolTensor of shape (seq_len,)
            - 'modality_mask': BoolTensor of shape (seq_len, 3) for [audio, visual, track]
            - 'video_id': str
        """
        video_id = self._get_video_id(idx)
        
        # Get track sequence and mask
        track_seq = self.track_sequences[idx]  # (seq_len, 180)
        padding_mask = self.masks[idx]  # (seq_len,)
        seq_len = len(track_seq)
        
        # Generate timestamps (assuming 10 FPS = 0.1s per frame)
        track_times = np.arange(seq_len) * 0.1
        
        # Initialize outputs
        audio_features = None
        visual_features = None
        modality_mask = np.ones((seq_len, 3), dtype=bool)
        modality_mask[:, 2] = True  # Track is always available
        
        # Load and align multimodal features if enabled
        if self.enable_multimodal:
            audio_df = self._load_audio_features(video_id)
            visual_df = self._load_visual_features(video_id)
            
            # Align features
            aligned_audio, aligned_visual, modality_mask, stats = self.aligner.align_features(
                track_times, audio_df, visual_df, video_id
            )
            
            # Apply preprocessing if available and dimensions match
            if aligned_audio is not None and self.audio_preprocessor is not None:
                # Check if dimensions match
                expected_dim = len(self.audio_preprocessor.feature_names)
                actual_dim = aligned_audio.shape[1]
                if expected_dim == actual_dim:
                    audio_features = self.audio_preprocessor.transform(aligned_audio)
                else:
                    logger.warning(f"{video_id}: Audio preprocessor expects {expected_dim} features, got {actual_dim}. Skipping preprocessing.")
                    audio_features = aligned_audio
            else:
                audio_features = aligned_audio
            
            # Replace NaN with 0 in audio features
            if audio_features is not None:
                audio_features = np.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            if aligned_visual is not None and self.visual_preprocessor is not None:
                # Extract face_count for zero-filling
                face_counts = aligned_visual[:, 4].astype(int) if aligned_visual is not None else None
                visual_features = self.visual_preprocessor.transform(aligned_visual, face_counts)
            else:
                visual_features = aligned_visual
            
            # Replace NaN with 0 in visual features
            if visual_features is not None:
                visual_features = np.nan_to_num(visual_features, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Track-only mode
            modality_mask[:, 0] = False  # No audio
            modality_mask[:, 1] = False  # No visual
        
        # Reshape track sequence to (seq_len, 20, 12) for targets
        targets = track_seq.reshape(seq_len, 20, 12)
        
        # Determine audio feature dimension
        # 4 base + 1 telop_active + 6 speech embedding + 6 telop embedding if enabled
        audio_dim = 17 if self.use_text_embedding else 5  # 5 = 4 base + 1 telop_active
        
        # Convert to tensors
        sample = {
            'audio': torch.FloatTensor(audio_features) if audio_features is not None else torch.zeros(seq_len, audio_dim),
            'visual': torch.FloatTensor(visual_features) if visual_features is not None else torch.zeros(seq_len, 522),
            'track': torch.FloatTensor(track_seq),
            'targets': torch.FloatTensor(targets),
            'padding_mask': torch.BoolTensor(padding_mask),
            'modality_mask': torch.BoolTensor(modality_mask),
            'video_id': video_id
        }
        
        return sample


def collate_fn(batch):
    """
    Custom collate function for batching
    
    Args:
        batch: List of samples from __getitem__
    
    Returns:
        Dict with batched tensors
    """
    # Stack all tensors
    audio = torch.stack([item['audio'] for item in batch])
    visual = torch.stack([item['visual'] for item in batch])
    track = torch.stack([item['track'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    padding_mask = torch.stack([item['padding_mask'] for item in batch])
    modality_mask = torch.stack([item['modality_mask'] for item in batch])
    
    result = {
        'audio': audio,
        'visual': visual,
        'track': track,
        'targets': targets,
        'padding_mask': padding_mask,
        'modality_mask': modality_mask,
        'video_ids': [item['video_id'] for item in batch]
    }
    
    return result


def create_multimodal_dataloaders(
    train_npz: str,
    val_npz: str,
    features_dir: str,
    audio_preprocessor_path: Optional[str] = None,
    visual_preprocessor_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    enable_multimodal: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for multimodal data
    
    Args:
        train_npz: Path to training .npz file
        val_npz: Path to validation .npz file
        features_dir: Directory containing feature CSV files
        audio_preprocessor_path: Path to saved AudioFeaturePreprocessor
        visual_preprocessor_path: Path to saved VisualFeaturePreprocessor
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        pin_memory: Whether to pin memory for faster GPU transfer
        enable_multimodal: Whether to enable multimodal features
    
    Returns:
        (train_loader, val_loader)
    """
    logger.info("Creating multimodal dataloaders...")
    
    # Load preprocessors if provided
    audio_preprocessor = None
    visual_preprocessor = None
    
    if audio_preprocessor_path and Path(audio_preprocessor_path).exists():
        audio_preprocessor = AudioFeaturePreprocessor.load(audio_preprocessor_path)
        logger.info(f"Loaded audio preprocessor from {audio_preprocessor_path}")
    
    if visual_preprocessor_path and Path(visual_preprocessor_path).exists():
        visual_preprocessor = VisualFeaturePreprocessor.load(visual_preprocessor_path)
        logger.info(f"Loaded visual preprocessor from {visual_preprocessor_path}")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        sequences_npz=train_npz,
        features_dir=features_dir,
        audio_preprocessor=audio_preprocessor,
        visual_preprocessor=visual_preprocessor,
        enable_multimodal=enable_multimodal,
        use_text_embedding=True  # Enable text embedding
    )
    
    val_dataset = MultimodalDataset(
        sequences_npz=val_npz,
        features_dir=features_dir,
        audio_preprocessor=audio_preprocessor,
        visual_preprocessor=visual_preprocessor,
        enable_multimodal=enable_multimodal,
        use_text_embedding=True  # Enable text embedding
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multimodal dataset and dataloader")
    parser.add_argument("--train_npz", default="preprocessed_data/train_sequences.npz")
    parser.add_argument("--val_npz", default="preprocessed_data/val_sequences.npz")
    parser.add_argument("--features_dir", default="input_features")
    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, val_loader = create_multimodal_dataloaders(
        args.train_npz,
        args.val_npz,
        args.features_dir,
        batch_size=args.batch_size,
        enable_multimodal=True
    )
    
    # Test loading a batch
    logger.info("\nTesting batch loading...")
    for batch in train_loader:
        logger.info(f"Batch audio shape: {batch['audio'].shape}")
        logger.info(f"Batch visual shape: {batch['visual'].shape}")
        logger.info(f"Batch track shape: {batch['track'].shape}")
        logger.info(f"Batch targets shape: {batch['targets'].shape}")
        logger.info(f"Batch padding_mask shape: {batch['padding_mask'].shape}")
        logger.info(f"Batch modality_mask shape: {batch['modality_mask'].shape}")
        logger.info(f"Number of video_ids: {len(batch['video_ids'])}")
        logger.info(f"Sample video_id: {batch['video_ids'][0]}")
        logger.info(f"Modality mask sample:\n{batch['modality_mask'][0][:5]}")
        break
    
    logger.info("\n✅ Multimodal dataset and DataLoader test complete!")
