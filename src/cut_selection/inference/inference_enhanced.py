"""
Enhanced Cut Selection Inference Pipeline

Input: Video file
Output: Premiere Pro XML with selected clips

This script handles the complete pipeline:
1. Extract features from video (audio 235 + visual 543)
2. Add temporal features (6 dimensions)
3. Predict with Enhanced Cut Selection Model (3 modalities)
4. Filter and select clips
5. Generate Premiere Pro XML
"""
import torch
import numpy as np
import pandas as pd
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedCutSelectionInference:
    """Enhanced Cut Selection inference pipeline with 3 modalities"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        fps: float = 10.0
    ):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model (.pth file)
            device: 'cpu' or 'cuda'
            fps: Frame rate for feature extraction (default: 10.0)
        """
        self.device = device
        self.fps = fps
        self.model_path = Path(model_path)
        
        logger.info(f"🚀 Initializing Enhanced Cut Selection Inference")
        logger.info(f"   Device: {device}")
        logger.info(f"   FPS: {fps}")
        
        # Load model
        logger.info(f"📦 Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        self.config = checkpoint['config']
        self.best_threshold = checkpoint.get('best_threshold', -0.558)
        self.best_f1 = checkpoint.get('best_f1', None)
        
        logger.info(f"   Audio features: {self.config['audio_features']}")
        logger.info(f"   Visual features: {self.config['visual_features']}")
        logger.info(f"   Temporal features: {self.config['temporal_features']}")
        logger.info(f"   Best threshold: {self.best_threshold:.3f}")
        if self.best_f1:
            logger.info(f"   Best F1: {self.best_f1:.2%}")
        
        # Create model
        self.model = EnhancedCutSelectionModel(
            audio_features=self.config['audio_features'],
            visual_features=self.config['visual_features'],
            temporal_features=self.config['temporal_features'],
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_encoder_layers=self.config.get('num_encoder_layers', 6),
            dim_feedforward=self.config.get('dim_feedforward', 1024),
            dropout=self.config.get('dropout', 0.15)
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"✅ Model loaded successfully")
        
        # Load scalers
        model_dir = self.model_path.parent
        self.audio_scaler = self._load_scaler(model_dir / 'audio_scaler_cut_selection_enhanced.pkl', 'audio')
        self.visual_scaler = self._load_scaler(model_dir / 'visual_scaler_cut_selection_enhanced.pkl', 'visual')
        self.temporal_scaler = self._load_scaler(model_dir / 'temporal_scaler_cut_selection_enhanced.pkl', 'temporal')
        
        # Load inference parameters
        inference_params_path = model_dir / 'inference_params.yaml'
        if inference_params_path.exists():
            import yaml
            with open(inference_params_path, 'r', encoding='utf-8') as f:
                self.inference_params = yaml.safe_load(f)
            logger.info(f"📋 Loaded inference parameters")
        else:
            self.inference_params = {
                'min_clip_duration': 3.0,
                'max_gap_duration': 2.0,
                'target_duration': 90.0,
                'max_duration': 150.0
            }
            logger.warning(f"⚠️  Inference parameters not found, using defaults")
    
    def _load_scaler(self, path: Path, name: str):
        """Load StandardScaler from pickle file"""
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"   ✅ Loaded {name} scaler")
                return scaler
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to load {name} scaler: {e}")
                return None
        else:
            logger.warning(f"   ⚠️  {name} scaler not found at {path}")
            return None
    
    def extract_features(self, video_path: str, output_dir: str = None) -> str:
        """
        Extract features from video
        
        Args:
            video_path: Path to input video
            output_dir: Output directory for features (default: temp_features/)
        
        Returns:
            Path to extracted features CSV
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if output_dir is None:
            output_dir = project_root / 'temp_features'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = video_path.stem
        output_csv = output_dir / f"{video_name}_features.csv"
        
        # Check if features already exist
        if output_csv.exists():
            logger.info(f"✅ Features already exist, skipping extraction")
            logger.info(f"   Using: {output_csv}")
            return str(output_csv)
        
        logger.info(f"🎬 Extracting features from video...")
        logger.info(f"   Input: {video_path}")
        logger.info(f"   Output: {output_csv}")
        
        # Run feature extraction script
        cmd = [
            sys.executable, '-m', 'src.data_preparation.extract_video_features_parallel',
            str(video_path.parent),  # input_dir (positional argument)
            '--output-dir', str(output_dir),
            '--n-jobs', '1'  # Single video
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ Feature extraction completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            logger.error(f"   stdout: {e.stdout}")
            logger.error(f"   stderr: {e.stderr}")
            raise
        
        if not output_csv.exists():
            raise FileNotFoundError(f"Feature file not created: {output_csv}")
        
        return str(output_csv)
    
    def add_temporal_features(self, feature_csv: str) -> str:
        """
        Add temporal features to extracted features
        
        Args:
            feature_csv: Path to features CSV
        
        Returns:
            Path to enhanced features CSV
        """
        feature_path = Path(feature_csv)
        enhanced_csv = feature_path.parent / f"{feature_path.stem}_enhanced.csv"
        
        # Check if enhanced features already exist
        if enhanced_csv.exists():
            logger.info(f"✅ Enhanced features already exist, skipping temporal feature addition")
            logger.info(f"   Using: {enhanced_csv}")
            return str(enhanced_csv)
        
        logger.info(f"⏱️  Adding temporal features...")
        logger.info(f"   Input: {feature_csv}")
        logger.info(f"   Output: {enhanced_csv}")
        
        # Load features
        df = pd.read_csv(feature_csv)
        logger.info(f"   Loaded {len(df)} frames, {len(df.columns)} columns")
        
        # Add temporal features directly
        df = self._add_temporal_features_to_df(df)
        
        # Save
        df.to_csv(enhanced_csv, index=False)
        logger.info(f"✅ Temporal features added: {len(df.columns)} columns")
        
        return str(enhanced_csv)
    
    def _add_temporal_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to dataframe"""
        # Moving averages for audio_energy_rms
        if 'audio_energy_rms' in df.columns:
            for window in [5, 10, 30]:
                df[f'audio_energy_rms_ma{window}'] = df['audio_energy_rms'].rolling(window=window, center=True, min_periods=1).mean()
        
        # Moving averages for visual_motion
        if 'visual_motion' in df.columns:
            for window in [5, 10, 30]:
                df[f'visual_motion_ma{window}'] = df['visual_motion'].rolling(window=window, center=True, min_periods=1).mean()
        
        # CLIP similarity (if CLIP features exist)
        clip_cols = [col for col in df.columns if col.startswith('clip_')]
        if clip_cols:
            clip_features = df[clip_cols].values
            # Compute cosine similarity with previous frame
            clip_sim_prev = np.zeros(len(df))
            for i in range(1, len(df)):
                sim = np.dot(clip_features[i], clip_features[i-1]) / (np.linalg.norm(clip_features[i]) * np.linalg.norm(clip_features[i-1]) + 1e-8)
                clip_sim_prev[i] = sim
            df['clip_sim_prev'] = clip_sim_prev
            
            # Compute similarity with next frame
            clip_sim_next = np.zeros(len(df))
            for i in range(len(df)-1):
                sim = np.dot(clip_features[i], clip_features[i+1]) / (np.linalg.norm(clip_features[i]) * np.linalg.norm(clip_features[i+1]) + 1e-8)
                clip_sim_next[i] = sim
            df['clip_sim_next'] = clip_sim_next
            
            # Mean similarity with surrounding 5 frames
            clip_sim_mean5 = np.zeros(len(df))
            for i in range(len(df)):
                start = max(0, i-2)
                end = min(len(df), i+3)
                if end - start > 1:
                    sims = []
                    for j in range(start, end):
                        if j != i:
                            sim = np.dot(clip_features[i], clip_features[j]) / (np.linalg.norm(clip_features[i]) * np.linalg.norm(clip_features[j]) + 1e-8)
                            sims.append(sim)
                    clip_sim_mean5[i] = np.mean(sims) if sims else 0.0
            df['clip_sim_mean5'] = clip_sim_mean5
        
        # Audio change detection
        if 'audio_energy_rms' in df.columns:
            df['audio_change_score'] = df['audio_energy_rms'].diff().abs().fillna(0)
        
        # Speaker change
        if 'speaker_id' in df.columns:
            df['speaker_change'] = (df['speaker_id'].diff() != 0).astype(int).fillna(0)
        
        # Visual motion change
        if 'visual_motion' in df.columns:
            df['visual_motion_change'] = df['visual_motion'].diff().abs().fillna(0)
        
        # Face count change
        if 'face_count' in df.columns:
            df['face_count_change'] = df['face_count'].diff().abs().fillna(0)
        
        # Cut timing features (simplified - assume uniform sampling)
        df['time_since_prev'] = 0.1  # 0.1s per frame at 10fps
        df['time_to_next'] = 0.1
        df['cut_duration'] = len(df) * 0.1
        df['position_in_video'] = np.linspace(0, 1, len(df))
        df['cut_density_10s'] = 1.0  # Placeholder
        df['cumulative_adoption_rate'] = 0.23  # Placeholder (target adoption rate)
        
        return df
    
    def load_and_prepare_features(self, enhanced_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare features for model input
        
        Args:
            enhanced_csv: Path to enhanced features CSV
        
        Returns:
            Tuple of (audio_features, visual_features, temporal_features)
        """
        logger.info(f"📊 Loading features from {enhanced_csv}")
        
        df = pd.read_csv(enhanced_csv)
        logger.info(f"   Loaded {len(df)} frames, {len(df.columns)} columns")
        
        # Identify feature columns (more comprehensive)
        # Audio features: basic audio + speaker embedding + acoustic features + text flags
        audio_cols = []
        
        # Basic audio features
        for col in ['audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms']:
            if col in df.columns:
                audio_cols.append(col)
        
        # Speaker features
        if 'speaker_id' in df.columns:
            audio_cols.append('speaker_id')
        
        # Speaker embeddings (192 dimensions)
        speaker_emb_cols = [col for col in df.columns if col.startswith('speaker_emb_')]
        audio_cols.extend(speaker_emb_cols)
        
        # Acoustic features
        acoustic_cols = [col for col in df.columns if col.startswith(('pitch', 'spectral_', 'zcr', 'mfcc_'))]
        audio_cols.extend(acoustic_cols)
        
        # Text features (numeric only - exclude text content columns)
        if 'text_is_active' in df.columns:
            audio_cols.append('text_is_active')
        # NOTE: text_word and telop_text contain actual text strings, not numeric features
        # Only include telop_active (numeric flag)
        if 'telop_active' in df.columns:
            audio_cols.append('telop_active')
        
        # Audio temporal features (moving averages, changes, etc.)
        audio_temporal_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in [
            'audio_energy_rms_ma', 'audio_energy_rms_std', 'audio_energy_rms_diff',
            'audio_change_score', 'silence_to_speech', 'speech_to_silence', 
            'speaker_change', 'pitch_change'
        ])]
        audio_cols.extend(audio_temporal_cols)
        
        # Visual features: basic visual + CLIP + face + visual temporal
        visual_cols = []
        
        # Basic visual features
        for col in ['scene_change', 'visual_motion', 'saliency_x', 'saliency_y']:
            if col in df.columns:
                visual_cols.append(col)
        
        # Face features
        face_cols = [col for col in df.columns if col.startswith('face_')]
        visual_cols.extend(face_cols)
        
        # CLIP features (512 dimensions)
        clip_cols = [col for col in df.columns if col.startswith('clip_') and col[5:].isdigit()]
        visual_cols.extend(sorted(clip_cols, key=lambda x: int(x.split('_')[1])))
        
        # Visual temporal features
        visual_temporal_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in [
            'visual_motion_ma', 'visual_motion_std', 'visual_motion_diff',
            'visual_motion_change', 'face_count_change', 'saliency_movement',
            'clip_sim_'
        ])]
        visual_cols.extend(visual_temporal_cols)
        
        # Temporal features (cut timing)
        temporal_cols = ['time_since_prev', 'time_to_next', 'cut_duration', 'position_in_video', 'cut_density_10s', 'cumulative_adoption_rate']
        
        # Filter existing columns
        audio_cols = [col for col in audio_cols if col in df.columns]
        visual_cols = [col for col in visual_cols if col in df.columns]
        temporal_cols = [col for col in temporal_cols if col in df.columns]
        
        logger.info(f"   Audio features: {len(audio_cols)}")
        logger.info(f"   Visual features: {len(visual_cols)}")
        logger.info(f"   Temporal features: {len(temporal_cols)}")
        
        # Extract features
        audio_features = df[audio_cols].values.astype(np.float32)
        visual_features = df[visual_cols].values.astype(np.float32)
        temporal_features = df[temporal_cols].values.astype(np.float32) if temporal_cols else np.zeros((len(df), 6), dtype=np.float32)
        
        # Pad features to expected dimensions if needed
        expected_audio_dim = self.config['audio_features']
        expected_visual_dim = self.config['visual_features']
        expected_temporal_dim = self.config['temporal_features']
        
        if audio_features.shape[1] < expected_audio_dim:
            logger.warning(f"   ⚠️  Audio features: {audio_features.shape[1]} < {expected_audio_dim}, padding with zeros")
            padding = np.zeros((len(df), expected_audio_dim - audio_features.shape[1]), dtype=np.float32)
            audio_features = np.concatenate([audio_features, padding], axis=1)
        elif audio_features.shape[1] > expected_audio_dim:
            logger.warning(f"   ⚠️  Audio features: {audio_features.shape[1]} > {expected_audio_dim}, truncating")
            audio_features = audio_features[:, :expected_audio_dim]
        
        if visual_features.shape[1] < expected_visual_dim:
            logger.warning(f"   ⚠️  Visual features: {visual_features.shape[1]} < {expected_visual_dim}, padding with zeros")
            padding = np.zeros((len(df), expected_visual_dim - visual_features.shape[1]), dtype=np.float32)
            visual_features = np.concatenate([visual_features, padding], axis=1)
        elif visual_features.shape[1] > expected_visual_dim:
            logger.warning(f"   ⚠️  Visual features: {visual_features.shape[1]} > {expected_visual_dim}, truncating")
            visual_features = visual_features[:, :expected_visual_dim]
        
        if temporal_features.shape[1] < expected_temporal_dim:
            logger.warning(f"   ⚠️  Temporal features: {temporal_features.shape[1]} < {expected_temporal_dim}, padding with zeros")
            padding = np.zeros((len(df), expected_temporal_dim - temporal_features.shape[1]), dtype=np.float32)
            temporal_features = np.concatenate([temporal_features, padding], axis=1)
        elif temporal_features.shape[1] > expected_temporal_dim:
            logger.warning(f"   ⚠️  Temporal features: {temporal_features.shape[1]} > {expected_temporal_dim}, truncating")
            temporal_features = temporal_features[:, :expected_temporal_dim]
        
        # Handle NaN values
        audio_features = np.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        visual_features = np.nan_to_num(visual_features, nan=0.0, posinf=0.0, neginf=0.0)
        temporal_features = np.nan_to_num(temporal_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize with scalers
        if self.audio_scaler is not None:
            audio_features = self.audio_scaler.transform(audio_features)
        if self.visual_scaler is not None:
            visual_features = self.visual_scaler.transform(visual_features)
        if self.temporal_scaler is not None:
            temporal_features = self.temporal_scaler.transform(temporal_features)
        
        logger.info(f"✅ Features prepared")
        logger.info(f"   Audio shape: {audio_features.shape}")
        logger.info(f"   Visual shape: {visual_features.shape}")
        logger.info(f"   Temporal shape: {temporal_features.shape}")
        
        return audio_features, visual_features, temporal_features
    
    def predict(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        temporal_features: np.ndarray,
        smooth: bool = True
    ) -> Dict:
        """
        Predict active labels with Enhanced Cut Selection Model
        
        Args:
            audio_features: (seq_len, 235) array
            visual_features: (seq_len, 543) array
            temporal_features: (seq_len, 6) array
            smooth: Whether to smooth predictions
        
        Returns:
            dict with:
                - confidence_scores: (seq_len,) array
                - active_probs: (seq_len,) array
                - active_binary: (seq_len,) array
                - clips: list of (start_time, end_time) tuples
        """
        seq_len = len(audio_features)
        max_chunk_size = 4500  # Leave margin below 5000 max_len
        
        logger.info(f"🔮 Predicting with Enhanced Cut Selection Model...")
        logger.info(f"   Sequence length: {seq_len} frames ({seq_len/self.fps:.1f}s)")
        
        # Check if we need chunking
        if seq_len > max_chunk_size:
            logger.info(f"   Processing in chunks (max_chunk_size={max_chunk_size})...")
            return self._predict_chunked(audio_features, visual_features, temporal_features, smooth, max_chunk_size)
        
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0).to(self.device)
        visual_tensor = torch.from_numpy(visual_features).float().unsqueeze(0).to(self.device)
        temporal_tensor = torch.from_numpy(temporal_features).float().unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(audio_tensor, visual_tensor, temporal_tensor)
            active_logits = outputs['active']  # (1, seq_len, 2)
            
            # Get probabilities
            probs = torch.softmax(active_logits, dim=-1)
            inactive_probs = probs[..., 0]
            active_probs = probs[..., 1]
            
            # Confidence score: Active - Inactive
            confidence_scores = active_probs - inactive_probs
        
        confidence_scores = confidence_scores.squeeze(0).cpu().numpy()
        active_probs = active_probs.squeeze(0).cpu().numpy()
        
        # Smooth predictions
        if smooth:
            confidence_scores = self._smooth_predictions(confidence_scores, method='savgol', window_size=5)
        
        # Binarize with threshold
        active_binary = (confidence_scores >= self.best_threshold).astype(int)
        
        logger.info(f"   Confidence score - min: {confidence_scores.min():.4f}, max: {confidence_scores.max():.4f}, mean: {confidence_scores.mean():.4f}")
        logger.info(f"   Threshold: {self.best_threshold:.4f}")
        logger.info(f"   Frames above threshold: {(confidence_scores >= self.best_threshold).sum()} / {len(confidence_scores)} ({(confidence_scores >= self.best_threshold).sum()/len(confidence_scores)*100:.1f}%)")
        
        # Extract clips
        clips = self._extract_clips(active_binary)
        logger.info(f"   Extracted {len(clips)} raw clips")
        
        # Filter and select clips
        selected_clips = self._filter_and_select_clips(clips, active_probs)
        
        return {
            'confidence_scores': confidence_scores,
            'active_probs': active_probs,
            'active_binary': active_binary,
            'clips': selected_clips
        }
    
    def _predict_chunked(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        temporal_features: np.ndarray,
        smooth: bool,
        chunk_size: int
    ) -> Dict:
        """Predict on long sequences by processing in chunks"""
        seq_len = len(audio_features)
        overlap = 100  # 10 seconds at 10fps
        
        all_confidence_scores = np.zeros(seq_len)
        all_active_probs = np.zeros(seq_len)
        counts = np.zeros(seq_len)
        
        num_chunks = (seq_len + chunk_size - overlap - 1) // (chunk_size - overlap)
        logger.info(f"   Processing {num_chunks} chunks...")
        
        for i in range(num_chunks):
            start_idx = i * (chunk_size - overlap)
            end_idx = min(start_idx + chunk_size, seq_len)
            
            # Extract chunk
            audio_chunk = audio_features[start_idx:end_idx]
            visual_chunk = visual_features[start_idx:end_idx]
            temporal_chunk = temporal_features[start_idx:end_idx]
            
            # Convert to tensors
            audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)
            visual_tensor = torch.from_numpy(visual_chunk).float().unsqueeze(0).to(self.device)
            temporal_tensor = torch.from_numpy(temporal_chunk).float().unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(audio_tensor, visual_tensor, temporal_tensor)
                active_logits = outputs['active']
                
                probs = torch.softmax(active_logits, dim=-1)
                active_probs = probs[..., 1]
                inactive_probs = probs[..., 0]
                confidence_scores = active_probs - inactive_probs
            
            confidence_scores = confidence_scores.squeeze(0).cpu().numpy()
            active_probs = active_probs.squeeze(0).cpu().numpy()
            
            # Accumulate
            all_confidence_scores[start_idx:end_idx] += confidence_scores
            all_active_probs[start_idx:end_idx] += active_probs
            counts[start_idx:end_idx] += 1
        
        # Average overlapping regions
        all_confidence_scores /= np.maximum(counts, 1)
        all_active_probs /= np.maximum(counts, 1)
        
        # Smooth
        if smooth:
            all_confidence_scores = self._smooth_predictions(all_confidence_scores, method='savgol', window_size=5)
        
        # Binarize
        active_binary = (all_confidence_scores >= self.best_threshold).astype(int)
        
        logger.info(f"   Confidence score - min: {all_confidence_scores.min():.4f}, max: {all_confidence_scores.max():.4f}, mean: {all_confidence_scores.mean():.4f}")
        logger.info(f"   Frames above threshold: {(all_confidence_scores >= self.best_threshold).sum()} / {len(all_confidence_scores)}")
        
        # Extract and select clips
        clips = self._extract_clips(active_binary)
        selected_clips = self._filter_and_select_clips(clips, all_active_probs)
        
        return {
            'confidence_scores': all_confidence_scores,
            'active_probs': all_active_probs,
            'active_binary': active_binary,
            'clips': selected_clips
        }
    
    def _smooth_predictions(self, predictions: np.ndarray, method: str = 'savgol', window_size: int = 5) -> np.ndarray:
        """Smooth predictions to reduce jitter"""
        if len(predictions) < window_size:
            return predictions
        
        if method == 'savgol':
            polyorder = min(2, window_size - 1)
            smoothed = savgol_filter(predictions, window_size, polyorder, mode='nearest')
        elif method == 'moving_average':
            smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='same')
        else:
            smoothed = predictions
        
        return smoothed
    
    def _extract_clips(self, active_binary: np.ndarray) -> List[Tuple[float, float]]:
        """Extract clip segments from binary predictions"""
        clips = []
        in_clip = False
        start_idx = 0
        
        for i, active in enumerate(active_binary):
            if active == 1 and not in_clip:
                start_idx = i
                in_clip = True
            elif active == 0 and in_clip:
                end_idx = i
                start_time = start_idx / self.fps
                end_time = end_idx / self.fps
                clips.append((start_time, end_time))
                in_clip = False
        
        # Handle last clip
        if in_clip:
            end_idx = len(active_binary)
            start_time = start_idx / self.fps
            end_time = end_idx / self.fps
            clips.append((start_time, end_time))
        
        return clips
    
    def _filter_and_select_clips(self, clips: List[Tuple[float, float]], active_probs: np.ndarray) -> List[Tuple[float, float]]:
        """Filter and select top clips based on duration and confidence"""
        if not clips:
            logger.warning(f"   ⚠️  No clips found!")
            return clips
        
        min_duration = self.inference_params['min_clip_duration']
        max_gap = self.inference_params['max_gap_duration']
        target_duration = self.inference_params['target_duration']
        max_duration = self.inference_params['max_duration']
        
        # Filter by minimum duration
        filtered_clips = [(s, e) for s, e in clips if (e - s) >= min_duration]
        logger.info(f"   After min duration filter ({min_duration}s): {len(filtered_clips)} clips")
        
        # Merge clips with small gaps
        if len(filtered_clips) > 1:
            merged_clips = []
            current_start, current_end = filtered_clips[0]
            
            for start, end in filtered_clips[1:]:
                gap = start - current_end
                if gap <= max_gap:
                    # Merge
                    current_end = end
                else:
                    merged_clips.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged_clips.append((current_start, current_end))
            filtered_clips = merged_clips
            logger.info(f"   After gap merging ({max_gap}s): {len(filtered_clips)} clips")
        
        # Calculate scores and select top clips
        clip_scores = []
        for start, end in filtered_clips:
            start_idx = int(start * self.fps)
            end_idx = int(end * self.fps)
            avg_prob = active_probs[start_idx:end_idx].mean()
            duration = end - start
            score = avg_prob * duration
            clip_scores.append((start, end, score, duration))
        
        # Sort by score (descending)
        clip_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Select clips until target duration
        selected = []
        total_duration = 0.0
        
        for start, end, score, duration in clip_scores:
            if total_duration + duration <= max_duration:
                selected.append((start, end))
                total_duration += duration
                
                if total_duration >= target_duration:
                    break
        
        # Sort by start time
        selected.sort(key=lambda x: x[0])
        
        logger.info(f"✅ Selected {len(selected)} clips, total duration: {total_duration:.1f}s (target: {target_duration}s)")
        
        return selected
    
    def generate_xml(self, video_path: str, clips: List[Tuple[float, float]], output_path: str):
        """
        Generate Premiere Pro XML from selected clips
        
        Args:
            video_path: Path to source video
            clips: List of (start_time, end_time) tuples
            output_path: Output XML path
        """
        logger.info(f"📝 Generating Premiere Pro XML...")
        logger.info(f"   Video: {video_path}")
        logger.info(f"   Clips: {len(clips)}")
        logger.info(f"   Output: {output_path}")
        
        # Use direct XML generator
        from src.inference.direct_xml_generator import DirectXMLGenerator
        
        generator = DirectXMLGenerator()
        generator.generate_xml(
            video_path=video_path,
            clips=clips,
            output_path=output_path
        )
        
        logger.info(f"✅ XML generated successfully")
    
    def run(self, video_path: str, output_xml: str):
        """
        Run complete inference pipeline
        
        Args:
            video_path: Path to input video
            output_xml: Path to output XML
        """
        logger.info(f"🎬 Starting Enhanced Cut Selection Inference Pipeline")
        logger.info(f"   Input video: {video_path}")
        logger.info(f"   Output XML: {output_xml}")
        logger.info("")
        
        # Step 1: Extract features
        feature_csv = self.extract_features(video_path)
        logger.info("")
        
        # Step 2: Add temporal features
        enhanced_csv = self.add_temporal_features(feature_csv)
        logger.info("")
        
        # Step 3: Load and prepare features
        audio_features, visual_features, temporal_features = self.load_and_prepare_features(enhanced_csv)
        logger.info("")
        
        # Step 4: Predict
        results = self.predict(audio_features, visual_features, temporal_features)
        logger.info("")
        
        # Step 5: Generate XML
        self.generate_xml(video_path, results['clips'], output_xml)
        logger.info("")
        
        logger.info(f"🎉 Inference pipeline completed successfully!")
        logger.info(f"   Total clips: {len(results['clips'])}")
        logger.info(f"   Total duration: {sum(e-s for s, e in results['clips']):.1f}s")
        logger.info(f"   Output: {output_xml}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Cut Selection Inference')
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--output', type=str, default='outputs/output.xml', help='Output XML path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cpu or cuda)')
    parser.add_argument('--fps', type=float, default=10.0, help='Frame rate for feature extraction')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    pipeline = EnhancedCutSelectionInference(
        model_path=args.model,
        device=args.device,
        fps=args.fps
    )
    
    pipeline.run(args.video, args.output)


if __name__ == '__main__':
    main()
