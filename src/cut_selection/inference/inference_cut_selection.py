"""
Cut Selection Inference Pipeline

Input: Video file
Output: Active predictions (which parts to keep)
"""
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple
from scipy.signal import savgol_filter

from src.cut_selection.archive.cut_model import CutSelectionModel
from src.training.multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor
from src.utils.feature_alignment import FeatureAligner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CutSelectionInference:
    """Cut selection inference pipeline"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        fps: float = 10.0
    ):
        """
        Initialize
        
        Args:
            model_path: Path to trained model
            device: 'cpu' or 'cuda'
            fps: Frame rate for feature extraction
        """
        self.device = device
        self.fps = fps
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.config = checkpoint['config']
        
        # Create model
        self.model = CutSelectionModel(
            audio_features=self.config['audio_features'],
            visual_features=self.config['visual_features'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_encoder_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            fusion_type=self.config.get('fusion_type', 'gated')
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        
        # Load inference parameters
        model_dir = Path(model_path).parent
        inference_params_path = model_dir / 'inference_params.yaml'
        
        if inference_params_path.exists():
            import yaml
            with open(inference_params_path, 'r', encoding='utf-8') as f:
                self.inference_params = yaml.safe_load(f)
            logger.info(f"Loaded inference parameters: {self.inference_params}")
        else:
            # Default parameters
            self.inference_params = {
                'active_threshold': 0.5,
                'target_duration': 90.0,
                'max_duration': 150.0
            }
            logger.warning(f"Inference parameters not found, using defaults")
        
        # Feature aligner
        self.aligner = FeatureAligner(tolerance=0.05)
        
        # Load preprocessors
        audio_preprocessor_path = model_dir / 'audio_preprocessor.pkl'
        visual_preprocessor_path = model_dir / 'visual_preprocessor.pkl'

        if audio_preprocessor_path.exists():
            # Try loading as StandardScaler (from sklearn)
            import pickle
            try:
                with open(audio_preprocessor_path, 'rb') as f:
                    self.audio_preprocessor = pickle.load(f)
                logger.info(f"Loaded audio preprocessor (StandardScaler)")
            except Exception as e:
                logger.warning(f"Failed to load audio preprocessor: {e}")
                self.audio_preprocessor = None
        else:
            self.audio_preprocessor = None
            logger.warning(f"Audio preprocessor not found")

        if visual_preprocessor_path.exists():
            # Try loading as StandardScaler (from sklearn)
            import pickle
            try:
                with open(visual_preprocessor_path, 'rb') as f:
                    self.visual_preprocessor = pickle.load(f)
                logger.info(f"Loaded visual preprocessor (StandardScaler)")
            except Exception as e:
                logger.warning(f"Failed to load visual preprocessor: {e}")
                self.visual_preprocessor = None
        else:
            self.visual_preprocessor = None
            logger.warning(f"Visual preprocessor not found")
    
    def smooth_predictions(self, predictions: np.ndarray, method: str = 'savgol', window_size: int = 5) -> np.ndarray:
        """
        Smooth predictions to reduce jitter
        
        Args:
            predictions: (seq_len,) array of probabilities
            method: 'savgol', 'moving_average', or 'ema'
            window_size: Window size for smoothing
        
        Returns:
            Smoothed predictions
        """
        if len(predictions) < window_size:
            return predictions
        
        if method == 'savgol':
            # Savitzky-Golay filter
            polyorder = min(2, window_size - 1)
            smoothed = savgol_filter(predictions, window_size, polyorder, mode='nearest')
        elif method == 'moving_average':
            # Moving average
            smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='same')
        elif method == 'ema':
            # Exponential moving average
            alpha = 2.0 / (window_size + 1)
            smoothed = np.zeros_like(predictions)
            smoothed[0] = predictions[0]
            for i in range(1, len(predictions)):
                smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i-1]
        else:
            smoothed = predictions
        
        return smoothed

    def _predict_chunked(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        smooth: bool,
        chunk_size: int
    ) -> Dict:
        """
        Predict on long sequences by processing in chunks with overlap

        Args:
            audio_features: (seq_len, audio_dim) array (already normalized)
            visual_features: (seq_len, visual_dim) array (already normalized)
            smooth: Whether to smooth predictions
            chunk_size: Maximum chunk size

        Returns:
            Same as predict()
        """
        seq_len = len(audio_features)
        overlap = 100  # Overlap between chunks (10 seconds at 10fps)

        # Initialize arrays for full sequence
        all_confidence_scores = np.zeros(seq_len)
        all_active_probs = np.zeros(seq_len)
        counts = np.zeros(seq_len)  # For averaging overlapping regions

        # Process in chunks
        num_chunks = (seq_len + chunk_size - overlap - 1) // (chunk_size - overlap)
        logger.info(f"    Processing {num_chunks} chunks (chunk_size={chunk_size}, overlap={overlap})...")

        for i in range(num_chunks):
            start_idx = i * (chunk_size - overlap)
            end_idx = min(start_idx + chunk_size, seq_len)

            # Extract chunk
            audio_chunk = audio_features[start_idx:end_idx]
            visual_chunk = visual_features[start_idx:end_idx]

            # Convert to tensors
            audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)
            visual_tensor = torch.from_numpy(visual_chunk).float().unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(audio_tensor, visual_tensor)
                active_logits = outputs['active']  # (1, chunk_len, 2)

                probs = torch.softmax(active_logits, dim=-1)
                inactive_probs = probs[..., 0]
                active_probs = probs[..., 1]
                confidence_scores = active_probs - inactive_probs

            # Convert to numpy
            confidence_scores = confidence_scores.squeeze(0).cpu().numpy()
            active_probs = active_probs.squeeze(0).cpu().numpy()

            # Accumulate (average overlapping regions)
            all_confidence_scores[start_idx:end_idx] += confidence_scores
            all_active_probs[start_idx:end_idx] += active_probs
            counts[start_idx:end_idx] += 1

        # Average overlapping regions
        all_confidence_scores /= np.maximum(counts, 1)
        all_active_probs /= np.maximum(counts, 1)

        # Smooth predictions (using confidence scores)
        if smooth:
            all_confidence_scores = self.smooth_predictions(all_confidence_scores, method='savgol', window_size=5)

        # Binarize with threshold
        threshold = self.inference_params.get('confidence_threshold', 0.0)
        active_binary = (all_confidence_scores >= threshold).astype(int)

        logger.info(f"  Prediction statistics:")
        logger.info(f"    Active prob - min: {all_active_probs.min():.4f}, max: {all_active_probs.max():.4f}, mean: {all_active_probs.mean():.4f}")
        logger.info(f"    Confidence score - min: {all_confidence_scores.min():.4f}, max: {all_confidence_scores.max():.4f}, mean: {all_confidence_scores.mean():.4f}")
        logger.info(f"    Threshold: {threshold}")
        logger.info(f"    Frames above threshold: {(all_confidence_scores >= threshold).sum()} / {len(all_confidence_scores)}")

        # Extract clips
        clips = self.extract_clips(active_binary)

        logger.info(f"  Extracted {len(clips)} clips before selection")

        # Select top clips
        selected_clips = self.select_top_clips(clips, all_active_probs)

        return {
            'confidence_scores': all_confidence_scores,
            'active_probs': all_active_probs,
            'active_binary': active_binary,
            'clips': selected_clips
        }

    def predict(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray,
        smooth: bool = True
    ) -> Dict:
        """
        Predict active labels

        Args:
            audio_features: (seq_len, audio_dim) array
            visual_features: (seq_len, visual_dim) array
            smooth: Whether to smooth predictions

        Returns:
            dict with:
                - active_probs: (seq_len,) array of probabilities
                - active_binary: (seq_len,) array of binary predictions
                - clips: list of (start_time, end_time) tuples
        """
        # Normalize features
        if self.audio_preprocessor is not None:
            audio_features = self.audio_preprocessor.transform(audio_features)

        if self.visual_preprocessor is not None:
            visual_features = self.visual_preprocessor.transform(visual_features)

        seq_len = len(audio_features)
        max_chunk_size = 4500  # Leave some margin below 5000 max_len

        # Check if we need to process in chunks
        if seq_len > max_chunk_size:
            logger.info(f"  Sequence length ({seq_len}) exceeds max chunk size ({max_chunk_size}), processing in chunks...")
            return self._predict_chunked(audio_features, visual_features, smooth, max_chunk_size)

        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0).to(self.device)
        visual_tensor = torch.from_numpy(visual_features).float().unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(audio_tensor, visual_tensor)
            active_logits = outputs['active']  # (1, seq_len, 2)

            # Get probabilities for both classes
            probs = torch.softmax(active_logits, dim=-1)  # (1, seq_len, 2)
            inactive_probs = probs[..., 0]  # (1, seq_len) - Inactive probability
            active_probs = probs[..., 1]    # (1, seq_len) - Active probability

            # Confidence score: Active - Inactive
            # Positive = more likely to be active
            # Negative = more likely to be inactive
            confidence_scores = active_probs - inactive_probs  # (1, seq_len)

        confidence_scores = confidence_scores.squeeze(0).cpu().numpy()  # (seq_len,)
        active_probs = active_probs.squeeze(0).cpu().numpy()  # (seq_len,)
        
        # Smooth predictions (using confidence scores)
        if smooth:
            confidence_scores = self.smooth_predictions(confidence_scores, method='savgol', window_size=5)
        
        # Binarize with threshold (using confidence scores)
        # threshold=0 means: active_prob > inactive_prob
        threshold = self.inference_params.get('confidence_threshold', 0.0)
        active_binary = (confidence_scores >= threshold).astype(int)

        logger.info(f"  Prediction statistics:")
        logger.info(f"    Active prob - min: {active_probs.min():.4f}, max: {active_probs.max():.4f}, mean: {active_probs.mean():.4f}")
        logger.info(f"    Confidence score - min: {confidence_scores.min():.4f}, max: {confidence_scores.max():.4f}, mean: {confidence_scores.mean():.4f}")
        logger.info(f"    Threshold: {threshold}")
        logger.info(f"    Frames above threshold: {(confidence_scores >= threshold).sum()} / {len(confidence_scores)}")

        # Extract clips
        clips = self.extract_clips(active_binary)

        logger.info(f"  Extracted {len(clips)} clips before selection")

        # Select top clips based on active probabilities (for positive scoring)
        selected_clips = self.select_top_clips(clips, active_probs)

        return {
            'confidence_scores': confidence_scores,  # Active - Inactive
            'active_probs': active_probs,
            'active_binary': active_binary,
            'clips': selected_clips
        }
    
    def extract_clips(self, active_binary: np.ndarray) -> list:
        """
        Extract clip segments from binary predictions
        
        Args:
            active_binary: (seq_len,) array of 0/1
        
        Returns:
            List of (start_time, end_time) tuples
        """
        clips = []
        in_clip = False
        start_idx = 0
        
        for i, active in enumerate(active_binary):
            if active == 1 and not in_clip:
                # Start of clip
                start_idx = i
                in_clip = True
            elif active == 0 and in_clip:
                # End of clip
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
    
    def select_top_clips(self, clips: list, active_probs: np.ndarray) -> list:
        """
        Select top clips based on confidence and duration constraints
        
        Args:
            clips: List of (start_time, end_time) tuples
            active_probs: (seq_len,) array of probabilities
        
        Returns:
            Selected clips
        """
        if not clips:
            return clips
        
        target_duration = self.inference_params['target_duration']
        max_duration = self.inference_params['max_duration']
        
        # Calculate score for each clip (average probability * duration)
        clip_scores = []
        for start, end in clips:
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
        
        logger.info(f"Selected {len(selected)} clips, total duration: {total_duration:.1f}s")
        
        return selected


def main():
    """Test inference"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_clips.txt', help='Output file')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    # Create inference pipeline
    pipeline = CutSelectionInference(args.model, device=args.device)
    
    # TODO: Extract features from video
    # For now, this is a placeholder
    logger.info("Feature extraction not implemented yet")
    logger.info("Use src/data_preparation/extract_video_features.py to extract features first")


if __name__ == '__main__':
    main()
