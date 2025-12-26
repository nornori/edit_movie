"""
Ensemble Predictor for Cut Selection

Combines predictions from multiple K-Fold models for improved accuracy.
Uses voting and confidence averaging strategies.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import yaml

from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple K-Fold models
    
    Strategies:
    - Soft voting: Average confidence scores from all models
    - Hard voting: Majority vote from all models
    - Weighted voting: Weight models by their validation F1 score
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        n_folds: int = 5,
        device: str = 'cuda',
        voting_strategy: str = 'soft'  # 'soft', 'hard', or 'weighted'
    ):
        """
        Initialize ensemble predictor
        
        Args:
            checkpoint_dir: Directory containing fold models
            n_folds: Number of folds
            device: Device to run inference on
            voting_strategy: Voting strategy ('soft', 'hard', 'weighted')
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.n_folds = n_folds
        self.device = device
        self.voting_strategy = voting_strategy
        
        self.models = []
        self.model_weights = []
        
        logger.info(f"🎯 Initializing Ensemble Predictor")
        logger.info(f"   Checkpoint dir: {checkpoint_dir}")
        logger.info(f"   Number of folds: {n_folds}")
        logger.info(f"   Voting strategy: {voting_strategy}")
        logger.info(f"   Device: {device}")
        
        self._load_models()
    
    def _load_models(self):
        """Load all fold models"""
        logger.info(f"\n📦 Loading {self.n_folds} fold models...")
        
        for fold in range(1, self.n_folds + 1):
            model_path = self.checkpoint_dir / f"fold_{fold}_best_model.pth"
            
            if not model_path.exists():
                logger.warning(f"⚠️  Model not found: {model_path}")
                continue
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint['config']
            val_metrics = checkpoint['val_metrics']
            
            # Create model
            model = EnhancedCutSelectionModel(
                audio_features=config['audio_features'],
                visual_features=config['visual_features'],
                temporal_features=config.get('temporal_features', 6),
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_encoder_layers=config['num_encoder_layers'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout']
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.models.append(model)
            
            # Weight by validation F1 score for weighted voting
            f1_score = val_metrics.get('f1', 0.5)
            self.model_weights.append(f1_score)
            
            logger.info(f"   ✅ Fold {fold}: F1={f1_score:.4f}, Epoch={checkpoint['epoch']}")
        
        # Normalize weights
        if self.voting_strategy == 'weighted':
            total_weight = sum(self.model_weights)
            self.model_weights = [w / total_weight for w in self.model_weights]
            logger.info(f"\n📊 Model weights (normalized): {[f'{w:.3f}' for w in self.model_weights]}")
        
        logger.info(f"\n✅ Loaded {len(self.models)} models successfully")
    
    def predict(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        temporal: torch.Tensor,
        threshold: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Predict using ensemble
        
        Args:
            audio: (batch, seq_len, audio_dim)
            visual: (batch, seq_len, visual_dim)
            temporal: (batch, seq_len, temporal_dim)
            threshold: Confidence threshold for binary prediction
        
        Returns:
            dict with:
                - predictions: (batch, seq_len) - binary predictions
                - confidence: (batch, seq_len) - confidence scores
                - probabilities: (batch, seq_len, 2) - class probabilities
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded!")
        
        batch_size, seq_len = audio.shape[0], audio.shape[1]
        
        # Collect predictions from all models
        all_logits = []
        all_probs = []
        all_confidences = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                outputs = model(audio, visual, temporal)
                logits = outputs['active']  # (batch, seq_len, 2)
                probs = torch.softmax(logits, dim=-1)  # (batch, seq_len, 2)
                confidence = probs[..., 1] - probs[..., 0]  # (batch, seq_len)
                
                all_logits.append(logits)
                all_probs.append(probs)
                all_confidences.append(confidence)
        
        # Ensemble predictions based on strategy
        if self.voting_strategy == 'soft':
            # Average probabilities
            ensemble_probs = torch.stack(all_probs).mean(dim=0)  # (batch, seq_len, 2)
            ensemble_confidence = ensemble_probs[..., 1] - ensemble_probs[..., 0]
            
        elif self.voting_strategy == 'hard':
            # Majority vote
            hard_predictions = []
            for conf in all_confidences:
                hard_pred = (conf >= threshold).long()
                hard_predictions.append(hard_pred)
            
            hard_predictions = torch.stack(hard_predictions)  # (n_models, batch, seq_len)
            ensemble_predictions = (hard_predictions.float().mean(dim=0) >= 0.5).long()
            
            # For confidence, use average of confidences
            ensemble_confidence = torch.stack(all_confidences).mean(dim=0)
            
            # Reconstruct probabilities from hard predictions
            ensemble_probs = torch.zeros(batch_size, seq_len, 2, device=self.device)
            ensemble_probs[..., 1] = ensemble_predictions.float()
            ensemble_probs[..., 0] = 1 - ensemble_predictions.float()
            
        elif self.voting_strategy == 'weighted':
            # Weighted average by F1 scores
            weights = torch.tensor(self.model_weights, device=self.device).view(-1, 1, 1, 1)
            all_probs_stacked = torch.stack(all_probs)  # (n_models, batch, seq_len, 2)
            ensemble_probs = (all_probs_stacked * weights).sum(dim=0)  # (batch, seq_len, 2)
            ensemble_confidence = ensemble_probs[..., 1] - ensemble_probs[..., 0]
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        # Final predictions
        ensemble_predictions = (ensemble_confidence >= threshold).long()
        
        return {
            'predictions': ensemble_predictions,
            'confidence': ensemble_confidence,
            'probabilities': ensemble_probs
        }
    
    def predict_batch(
        self,
        dataloader,
        threshold: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Predict on entire dataset
        
        Args:
            dataloader: DataLoader with batches
            threshold: Confidence threshold
        
        Returns:
            dict with numpy arrays:
                - predictions: (N, seq_len)
                - confidence: (N, seq_len)
                - labels: (N, seq_len) - ground truth if available
        """
        all_predictions = []
        all_confidences = []
        all_labels = []
        
        for batch in dataloader:
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            temporal = batch['temporal'].to(self.device)
            labels = batch['active']
            
            results = self.predict(audio, visual, temporal, threshold)
            
            all_predictions.append(results['predictions'].cpu().numpy())
            all_confidences.append(results['confidence'].cpu().numpy())
            all_labels.append(labels.numpy())
        
        return {
            'predictions': np.concatenate(all_predictions, axis=0),
            'confidence': np.concatenate(all_confidences, axis=0),
            'labels': np.concatenate(all_labels, axis=0)
        }
    
    def evaluate(
        self,
        dataloader,
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Evaluate ensemble on dataset
        
        Args:
            dataloader: DataLoader with batches
            threshold: Confidence threshold
        
        Returns:
            dict with metrics
        """
        results = self.predict_batch(dataloader, threshold)
        
        predictions = results['predictions'].flatten()
        labels = results['labels'].flatten()
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        tn = np.sum((predictions == 0) & (labels == 0))
        
        accuracy = (predictions == labels).mean()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'threshold': threshold
        }
        
        logger.info(f"\n📊 Ensemble Evaluation Results:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1 Score: {f1:.4f}")
        logger.info(f"   Specificity: {specificity:.4f}")
        logger.info(f"   Threshold: {threshold:.4f}")
        
        return metrics
    
    def find_optimal_threshold(
        self,
        dataloader,
        min_recall: float = 0.71
    ) -> float:
        """
        Find optimal threshold for ensemble
        
        Args:
            dataloader: Validation dataloader
            min_recall: Minimum recall constraint
        
        Returns:
            Optimal threshold
        """
        logger.info(f"\n🔍 Finding optimal threshold (min_recall={min_recall:.2%})...")
        
        # Get predictions with threshold=0 (use confidence scores)
        results = self.predict_batch(dataloader, threshold=-999)  # Get all predictions
        
        confidences = results['confidence'].flatten()
        labels = results['labels'].flatten()
        
        # Find optimal threshold
        from sklearn.metrics import precision_recall_curve
        
        precisions, recalls, thresholds = precision_recall_curve(labels, confidences)
        
        best_threshold = 0.0
        best_f1 = 0.0
        
        for prec, rec, thresh in zip(precisions, recalls, thresholds):
            if rec < min_recall:
                continue
            
            f1 = 2 * prec * rec / (prec + rec + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        logger.info(f"   ✅ Optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
        
        return best_threshold


if __name__ == "__main__":
    # Test ensemble predictor
    print("Testing EnsemblePredictor...")
    
    checkpoint_dir = "checkpoints_cut_selection_kfold_enhanced"
    
    # Create ensemble
    ensemble = EnsemblePredictor(
        checkpoint_dir=checkpoint_dir,
        n_folds=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        voting_strategy='soft'
    )
    
    # Test prediction
    batch_size = 2
    seq_len = 1000
    audio = torch.randn(batch_size, seq_len, 235)
    visual = torch.randn(batch_size, seq_len, 543)
    temporal = torch.randn(batch_size, seq_len, 6)
    
    if torch.cuda.is_available():
        audio = audio.cuda()
        visual = visual.cuda()
        temporal = temporal.cuda()
    
    results = ensemble.predict(audio, visual, temporal, threshold=0.0)
    
    print(f"\nPredictions shape: {results['predictions'].shape}")
    print(f"Confidence shape: {results['confidence'].shape}")
    print(f"Probabilities shape: {results['probabilities'].shape}")
    
    print("\n✅ EnsemblePredictor test passed!")
