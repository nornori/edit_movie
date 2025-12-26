"""
Cut Selection Module

Stage 1 of the video editing pipeline:
Predicts which parts of the video should be kept (active) based on audio and visual features only.
"""

from .models import EnhancedCutSelectionModel
from .datasets import EnhancedCutSelectionDatasetFullVideo, EnhancedCutSelectionDataset, collate_fn_fullvideo
from .utils import CombinedCutSelectionLoss, FocalLoss, PositionalEncoding
from .evaluation import EnsemblePredictor
from .inference import CutSelectionInference

__all__ = [
    'EnhancedCutSelectionModel',
    'EnhancedCutSelectionDatasetFullVideo',
    'EnhancedCutSelectionDataset',
    'collate_fn_fullvideo',
    'CombinedCutSelectionLoss',
    'FocalLoss',
    'PositionalEncoding',
    'EnsemblePredictor',
    'CutSelectionInference'
]
