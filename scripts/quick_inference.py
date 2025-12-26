"""
Quick inference script for Enhanced Cut Selection Model
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/quick_inference.py <model_path> <video_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    video_path = sys.argv[2]
    
    print(f"📹 Video: {video_path}")
    print(f"🤖 Model: {model_path}")
    print()
    print("⚠️  This is a simplified inference script.")
    print("⚠️  Full inference pipeline (feature extraction + XML generation) is not yet implemented for Enhanced model.")
    print()
    print("To use this model, you need to:")
    print("1. Extract features from the video using extract_video_features_parallel.py")
    print("2. Add temporal features using add_temporal_features.py")
    print("3. Load features and run model prediction")
    print("4. Generate XML from predictions")
    print()
    print("For now, loading model to verify it works...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Audio features: {config['audio_features']}")
    print(f"   Visual features: {config['visual_features']}")
    print(f"   Temporal features: {config['temporal_features']}")
    print(f"   Best F1: {checkpoint.get('best_f1', 'N/A')}")
    print(f"   Best threshold: {checkpoint.get('best_threshold', 'N/A')}")
    
    # Create model
    model = EnhancedCutSelectionModel(
        audio_features=config['audio_features'],
        visual_features=config['visual_features'],
        temporal_features=config['temporal_features'],
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.15)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\n✅ Model initialized and ready!")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

if __name__ == '__main__':
    main()
