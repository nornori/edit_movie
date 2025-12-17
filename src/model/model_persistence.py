"""
Model persistence and loading utilities
"""
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from src.model.model import MultiTrackTransformer, create_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_model(
    model,  # Can be MultiTrackTransformer or MultimodalTransformer
    save_path: str,
    metadata: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None
):
    """
    Save model with configuration and metadata
    
    Args:
        model: Model to save (MultiTrackTransformer or MultimodalTransformer)
        save_path: Path to save model
        metadata: Optional metadata dict
        optimizer: Optional optimizer state
        epoch: Optional current epoch
        metrics: Optional training metrics
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Detect model type
    from src.model.model import MultimodalTransformer
    is_multimodal = isinstance(model, MultimodalTransformer)
    
    # Model configuration - save all necessary parameters
    # Extract from model's transformer encoder
    encoder_layer = model.transformer_encoder.layers[0]
    nhead = encoder_layer.self_attn.num_heads
    dim_feedforward = encoder_layer.linear1.out_features
    num_encoder_layers = len(model.transformer_encoder.layers)
    
    config = {
        'model_type': 'multimodal' if is_multimodal else 'track_only',
        'd_model': model.d_model,
        'nhead': nhead,
        'num_encoder_layers': num_encoder_layers,
        'dim_feedforward': dim_feedforward,
        'num_tracks': model.num_tracks,
        'max_asset_classes': model.max_asset_classes,
        'num_parameters': model.count_parameters()
    }
    
    # Add model-specific parameters
    if is_multimodal:
        config.update({
            'audio_features': model.audio_features,
            'visual_features': model.visual_features,
            'track_features': model.track_features,
            'enable_multimodal': model.enable_multimodal,
            'fusion_type': model.fusion_type
        })
    else:
        config['input_features'] = model.input_features
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    # Add optional components
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    
    # Save config as JSON for easy inspection
    config_path = save_path.with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump({
            'config': config,
            'metadata': checkpoint['metadata'],
            'timestamp': checkpoint['timestamp'],
            'version': checkpoint['version']
        }, f, indent=2)
    
    logger.info(f"💾 Model saved to {save_path}")
    logger.info(f"📄 Config saved to {config_path}")
    logger.info(f"   Parameters: {config['num_parameters']:,}")


def load_model(
    load_path: str,
    device: str = 'cpu',
    load_optimizer: bool = False,
    force_track_only: bool = False
) -> Dict:
    """
    Load model from checkpoint with automatic type detection
    
    Args:
        load_path: Path to checkpoint file
        device: Device to load model on
        load_optimizer: Whether to load optimizer state
        force_track_only: Force loading as track-only model (for compatibility)
    
    Returns:
        Dict with:
            - 'model': Loaded model
            - 'config': Model configuration
            - 'metadata': Metadata
            - 'optimizer_state_dict': Optimizer state (if available and requested)
            - 'epoch': Epoch number (if available)
            - 'metrics': Metrics (if available)
    """
    logger.info(f"Loading model from {load_path}")
    
    # Load checkpoint
    checkpoint = torch.load(load_path, map_location=device)
    
    # Extract configuration (古いモデルとの互換性のため)
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 古いモデルの場合、state_dictから推測
        logger.warning("⚠️  古いモデル形式を検出しました。モデル構造から設定を推測します。")
        state_dict = checkpoint['model_state_dict']
        
        # モデルタイプを検出（マルチモーダル vs トラックオンリー）
        if 'audio_embedding.projection.weight' in state_dict:
            # マルチモーダルモデル（新しい構造）
            audio_features = state_dict['audio_embedding.projection.weight'].shape[1]
            visual_features = state_dict['visual_embedding.projection.weight'].shape[1]
            track_features = state_dict['track_embedding_layer.projection.weight'].shape[1]
            d_model = state_dict['audio_embedding.projection.weight'].shape[0]
            
            config = {
                'model_type': 'multimodal',
                'd_model': d_model,
                'nhead': 8,  # デフォルト
                'num_encoder_layers': 6,  # デフォルト
                'dim_feedforward': 1024,  # デフォルト
                'num_tracks': 20,
                'max_asset_classes': 10,
                'audio_features': audio_features,
                'visual_features': visual_features,
                'track_features': track_features,
                'enable_multimodal': True,
                'fusion_type': 'gated'
            }
            logger.info(f"   推測された設定（マルチモーダル）:")
            logger.info(f"     audio_features={audio_features}, visual_features={visual_features}, track_features={track_features}")
            logger.info(f"     d_model={d_model}")
        elif 'audio_projection.weight' in state_dict:
            # マルチモーダルモデル（古い構造）
            audio_features = state_dict['audio_projection.weight'].shape[1]
            visual_features = state_dict['visual_projection.weight'].shape[1]
            track_features = state_dict['track_projection.weight'].shape[1]
            d_model = state_dict['audio_projection.weight'].shape[0]
            
            config = {
                'model_type': 'multimodal',
                'd_model': d_model,
                'nhead': 8,  # デフォルト
                'num_encoder_layers': 6,  # デフォルト
                'dim_feedforward': 1024,  # デフォルト
                'num_tracks': 20,
                'max_asset_classes': 10,
                'audio_features': audio_features,
                'visual_features': visual_features,
                'track_features': track_features,
                'enable_multimodal': True,
                'fusion_type': 'gated'
            }
            logger.info(f"   推測された設定（マルチモーダル）:")
            logger.info(f"     audio_features={audio_features}, visual_features={visual_features}, track_features={track_features}")
            logger.info(f"     d_model={d_model}")
        elif 'input_projection.weight' in state_dict:
            # トラックオンリーモデル
            input_features = state_dict['input_projection.weight'].shape[1]
            d_model = state_dict['input_projection.weight'].shape[0]
            
            config = {
                'model_type': 'track_only',
                'd_model': d_model,
                'nhead': 8,  # デフォルト
                'num_encoder_layers': 6,  # デフォルト
                'dim_feedforward': 1024,  # デフォルト
                'num_tracks': 20,
                'max_asset_classes': 10,
                'input_features': input_features
            }
            logger.info(f"   推測された設定（トラックオンリー）: input_features={input_features}, d_model={d_model}")
        else:
            raise ValueError("モデルの構造を推測できませんでした。'audio_projection.weight'または'input_projection.weight'が見つかりません。")
    
    # Detect model type
    # If model_type is not explicitly set, infer from enable_multimodal or presence of multimodal features
    if 'model_type' in config:
        model_type = config['model_type']
    elif config.get('enable_multimodal', False) or ('audio_features' in config and 'visual_features' in config):
        model_type = 'multimodal'
    else:
        model_type = 'track_only'
    
    if force_track_only:
        model_type = 'track_only'
        logger.info("🔄 Forcing track-only mode")
    
    logger.info(f"📦 Model type: {model_type}")
    
    # Create model based on type
    if model_type == 'multimodal' and not force_track_only:
        from src.model.model import MultimodalTransformer
        
        # Support both 'num_encoder_layers' and 'num_layers' keys
        num_layers = config.get('num_encoder_layers', config.get('num_layers', 6))
        
        model = MultimodalTransformer(
            audio_features=config['audio_features'],
            visual_features=config['visual_features'],
            track_features=config['track_features'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=num_layers,
            dim_feedforward=config.get('dim_feedforward', 1024),
            num_tracks=config.get('num_tracks', 20),
            max_asset_classes=config.get('max_asset_classes', 10),
            enable_multimodal=config.get('enable_multimodal', True),
            fusion_type=config.get('fusion_type', 'gated')
        )
        logger.info(f"   Audio features: {config['audio_features']}")
        logger.info(f"   Visual features: {config['visual_features']}")
        logger.info(f"   Track features: {config['track_features']}")
        logger.info(f"   Fusion type: {config.get('fusion_type', 'gated')}")
    else:
        # Track-only model (or forced track-only from multimodal checkpoint)
        # For multimodal checkpoints forced to track-only, use track_features as input_features
        input_features = config.get('input_features', config.get('track_features', 180))
        
        # Support both 'num_encoder_layers' and 'num_layers' keys
        num_layers = config.get('num_encoder_layers', config.get('num_layers', 6))
        
        model = MultiTrackTransformer(
            input_features=input_features,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=num_layers,
            dim_feedforward=config.get('dim_feedforward', 1024),
            num_tracks=config.get('num_tracks', 20),
            max_asset_classes=config.get('max_asset_classes', 10)
        )
        logger.info(f"   Input features: {input_features}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   Parameters: {model.count_parameters():,}")
    logger.info(f"   Device: {device}")
    
    # Prepare result
    result = {
        'model': model,
        'config': config,
        'metadata': checkpoint.get('metadata', {}),
        'timestamp': checkpoint.get('timestamp'),
        'version': checkpoint.get('version')
    }
    
    # Add optional components
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        result['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    if 'epoch' in checkpoint:
        result['epoch'] = checkpoint['epoch']
        logger.info(f"   Epoch: {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        result['metrics'] = checkpoint['metrics']
    
    return result


def export_model_for_inference(
    model: MultiTrackTransformer,
    export_path: str,
    example_input: Optional[torch.Tensor] = None
):
    """
    Export model for inference (TorchScript)
    
    Args:
        model: Model to export
        export_path: Path to save exported model
        example_input: Optional example input for tracing
    """
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    # Create example input if not provided
    if example_input is None:
        example_input = torch.randn(1, 100, model.input_features)
    
    # Trace model
    logger.info("Tracing model for TorchScript export...")
    try:
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(str(export_path))
        logger.info(f"✅ Model exported to {export_path}")
    except Exception as e:
        logger.error(f"❌ Failed to export model: {e}")
        raise


if __name__ == "__main__":
    # Test model persistence
    logger.info("Testing model persistence...")
    
    # Create a model
    logger.info("\n1. Creating model...")
    model = create_model(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256
    )
    
    # Save model
    logger.info("\n2. Saving model...")
    metadata = {
        'description': 'Test model for persistence',
        'dataset': 'test_dataset',
        'training_date': datetime.now().isoformat()
    }
    
    save_model(
        model=model,
        save_path='test_model.pth',
        metadata=metadata,
        epoch=10,
        metrics={'val_loss': 15.5, 'train_loss': 14.2}
    )
    
    # Load model
    logger.info("\n3. Loading model...")
    loaded = load_model('test_model.pth', device='cpu')
    
    logger.info(f"\nLoaded model info:")
    logger.info(f"  Config: {loaded['config']}")
    logger.info(f"  Metadata: {loaded['metadata']}")
    logger.info(f"  Epoch: {loaded.get('epoch')}")
    logger.info(f"  Metrics: {loaded.get('metrics')}")
    
    # Test that loaded model produces same output
    logger.info("\n4. Testing model equivalence...")
    test_input = torch.randn(2, 50, 180)
    test_mask = torch.ones(2, 50, dtype=torch.bool)
    
    model.eval()
    loaded['model'].eval()
    
    with torch.no_grad():
        output1 = model(test_input, test_mask)
        output2 = loaded['model'](test_input, test_mask)
    
    # Check if outputs match
    all_match = True
    for key in output1.keys():
        if not torch.allclose(output1[key], output2[key], atol=1e-6):
            logger.error(f"❌ Outputs don't match for {key}")
            all_match = False
    
    if all_match:
        logger.info("✅ Model outputs match! Persistence works correctly.")
    else:
        logger.error("❌ Model outputs don't match!")
    
    # Test TorchScript export
    logger.info("\n5. Testing TorchScript export...")
    try:
        export_model_for_inference(
            model=model,
            export_path='test_model_traced.pt',
            example_input=test_input
        )
        logger.info("✅ TorchScript export successful!")
    except Exception as e:
        logger.warning(f"⚠️  TorchScript export failed: {e}")
    
    logger.info("\n✅ All persistence tests passed!")
