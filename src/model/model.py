"""
Multi-Track Transformer model for video editing prediction
"""
import torch
import torch.nn as nn
import math
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TrackEmbedding(nn.Module):
    """Embedding layer for track-specific information"""
    
    def __init__(self, num_tracks: int = 20, d_model: int = 256):
        super().__init__()
        self.track_embedding = nn.Embedding(num_tracks, d_model)
        self.num_tracks = num_tracks
    
    def forward(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create track embeddings for the batch
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device to create tensor on
        
        Returns:
            Tensor of shape (batch, seq_len, num_tracks, d_model)
        """
        # Create track indices [0, 1, 2, ..., 19]
        track_ids = torch.arange(self.num_tracks, device=device)
        
        # Get embeddings for all tracks
        track_emb = self.track_embedding(track_ids)  # (num_tracks, d_model)
        
        # Expand to batch and sequence dimensions
        track_emb = track_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, num_tracks, d_model)
        track_emb = track_emb.expand(batch_size, seq_len, -1, -1)  # (batch, seq_len, num_tracks, d_model)
        
        return track_emb



class MultiTrackTransformer(nn.Module):
    """
    Multi-Track Transformer for video editing prediction
    
    Predicts 12 parameters for each of 20 tracks:
    - active (binary classification)
    - asset_id (classification)
    - scale, x, y, anchor_x, anchor_y, rotation, crop_l, crop_r, crop_t, crop_b (regression)
    """
    
    def __init__(
        self,
        input_features: int = 180,  # 20 tracks × 9 parameters
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_tracks: int = 20,
        max_asset_classes: int = 10
    ):
        super().__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        self.num_tracks = num_tracks
        self.max_asset_classes = max_asset_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Track embedding
        self.track_embedding = TrackEmbedding(num_tracks, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output heads for each parameter type
        # Each track has 12 parameters, so we need 12 output heads
        
        # Per-track output projection
        self.track_projection = nn.Linear(d_model, d_model)
        
        # Classification heads
        self.active_head = nn.Linear(d_model, 2)  # Binary: active or not
        self.asset_head = nn.Linear(d_model, max_asset_classes)  # Multi-class: asset ID
        
        # Regression heads
        self.scale_head = nn.Linear(d_model, 1)
        self.pos_x_head = nn.Linear(d_model, 1)
        self.pos_y_head = nn.Linear(d_model, 1)
        self.anchor_x_head = nn.Linear(d_model, 1)
        self.anchor_y_head = nn.Linear(d_model, 1)
        self.rotation_head = nn.Linear(d_model, 1)
        self.crop_l_head = nn.Linear(d_model, 1)
        self.crop_r_head = nn.Linear(d_model, 1)
        self.crop_t_head = nn.Linear(d_model, 1)
        self.crop_b_head = nn.Linear(d_model, 1)
        
        self._init_weights()
        
        logger.info(f"MultiTrackTransformer initialized:")
        logger.info(f"  Input features: {input_features}")
        logger.info(f"  Model dimension: {d_model}")
        logger.info(f"  Attention heads: {nhead}")
        logger.info(f"  Encoder layers: {num_encoder_layers}")
        logger.info(f"  Num tracks: {num_tracks}")
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_features)
            mask: Optional padding mask of shape (batch, seq_len)
        
        Returns:
            Dict with predictions for each parameter:
            - 'active': (batch, seq_len, num_tracks, 2) - logits for binary classification
            - 'asset': (batch, seq_len, num_tracks, max_asset_classes) - logits for asset classification
            - 'scale': (batch, seq_len, num_tracks, 1) - regression values
            - 'pos_x': (batch, seq_len, num_tracks, 1)
            - 'pos_y': (batch, seq_len, num_tracks, 1)
            - 'anchor_x': (batch, seq_len, num_tracks, 1)
            - 'anchor_y': (batch, seq_len, num_tracks, 1)
            - 'rotation': (batch, seq_len, num_tracks, 1)
            - 'crop_l': (batch, seq_len, num_tracks, 1)
            - 'crop_r': (batch, seq_len, num_tracks, 1)
            - 'crop_t': (batch, seq_len, num_tracks, 1)
            - 'crop_b': (batch, seq_len, num_tracks, 1)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)
        
        # Create attention mask for transformer
        # mask is True for valid positions, False for padding
        # Transformer expects True for positions to IGNORE
        if mask is not None:
            attn_mask = ~mask  # Invert: True -> False (valid), False -> True (ignore)
        else:
            attn_mask = None
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # (batch, seq_len, d_model)
        
        # Get track embeddings
        track_emb = self.track_embedding(batch_size, seq_len, device)  # (batch, seq_len, num_tracks, d_model)
        
        # Combine encoded features with track embeddings
        # Expand encoded to track dimension
        encoded_expanded = encoded.unsqueeze(2)  # (batch, seq_len, 1, d_model)
        encoded_expanded = encoded_expanded.expand(-1, -1, self.num_tracks, -1)  # (batch, seq_len, num_tracks, d_model)
        
        # Add track embeddings
        track_features = encoded_expanded + track_emb  # (batch, seq_len, num_tracks, d_model)
        
        # Project track features
        track_features = self.track_projection(track_features)  # (batch, seq_len, num_tracks, d_model)
        
        # Apply output heads
        outputs = {
            'active': self.active_head(track_features),  # (batch, seq_len, num_tracks, 2)
            'asset': self.asset_head(track_features),    # (batch, seq_len, num_tracks, max_asset_classes)
            'scale': self.scale_head(track_features),    # (batch, seq_len, num_tracks, 1)
            'pos_x': self.pos_x_head(track_features),
            'pos_y': self.pos_y_head(track_features),
            'anchor_x': self.anchor_x_head(track_features),
            'anchor_y': self.anchor_y_head(track_features),
            'rotation': self.rotation_head(track_features),
            'crop_l': self.crop_l_head(track_features),
            'crop_r': self.crop_r_head(track_features),
            'crop_t': self.crop_t_head(track_features),
            'crop_b': self.crop_b_head(track_features)
        }
        
        return outputs
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



def create_model(
    input_features: int = 180,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    num_tracks: int = 20,
    max_asset_classes: int = 10,
    enable_multimodal: bool = False,
    audio_features: int = 17,
    visual_features: int = 522,
    fusion_type: str = 'gated'
):
    """
    Create a Multi-Track Transformer model (unimodal or multimodal)
    
    Args:
        input_features: Number of input features (default: 180 = 20 tracks × 9 params)
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        num_tracks: Number of video tracks
        max_asset_classes: Maximum number of asset classes
        enable_multimodal: Whether to use multimodal model
        audio_features: Number of audio features (for multimodal)
        visual_features: Number of visual features (for multimodal)
        fusion_type: Fusion strategy ('gated', 'concat', 'add')
    
    Returns:
        MultiTrackTransformer or MultimodalTransformer model
    """
    if enable_multimodal:
        model = MultimodalTransformer(
            audio_features=audio_features,
            visual_features=visual_features,
            track_features=input_features,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_tracks=num_tracks,
            max_asset_classes=max_asset_classes,
            fusion_type=fusion_type
        )
    else:
        model = MultiTrackTransformer(
            input_features=input_features,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_tracks=num_tracks,
            max_asset_classes=max_asset_classes
        )
    
    num_params = model.count_parameters()
    logger.info(f"Model created with {num_params:,} trainable parameters")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    logger.info("Testing Multi-Track Transformer model...")
    
    # Create model
    model = create_model(
        input_features=180,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    # Create dummy input
    batch_size = 4
    seq_len = 100
    input_features = 180
    
    x = torch.randn(batch_size, seq_len, input_features)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Simulate padding in last 20 frames
    mask[:, -20:] = False
    
    logger.info(f"\nInput shape: {x.shape}")
    logger.info(f"Mask shape: {mask.shape}")
    logger.info(f"Valid frames: {mask.sum().item()} / {mask.numel()}")
    
    # Forward pass
    logger.info("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(x, mask)
    
    # Check outputs
    logger.info("\nOutput shapes:")
    for key, value in outputs.items():
        logger.info(f"  {key}: {value.shape}")
    
    # Verify output shapes
    expected_shapes = {
        'active': (batch_size, seq_len, 20, 2),
        'asset': (batch_size, seq_len, 20, 10),
        'scale': (batch_size, seq_len, 20, 1),
        'pos_x': (batch_size, seq_len, 20, 1),
        'pos_y': (batch_size, seq_len, 20, 1),
        'crop_l': (batch_size, seq_len, 20, 1),
        'crop_r': (batch_size, seq_len, 20, 1),
        'crop_t': (batch_size, seq_len, 20, 1),
        'crop_b': (batch_size, seq_len, 20, 1)
    }
    
    all_correct = True
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(outputs[key].shape)
        if actual_shape != expected_shape:
            logger.error(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
            all_correct = False
    
    if all_correct:
        logger.info("\n✅ All output shapes are correct!")
    else:
        logger.error("\n❌ Some output shapes are incorrect!")
    
    # Test with different sequence lengths
    logger.info("\nTesting with different sequence lengths...")
    for test_seq_len in [50, 100, 150]:
        x_test = torch.randn(2, test_seq_len, input_features)
        mask_test = torch.ones(2, test_seq_len, dtype=torch.bool)
        
        with torch.no_grad():
            outputs_test = model(x_test, mask_test)
        
        logger.info(f"  seq_len={test_seq_len}: active output shape = {outputs_test['active'].shape}")
    
    logger.info("\n✅ Model test complete!")



class MultimodalTransformer(nn.Module):
    """
    Multimodal Multi-Track Transformer for video editing prediction
    
    Extends the base MultiTrackTransformer to accept audio, visual, and track inputs,
    fusing them before processing through the transformer encoder.
    
    Predicts 12 parameters for each of 20 tracks:
    - active (binary classification)
    - asset_id (classification)
    - scale, x, y, anchor_x, anchor_y, rotation, crop_l, crop_r, crop_t, crop_b (regression)
    """
    
    def __init__(
        self,
        audio_features: int = 4,
        visual_features: int = 522,
        track_features: int = 180,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_tracks: int = 20,
        max_asset_classes: int = 10,
        enable_multimodal: bool = True,
        fusion_type: str = 'gated'
    ):
        """
        Initialize MultimodalTransformer
        
        Args:
            audio_features: Number of audio features (default: 4)
            visual_features: Number of visual features (default: 522)
            track_features: Number of track features (default: 180)
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_tracks: Number of video tracks
            max_asset_classes: Maximum number of asset classes
            enable_multimodal: Whether to use multimodal features (fallback to track-only if False)
            fusion_type: Fusion strategy ('concat', 'add', 'gated')
        """
        super().__init__()
        
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.track_features = track_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_encoder_layers
        self.num_tracks = num_tracks
        self.max_asset_classes = max_asset_classes
        self.enable_multimodal = enable_multimodal
        self.fusion_type = fusion_type
        
        # Import here to avoid circular dependency
        from src.model.multimodal_modules import ModalityEmbedding, ModalityFusion
        
        # Modality embeddings
        self.audio_embedding = ModalityEmbedding(audio_features, d_model, dropout)
        self.visual_embedding = ModalityEmbedding(visual_features, d_model, dropout)
        self.track_embedding_layer = ModalityEmbedding(track_features, d_model, dropout)
        
        # Modality fusion
        if enable_multimodal:
            self.modality_fusion = ModalityFusion(
                d_model, 
                num_modalities=3, 
                fusion_type=fusion_type,
                dropout=dropout
            )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Track embedding for track-specific predictions
        self.track_embedding = TrackEmbedding(num_tracks, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output heads for each parameter type
        self.track_projection = nn.Linear(d_model, d_model)
        
        # Classification heads
        self.active_head = nn.Linear(d_model, 2)
        self.asset_head = nn.Linear(d_model, max_asset_classes)
        
        # Regression heads
        self.scale_head = nn.Linear(d_model, 1)
        self.pos_x_head = nn.Linear(d_model, 1)
        self.pos_y_head = nn.Linear(d_model, 1)
        self.anchor_x_head = nn.Linear(d_model, 1)
        self.anchor_y_head = nn.Linear(d_model, 1)
        self.rotation_head = nn.Linear(d_model, 1)
        self.crop_l_head = nn.Linear(d_model, 1)
        self.crop_r_head = nn.Linear(d_model, 1)
        self.crop_t_head = nn.Linear(d_model, 1)
        self.crop_b_head = nn.Linear(d_model, 1)
        
        self._init_weights()
        
        logger.info(f"MultimodalTransformer initialized:")
        logger.info(f"  Audio features: {audio_features}")
        logger.info(f"  Visual features: {visual_features}")
        logger.info(f"  Track features: {track_features}")
        logger.info(f"  Model dimension: {d_model}")
        logger.info(f"  Attention heads: {nhead}")
        logger.info(f"  Encoder layers: {num_encoder_layers}")
        logger.info(f"  Enable multimodal: {enable_multimodal}")
        logger.info(f"  Fusion type: {fusion_type}")
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        track: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            audio: Audio features (batch, seq_len, audio_features)
            visual: Visual features (batch, seq_len, visual_features)
            track: Track features (batch, seq_len, track_features)
            padding_mask: Padding mask (batch, seq_len) - True for valid, False for padding
            modality_mask: Modality availability mask (batch, seq_len, 3) - [audio, visual, track]
        
        Returns:
            Dict with predictions for each parameter
        """
        batch_size, seq_len, _ = track.shape
        device = track.device
        
        if self.enable_multimodal:
            # Embed each modality
            audio_emb = self.audio_embedding(audio)  # (batch, seq_len, d_model)
            visual_emb = self.visual_embedding(visual)  # (batch, seq_len, d_model)
            track_emb = self.track_embedding_layer(track)  # (batch, seq_len, d_model)
            
            # Fuse modalities
            fused = self.modality_fusion(
                audio_emb, visual_emb, track_emb, modality_mask
            )  # (batch, seq_len, d_model)
        else:
            # Track-only mode
            fused = self.track_embedding_layer(track)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(fused)  # (batch, seq_len, d_model)
        
        # Create attention mask for transformer
        # padding_mask is True for valid positions, False for padding
        # Transformer expects True for positions to IGNORE
        if padding_mask is not None:
            attn_mask = ~padding_mask  # Invert
        else:
            attn_mask = None
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # (batch, seq_len, d_model)
        
        # Get track embeddings for track-specific predictions
        track_emb = self.track_embedding(batch_size, seq_len, device)  # (batch, seq_len, num_tracks, d_model)
        
        # Combine encoded features with track embeddings
        encoded_expanded = encoded.unsqueeze(2)  # (batch, seq_len, 1, d_model)
        encoded_expanded = encoded_expanded.expand(-1, -1, self.num_tracks, -1)  # (batch, seq_len, num_tracks, d_model)
        
        # Add track embeddings
        track_features = encoded_expanded + track_emb  # (batch, seq_len, num_tracks, d_model)
        
        # Project track features
        track_features = self.track_projection(track_features)  # (batch, seq_len, num_tracks, d_model)
        
        # Apply output heads
        outputs = {
            'active': self.active_head(track_features),
            'asset': self.asset_head(track_features),
            'scale': self.scale_head(track_features),
            'pos_x': self.pos_x_head(track_features),
            'pos_y': self.pos_y_head(track_features),
            'anchor_x': self.anchor_x_head(track_features),
            'anchor_y': self.anchor_y_head(track_features),
            'rotation': self.rotation_head(track_features),
            'crop_l': self.crop_l_head(track_features),
            'crop_r': self.crop_r_head(track_features),
            'crop_t': self.crop_t_head(track_features),
            'crop_b': self.crop_b_head(track_features)
        }
        
        return outputs
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_multimodal_model(
    audio_features: int = 4,
    visual_features: int = 522,
    track_features: int = 180,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    num_tracks: int = 20,
    max_asset_classes: int = 10,
    enable_multimodal: bool = True,
    fusion_type: str = 'gated'
) -> MultimodalTransformer:
    """
    Create a Multimodal Multi-Track Transformer model
    
    Args:
        audio_features: Number of audio features
        visual_features: Number of visual features
        track_features: Number of track features
        d_model: Model dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        num_tracks: Number of video tracks
        max_asset_classes: Maximum number of asset classes
        enable_multimodal: Whether to enable multimodal features
        fusion_type: Fusion strategy ('concat', 'add', 'gated')
    
    Returns:
        MultimodalTransformer model
    """
    model = MultimodalTransformer(
        audio_features=audio_features,
        visual_features=visual_features,
        track_features=track_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_tracks=num_tracks,
        max_asset_classes=max_asset_classes,
        enable_multimodal=enable_multimodal,
        fusion_type=fusion_type
    )
    
    num_params = model.count_parameters()
    logger.info(f"Multimodal model created with {num_params:,} trainable parameters")
    
    return model
