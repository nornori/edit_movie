"""
Enhanced Cut Selection Model V2 with Deeper Architecture and Improved Attention

Improvements over V1:
- Deeper encoder (8 layers instead of 6)
- Multi-head self-attention with more heads (16 instead of 8)
- Residual connections in fusion
- Layer normalization improvements
- Deeper classification head
"""
import torch
import torch.nn as nn
from src.model.multimodal_modules import ModalityEmbedding
from src.cut_selection.positional_encoding import PositionalEncoding


class ImprovedThreeModalityFusion(nn.Module):
    """
    Improved fusion module with residual connections and layer normalization
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Gating mechanism for each modality (deeper)
        self.audio_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.visual_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Fusion layers (deeper with residual)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual projection (for skip connection)
        self.residual_proj = nn.Linear(d_model * 3, d_model)
    
    def forward(self, audio: torch.Tensor, visual: torch.Tensor, temporal: torch.Tensor):
        """
        Fuse three modalities with residual connection
        
        Args:
            audio: (batch, seq_len, d_model)
            visual: (batch, seq_len, d_model)
            temporal: (batch, seq_len, d_model)
        
        Returns:
            fused: (batch, seq_len, d_model)
        """
        # Apply gating
        audio_gated = audio * self.audio_gate(audio)
        visual_gated = visual * self.visual_gate(visual)
        temporal_gated = temporal * self.temporal_gate(temporal)
        
        # Concatenate
        concat = torch.cat([audio_gated, visual_gated, temporal_gated], dim=-1)
        
        # Fusion with residual connection
        fused = self.fusion(concat)
        residual = self.residual_proj(concat)
        fused = fused + residual  # Residual connection
        
        return fused


class EnhancedCutSelectionModelV2(nn.Module):
    """
    Enhanced V2 model with deeper architecture and improved attention
    
    Key improvements:
    - 8 encoder layers (vs 6 in V1)
    - 16 attention heads (vs 8 in V1)
    - Improved fusion with residual connections
    - Deeper classification head (3 layers vs 2)
    - Better regularization
    """
    
    def __init__(
        self,
        audio_features: int,
        visual_features: int,
        temporal_features: int,
        d_model: int = 256,
        nhead: int = 16,  # Increased from 8
        num_encoder_layers: int = 8,  # Increased from 6
        dim_feedforward: int = 1024,
        dropout: float = 0.15  # Slightly increased for better regularization
    ):
        super().__init__()
        
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.temporal_features = temporal_features
        self.d_model = d_model
        
        # Modality embeddings (with layer norm)
        self.audio_embedding = nn.Sequential(
            ModalityEmbedding(audio_features, d_model, dropout),
            nn.LayerNorm(d_model)
        )
        self.visual_embedding = nn.Sequential(
            ModalityEmbedding(visual_features, d_model, dropout),
            nn.LayerNorm(d_model)
        )
        self.temporal_embedding = nn.Sequential(
            ModalityEmbedding(temporal_features, d_model, dropout),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        # Improved fusion
        self.fusion = ImprovedThreeModalityFusion(d_model, dropout=dropout)
        
        # Deeper Transformer encoder with more heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Deeper active prediction head (3 layers instead of 2)
        self.active_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 2 classes: inactive (0) or active (1)
        )
        
        print(f"EnhancedCutSelectionModelV2 initialized:")
        print(f"  Audio features: {audio_features}")
        print(f"  Visual features: {visual_features}")
        print(f"  Temporal features: {temporal_features}")
        print(f"  Total input features: {audio_features + visual_features + temporal_features}")
        print(f"  Model dimension: {d_model}")
        print(f"  Attention heads: {nhead} (↑ from 8)")
        print(f"  Encoder layers: {num_encoder_layers} (↑ from 6)")
        print(f"  Feedforward dim: {dim_feedforward}")
        print(f"  Dropout: {dropout}")
        print(f"  Improvements: Deeper network, more attention heads, residual fusion, pre-norm")
    
    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        temporal: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        """
        Forward pass
        
        Args:
            audio: (batch, seq_len, audio_features)
            visual: (batch, seq_len, visual_features)
            temporal: (batch, seq_len, temporal_features)
            padding_mask: (batch, seq_len) - True for padding positions
        
        Returns:
            dict with:
                - active: (batch, seq_len, 2) - logits for binary classification
        """
        # Embed modalities (with layer norm)
        audio_emb = self.audio_embedding(audio)  # (batch, seq_len, d_model)
        visual_emb = self.visual_embedding(visual)  # (batch, seq_len, d_model)
        temporal_emb = self.temporal_embedding(temporal)  # (batch, seq_len, d_model)
        
        # Fuse modalities (with residual)
        fused = self.fusion(audio_emb, visual_emb, temporal_emb)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        fused = self.positional_encoding(fused)  # (batch, seq_len, d_model)
        
        # Deeper transformer encoding (8 layers, 16 heads)
        encoded = self.transformer(fused, src_key_padding_mask=padding_mask)  # (batch, seq_len, d_model)
        
        # Deeper active prediction (3 layers)
        active_logits = self.active_head(encoded)  # (batch, seq_len, 2)
        
        return {
            'active': active_logits
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing EnhancedCutSelectionModelV2...")
    
    batch_size = 4
    seq_len = 1000
    audio_dim = 235
    visual_dim = 543
    temporal_dim = 6
    
    # Create model
    model = EnhancedCutSelectionModelV2(
        audio_features=audio_dim,
        visual_features=visual_dim,
        temporal_features=temporal_dim,
        d_model=256,
        nhead=16,
        num_encoder_layers=8,
        dim_feedforward=1024,
        dropout=0.15
    )
    
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Create dummy inputs
    audio = torch.randn(batch_size, seq_len, audio_dim)
    visual = torch.randn(batch_size, seq_len, visual_dim)
    temporal = torch.randn(batch_size, seq_len, temporal_dim)
    
    # Forward pass
    outputs = model(audio, visual, temporal)
    
    print(f"\nOutput shape: {outputs['active'].shape}")
    assert outputs['active'].shape == (batch_size, seq_len, 2)
    
    print("✅ EnhancedCutSelectionModelV2 test passed!")
