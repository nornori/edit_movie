"""
Enhanced Cut Selection Model with Temporal Features

Audio + Visual + Temporal → Active prediction (binary: used/not used in edit)
"""
import torch
import torch.nn as nn
from src.model.multimodal_modules import ModalityEmbedding
from src.cut_selection.positional_encoding import PositionalEncoding


class ThreeModalityFusion(nn.Module):
    """
    Fusion module for 3 modalities: audio, visual, temporal
    
    Uses gated fusion to combine modalities with learned importance weights
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Gating mechanism for each modality
        self.audio_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.visual_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, audio: torch.Tensor, visual: torch.Tensor, temporal: torch.Tensor):
        """
        Fuse three modalities
        
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
        
        # Concatenate and fuse
        concat = torch.cat([audio_gated, visual_gated, temporal_gated], dim=-1)
        fused = self.fusion(concat)
        
        return fused


class EnhancedCutSelectionModel(nn.Module):
    """
    Enhanced model for cut selection with temporal features
    
    Input: Audio + Visual + Temporal features from source video
    Output: Binary active prediction (0=not used, 1=used in edit)
    """
    
    def __init__(
        self,
        audio_features: int,
        visual_features: int,
        temporal_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.temporal_features = temporal_features
        self.d_model = d_model
        
        # Modality embeddings
        self.audio_embedding = ModalityEmbedding(audio_features, d_model, dropout)
        self.visual_embedding = ModalityEmbedding(visual_features, d_model, dropout)
        self.temporal_embedding = ModalityEmbedding(temporal_features, d_model, dropout)
        
        # Positional encoding (essential for temporal understanding)
        self.positional_encoding = PositionalEncoding(d_model, max_len=5000, dropout=dropout)
        
        # Fusion (3 modalities: audio + visual + temporal)
        self.fusion = ThreeModalityFusion(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Active prediction head (binary classification)
        self.active_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 2 classes: inactive (0) or active (1)
        )
        
        print(f"EnhancedCutSelectionModel initialized:")
        print(f"  Audio features: {audio_features}")
        print(f"  Visual features: {visual_features}")
        print(f"  Temporal features: {temporal_features}")
        print(f"  Total input features: {audio_features + visual_features + temporal_features}")
        print(f"  Model dimension: {d_model}")
        print(f"  Attention heads: {nhead}")
        print(f"  Encoder layers: {num_encoder_layers}")
    
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
        # Embed modalities
        audio_emb = self.audio_embedding(audio)  # (batch, seq_len, d_model)
        visual_emb = self.visual_embedding(visual)  # (batch, seq_len, d_model)
        temporal_emb = self.temporal_embedding(temporal)  # (batch, seq_len, d_model)
        
        # Fuse modalities
        fused = self.fusion(audio_emb, visual_emb, temporal_emb)  # (batch, seq_len, d_model)
        
        # Add positional encoding (critical for temporal understanding)
        fused = self.positional_encoding(fused)  # (batch, seq_len, d_model)
        
        # Transformer encoding
        encoded = self.transformer(fused, src_key_padding_mask=padding_mask)  # (batch, seq_len, d_model)
        
        # Active prediction
        active_logits = self.active_head(encoded)  # (batch, seq_len, 2)
        
        return {
            'active': active_logits
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
