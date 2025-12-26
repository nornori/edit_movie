"""
Two-Modality Fusion Module for Cut Selection

Specialized fusion for audio + visual features only.
"""
import torch
import torch.nn as nn
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class TwoModalityFusion(nn.Module):
    """
    Fusion module for combining two modality embeddings (audio + visual)
    
    Supports multiple fusion strategies:
    - 'concat': Concatenate embeddings and project to d_model
    - 'add': Weighted addition of embeddings
    - 'gated': Gated fusion with learned gates (recommended)
    """
    
    def __init__(
        self,
        d_model: int,
        fusion_type: Literal['concat', 'add', 'gated'] = 'gated',
        dropout: float = 0.1
    ):
        """
        Initialize TwoModalityFusion
        
        Args:
            d_model: Model dimension
            fusion_type: Fusion strategy ('concat', 'add', 'gated')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # Concatenation fusion: [audio; visual] -> Linear(2*d_model, d_model)
            self.fusion_projection = nn.Linear(2 * d_model, d_model)
            nn.init.xavier_uniform_(self.fusion_projection.weight)
            nn.init.zeros_(self.fusion_projection.bias)
            
        elif fusion_type == 'add':
            # Additive fusion with learned weights
            self.modality_weights = nn.Parameter(torch.ones(2))
            
        elif fusion_type == 'gated':
            # Gated fusion: gate = sigmoid(W * embedding + b)
            self.gate_audio = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            self.gate_visual = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            )
            
            # Initialize gate networks
            nn.init.xavier_uniform_(self.gate_audio[0].weight)
            nn.init.zeros_(self.gate_audio[0].bias)
            nn.init.xavier_uniform_(self.gate_visual[0].weight)
            nn.init.zeros_(self.gate_visual[0].bias)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Must be 'concat', 'add', or 'gated'")
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"TwoModalityFusion initialized: type={fusion_type}")
    
    def forward(
        self,
        audio_emb: torch.Tensor,
        visual_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse audio and visual embeddings
        
        Args:
            audio_emb: Audio embeddings (batch, seq_len, d_model)
            visual_emb: Visual embeddings (batch, seq_len, d_model)
        
        Returns:
            Fused embeddings (batch, seq_len, d_model)
        """
        if self.fusion_type == 'concat':
            # Concatenate along feature dimension
            concatenated = torch.cat([audio_emb, visual_emb], dim=-1)  # (batch, seq_len, 2*d_model)
            
            # Project back to d_model
            fused = self.fusion_projection(concatenated)  # (batch, seq_len, d_model)
            
        elif self.fusion_type == 'add':
            # Weighted addition
            # Normalize weights to sum to 1
            weights = torch.softmax(self.modality_weights, dim=0)
            
            fused = weights[0] * audio_emb + weights[1] * visual_emb
            
        elif self.fusion_type == 'gated':
            # Gated fusion: gate_i = sigmoid(W_i * emb_i + b_i)
            gate_a = self.gate_audio(audio_emb)  # (batch, seq_len, d_model)
            gate_v = self.gate_visual(visual_emb)  # (batch, seq_len, d_model)
            
            # Weighted sum: fused = gate_a ⊙ audio + gate_v ⊙ visual
            fused = gate_a * audio_emb + gate_v * visual_emb
        
        # Apply dropout
        fused = self.dropout(fused)
        
        return fused
