"""
Positional Encoding for Temporal Sequences

Adds positional information to embeddings so the model can understand temporal order.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences
    
    Uses sine and cosine functions of different frequencies to encode position.
    This allows the model to understand the temporal order of frames.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize PositionalEncoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added (batch, seq_len, d_model)
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding
    
    Instead of fixed sinusoidal patterns, this learns the best positional
    encoding during training.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize LearnablePositionalEncoding
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learnable positional encoding to input
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added (batch, seq_len, d_model)
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
