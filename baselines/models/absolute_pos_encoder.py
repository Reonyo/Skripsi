"""
Encoder dengan Absolute Positional Encoding (baseline comparison).
Menggunakan standard sinusoidal atau learned positional embeddings.
"""
import torch
import torch.nn as nn
from models.encoder import TransformerEncoderLayer


class AbsolutePositionalEncoding(nn.Module):
    """Sinusoidal Absolute Positional Encoding (seperti Transformer original)."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            (batch_size, seq_len, d_model) with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class AbsolutePosEncoder(nn.Module):
    """
    Transformer Encoder stack dengan Absolute Positional Encoding.
    
    Baseline untuk comparison dengan RoPE-based encoder.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        head_dim: int,
        d_ffn: int,
        dropout: float = 0.0,
        max_len: int = 512,
    ):
        super().__init__()
        
        # Absolute positional encoding
        self.pos_encoding = AbsolutePositionalEncoding(d_model, max_len)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                head_dim=head_dim,
                d_ffn=d_ffn,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through encoder stack.
        
        Args:
            x: Embeddings (batch_size, seq_len, d_model)
            attention_mask: Padding mask (batch_size, seq_len)
        
        Returns:
            Encoded representations (batch_size, seq_len, d_model)
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final normalization
        x = self.norm(x)
        
        return x
