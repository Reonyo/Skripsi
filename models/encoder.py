import torch
import torch.nn as nn

from models.attention import MultiHeadSelfAttention
from models.ffn import FeedForwardNetwork


class TransformerEncoderLayer(nn.Module):
    """
    Satu layer Transformer Encoder dengan arsitektur Pre-LN.

    Struktur:
    - LayerNorm → Multi-Head Self-Attention → Residual
    - LayerNorm → Feed-Forward Network (SwiGLU) → Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        d_ffn: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Layer Normalization (Pre-LN)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

        # Feed Forward Network (SwiGLU)
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ffn=d_ffn,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass satu encoder layer.

        Input
        -----
        x : torch.Tensor
            (batch_size, seq_len, d_model)

        attention_mask : torch.Tensor, opsional
            Mask padding token (batch_size, seq_len)

        Output
        ------
        torch.Tensor
            (batch_size, seq_len, d_model)
        """

        # ===== Self-Attention Block =====
        residual = x
        x = self.norm_attn(x)
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = x + residual

        # ===== Feed Forward Network Block =====
        residual = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder yang terdiri dari beberapa layer encoder.
    Digunakan sebagai backbone Generator dan Discriminator.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        head_dim: int,
        d_ffn: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    d_ffn=d_ffn,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )


    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass Transformer Encoder.

        Input
        -----
        x : torch.Tensor
            (batch_size, seq_len, d_model)

        attention_mask : torch.Tensor, opsional
            Mask padding token

        Output
        ------
        torch.Tensor
            (batch_size, seq_len, d_model)
        """

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x
