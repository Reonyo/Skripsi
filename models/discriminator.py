import torch
import torch.nn as nn

from models.encoder import TransformerEncoder


class Discriminator(nn.Module):
    """
    Discriminator untuk Replaced Token Detection (RTD).

    Discriminator mendeteksi apakah setiap token
    adalah token asli (1) atau hasil generator (0).
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

        # Encoder Transformer (model utama)
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            d_ffn=d_ffn,
            dropout=dropout,
        )

        # Binary classification head per token
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward Discriminator.

        Input
        -----
        input_embeddings : torch.Tensor
            (batch_size, seq_len, d_model)

        attention_mask : torch.Tensor, opsional
            Mask padding token

        Output
        ------
        logits : torch.Tensor
            (batch_size, seq_len)
            Logit untuk klasifikasi token asli / palsu
        """

        hidden_states = self.encoder(input_embeddings, attention_mask)
        logits = self.classifier(hidden_states).squeeze(-1)

        return logits
