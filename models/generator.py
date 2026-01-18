import torch
import torch.nn as nn

from models.encoder import TransformerEncoder


class Generator(nn.Module):
    """
    Generator untuk Replaced Token Detection (RTD).

    Generator bertugas memprediksi token yang dimasking
    (seperti Masked Language Model / MLM).
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        head_dim: int,
        d_ffn: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Encoder Transformer 
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            d_ffn=d_ffn,
            dropout=dropout,
        )

        # Output head ke vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward Generator.

        Input
        -----
        input_embeddings : torch.Tensor
            (batch_size, seq_len, d_model)

        attention_mask : torch.Tensor, opsional
            Mask padding token

        Output
        ------
        logits : torch.Tensor
            (batch_size, seq_len, vocab_size)
        """

        hidden_states = self.encoder(input_embeddings, attention_mask)
        logits = self.lm_head(hidden_states)

        return logits
