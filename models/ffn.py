import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network (FFN) pada Transformer Encoder
    dengan arsitektur SwiGLU.
    """

    def __init__(self, d_model: int, d_ffn: int):
        """
        Parameter
        ---------
        d_model : int
            Dimensi embedding input.
        d_ffn : int
            Dimensi hidden FFN.
        """
        super().__init__()

        # Linear Value dan Gate digabung dalam satu layer (MENGHEMAT KOMPUTASI)
        self.fc_in = nn.Linear(d_model, 2 * d_ffn, bias=False)
        # Linear output
        self.fc_out = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass FFN.

        Input
        -----
        x : torch.Tensor
            (batch_size, seq_len, d_model)

        Output
        ------
        torch.Tensor
            (batch_size, seq_len, d_model)
        """

        # Proyeksi linear awal
        x_proj = self.fc_in(x)
        # shape: (batch_size, seq_len, 2 * d_ffn)

        # Pisahkan menjadi value dan gate menggunakan chunk
        # 
        # chunk(chunks, dim): membagi tensor menjadi beberapa bagian yang sama besar
        # - chunks=2: bagi menjadi 2 bagian
        # - dim=-1: bagi pada dimensi terakhir (dimensi ke-2 dalam kasus ini)
        # Hasil: dua tensor terpisah, masing-masing (batch_size, seq_len, d_ffn)
        value, gate = x_proj.chunk(2, dim=-1)
        # masing-masing: (batch_size, seq_len, d_ffn)

        # Aktivasi SwiGLU
        hidden = F.silu(value) * gate

        # Proyeksi kembali ke d_model
        output = self.fc_out(hidden)

        return output
