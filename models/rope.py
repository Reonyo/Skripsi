import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)

    Parameter
    ---------
    dim : int
        Dimensi per attention head (head_dim).
        HARUS genap.
    base : int
        Basis frekuensi sinusoidal (default: 10000).

    Input
    -----
    q, k : Tensor
        Bentuk: (batch_size, num_heads, seq_len, head_dim)

    Output
    ------
    q_rot, k_rot : Tensor
        Query dan Key yang sudah diberi encoding posisi.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("head_dim harus genap untuk RoPE")

        self.dim = dim
        self.base = base

        # Menghitung inverse frequency (Omega_i) untuk setiap pasangan dimensi
        # Omega_i = 1 / (base ** (2i / dim))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        # Disimpan sebagai buffer
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        Menerapkan RoPE ke Query dan Key.

        q, k shape:
        (batch, heads, seq_len, head_dim)
        """
        seq_len = q.size(-2)
        device = q.device
        dtype = q.dtype

        # Posisi token: 0, 1, 2, ..., seq_len-1
        positions = torch.arange(seq_len, device=device)

        # Menghitung sudut rotasi
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)

        sin = freqs.sin().to(dtype)
        cos = freqs.cos().to(dtype)

        q = self._apply_rotary(q, sin, cos)
        k = self._apply_rotary(k, sin, cos)

        return q, k

    def _apply_rotary(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        """
        Menerapkan rotasi pada tensor.

        x shape:
        (batch, heads, seq_len, head_dim)
        """
        # Pisahkan dimensi genap dan ganjil
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Broadcast sin & cos
        # Nilai sinus dan cosinus awalnya memiliki bentuk (seq_len, head_dim/2),
        # yaitu hanya bergantung pada posisi token dan dimensi embedding.
        # 
        # Dengan menambahkan dua dimensi menggunakan unsqueeze, tensor sin dan cos
        # diubah menjadi bentuk (1, 1, seq_len, head_dim/2) sehingga dapat
        # dibroadcast secara implisit ke seluruh batch dan attention head.

        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        # Rumus rotasi
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Gabungkan kembali
        x_rot = torch.stack(
            (x_rot_even, x_rot_odd),
            dim=-1
        ).flatten(-2)

        return x_rot
