import math
import torch
import torch.nn as nn
from models.rope import RotaryEmbedding
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) dengan RoPE.

    Parameter
    ---------
    d_model : int
        Dimensi embedding model.
    num_heads : int
        Jumlah attention head.
    head_dim : int
        Dimensi per attention head.
        Harus memenuhi: d_model = num_heads × head_dim.
    dropout : float
        Dropout probability pada attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert d_model == num_heads * head_dim, (
            "d_model harus sama dengan num_heads × head_dim"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Linear projection untuk Query, Key, dan Value
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Rotary Positional Embedding (hanya untuk Q dan K)
        self.rope = RotaryEmbedding(head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass Multi-Head Self-Attention.

        Parameter
        ---------
        x : torch.Tensor
            Input tensor dengan bentuk:
            (batch_size, seq_len, d_model)

        attention_mask : torch.Tensor, opsional
            Mask untuk padding token dengan bentuk:
            (batch_size, seq_len),
            di mana nilai 0 menandakan token padding.

        Return
        ------
        out : torch.Tensor
            Output tensor dengan bentuk:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()

        # Proyeksi linear
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Memecah embedding d_model menjadi beberapa attention head
        # Dari: (batch_size, seq_len, d_model)
        # Menjadi: (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Menukar urutan dimensi agar head berada di depan
        # Dari: (batch_size, seq_len, num_heads, head_dim)
        # Menjadi: (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Terapkan RoPE pada Query dan Key
        q, k = self.rope(q, k)

        # Hitung attention scores menggunakan scaled dot-product attention
        # Langkah 1: Kalikan Q dengan K^T untuk mendapatkan skor kemiripan
        # 
        # torch.matmul pada tensor 4D melakukan batched matrix multiplication:
        # - Hanya operasi pada 2 dimensi terakhir (matrix multiplication)
        # - Dimensi awal (batch_size, num_heads) di-broadcast secara otomatis
        #
        # k.transpose(-2, -1) menukar dua dimensi terakhir:
        # dari (batch_size, num_heads, seq_len, head_dim)
        # ke   (batch_size, num_heads, head_dim, seq_len)
        # 
        # Hasil matmul untuk setiap batch dan head:
        # (seq_len, head_dim) @ (head_dim, seq_len) → (seq_len, seq_len)
    
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        # Hasil akhir: (batch_size, num_heads, seq_len, seq_len)
        
        # Terapkan attention mask
        # Attention mask digunakan untuk mengabaikan token padding dalam perhitungan attention
        # Token padding (biasanya <PAD>) tidak boleh mempengaruhi representasi token lain
        if attention_mask is not None:
            # Input attention_mask: (batch_size, seq_len)
            # Contoh: [1, 1, 1, 0, 0] → token ke-4 dan ke-5 adalah padding
            # 
            # unsqueeze(1): tambah dimensi di posisi 1
            # (batch_size, seq_len) → (batch_size, 1, seq_len)
            # 
            # unsqueeze(2): tambah dimensi di posisi 2
            # (batch_size, 1, seq_len) → (batch_size, 1, 1, seq_len)
            # 
            # Bentuk akhir cocok untuk di-broadcast dengan scores: (batch_size, num_heads, seq_len, seq_len)
            # Dimensi '1' akan otomatis diulang untuk semua heads dan query positions
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # masked_fill: ganti nilai di posisi mask == 0 dengan -inf
            # -inf akan menjadi ~0 setelah softmax, sehingga padding tidak berkontribusi
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax untuk mendapatkan attention weights (normalisasi menjadi distribusi probabilitas)
        # dim=-1 berarti softmax diterapkan pada dimensi terakhir
        # 
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        # dim=-1 adalah dimensi ke-3 (dimensi terakhir): seq_len kedua (Key positions)
        # 
        # Untuk setiap query position, kita normalisasi attention scores terhadap semua key positions
        # Hasil: setiap baris (query) memiliki total probabilitas = 1.0
        # Shape tetap: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Dropout: secara acak set beberapa attention weights menjadi 0
        attn_weights = self.dropout(attn_weights)

        # Weighted sum dengan Value
        # attn_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        # v: (batch_size, num_heads, seq_len_k, head_dim)
        # 
        # Untuk setiap query position, kita kalikan attention weights dengan value vectors
        # Hasil: (batch_size, num_heads, seq_len_q, head_dim)
        # di mana seq_len_q == seq_len_k (self-attention)
        out = torch.matmul(attn_weights, v)

        # Gabungkan kembali semua attention heads
        # 
        # transpose(1, 2): tukar dimensi num_heads dengan seq_len
        # dari: (batch_size, num_heads, seq_len, head_dim)
        # ke:   (batch_size, seq_len, num_heads, head_dim)
        # 
        # contiguous(): pastikan tensor disimpan secara berurutan di memory
        out = out.transpose(1, 2).contiguous()
        
        # view(): reshape tensor (gabungkan num_heads dan head_dim)
        # dari: (batch_size, seq_len, num_heads, head_dim)
        # ke:   (batch_size, seq_len, num_heads × head_dim)
        #     = (batch_size, seq_len, d_model)
        # 
        # Mengembalikan semua heads menjadi satu embedding tunggal per token
        out = out.view(batch_size, seq_len, self.d_model)

        # Proyeksi akhir: transformasi linear untuk menggabungkan informasi dari semua heads
        # Input:  (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, d_model)
        out = self.out_proj(out)

        return out
