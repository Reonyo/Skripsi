import torch
import torch.nn.functional as F
from typing import Tuple

def create_mlm_inputs(
    input_ids: torch.Tensor,
    mask_token_id: int,
    special_token_ids: list[int],
    mlm_probability: float = 0.15,
):
    """
    Masked Language Modeling (MLM) dengan pengecualian token spesial.
    - Memilih ~15% token untuk dimask (dari probability_matrix berisi 0.15).
    - Token yang dipilih diganti menjadi [MASK] (mask_token_id).
    - Token spesial (special_token_ids, mis. CLS/SEP/PAD) tidak pernah dimask.
    - Label MLM:
        * token yang dimask: label = token asli
        * token yang tidak dimask: label = -100 (diabaikan oleh loss)
    Parameter
    ---------
    input_ids : torch.Tensor
        Bentuk (batch_size, seq_len), berisi token input.
    mask_token_id : int
        ID token [MASK] untuk mengganti token yang dipilih.
    special_token_ids : list[int]
        Daftar ID token yang tidak boleh dimask (contoh: CLS/SEP/PAD).
    mlm_probability : float, default=0.15
        Probabilitas awal untuk memilih token agar dimask.
    Return
    ----
    masked_input_ids : torch.Tensor
        Input yang sudah diganti [MASK] pada posisi terpilih.
    mlm_labels : torch.Tensor
        Label untuk loss MLM; posisi tak terpilih diisi -100.
    """

    device = input_ids.device  # pastikan tensor baru dibuat di device yang sama (CPU/GPU)

    masked_input_ids = input_ids.clone()  # salin input untuk dimodifikasi jadi input bertopeng
    mlm_labels = input_ids.clone()        # salin input untuk label asli (hanya dipakai di posisi masked)

    # Probabilitas awal: tensor berisi nilai konstan mlm_probability untuk tiap posisi
    probability_matrix = torch.full(
        input_ids.shape, mlm_probability, device=device
    )

    # Jangan mask token spesial: set probabilitasnya 0 agar pasti tidak terpilih
    for token_id in special_token_ids:
        probability_matrix[input_ids == token_id] = 0.0

    # Sampling mask: bernoulli menghasilkan 1/0 (True/False) sesuai probabilitas per posisi
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Label hanya untuk token masked; lainnya -100 agar diabaikan oleh loss
    # 
    # masked_indices: tensor boolean dengan True di posisi yang akan dimask, False di posisi lain
    # Contoh: [True, False, True, False, False]
    # 
    # ~masked_indices: operator NOT (negasi) untuk membalik nilai boolean
    # Contoh: [False, True, False, True, True]
    # 
    # mlm_labels[~masked_indices] = -100
    # Artinya: pada posisi yang TIDAK dimask (bernilai False), set label menjadi -100
    # PyTorch's CrossEntropyLoss mengabaikan target bernilai -100 (default ignore_index=-100)
    # Jadi loss hanya dihitung untuk token yang dimask (label = token asli)
    # Token yang tidak dimask tidak berkontribusi pada loss
    mlm_labels[~masked_indices] = -100

    # Ganti token terpilih dengan [MASK]
    masked_input_ids[masked_indices] = mask_token_id

    return masked_input_ids, mlm_labels

def replace_with_generator(
    input_ids: torch.Tensor,
    masked_input_ids: torch.Tensor,
    mlm_labels: torch.Tensor,
    generator_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replaced Token Detection (RTD) - mengganti token MLM dengan prediksi generator
    dan membuat label untuk melatih discriminator.

    Alur:
    1. Generator memprediksi token di posisi [MASK]
    2. Ganti HANYA token yang dimask dengan prediksi generator (token asli tetap)
    3. Buat label RTD: 1 = token asli, 0 = token hasil generator (salah)

    Parameter
    ---------
    input_ids : torch.Tensor
        (batch_size, seq_len) - Token asli sebelum masking
    masked_input_ids : torch.Tensor
        (batch_size, seq_len) - Token dengan [MASK] di posisi tertentu (output dari create_mlm_inputs)
    mlm_labels : torch.Tensor
        (batch_size, seq_len) - Label MLM; -100 untuk posisi non-masked (dari create_mlm_inputs)
    generator_logits : torch.Tensor
        (batch_size, seq_len, vocab_size) - Output logits dari Generator model

    Return
    ----
    replaced_input_ids : torch.Tensor
        (batch_size, seq_len) - Input untuk Discriminator (token asli + prediksi generator di posisi masked)
    rtd_labels : torch.Tensor
        (batch_size, seq_len) - Label RTD untuk training:
            * 1: token asli (tidak dimask atau prediksi generator benar)
            * 0: token hasil generator yang salah (berbeda dengan asli)
    """

    # Prediksi generator: ambil token dengan probabilitas tertinggi per posisi
    generator_predictions = torch.argmax(generator_logits, dim=-1)
    # Shape: (batch_size, seq_len)

    # Token hasil replace: mulai dari input asli
    replaced_input_ids = input_ids.clone()

    # Posisi token yang dimask: mlm_labels != -100 berarti posisi ini dimask
    # -100 adalah marker untuk token yang tidak dimask (diabaikan loss)
    masked_positions = mlm_labels != -100

    # Ganti HANYA token di posisi masked dengan prediksi generator
    # Token di posisi non-masked tetap input asli
    replaced_input_ids[masked_positions] = generator_predictions[masked_positions]

    # RTD label: untuk setiap token, apakah sama dengan input asli?
    # Inisialisasi: semua token dianggap 1 (token asli)
    rtd_labels = torch.ones_like(input_ids)
    
    # Di posisi yang dimask:
    # - Jika prediksi == asli (generator benar): label = 1
    # - Jika prediksi != asli (generator salah): label = 0
    rtd_labels[masked_positions] = (
        replaced_input_ids[masked_positions] == input_ids[masked_positions]
    ).long()

    return replaced_input_ids, rtd_labels

