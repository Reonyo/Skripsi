import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers.optimization import get_cosine_schedule_with_warmup
from training.masking import (
    create_mlm_inputs,
    replace_with_generator,
)

def pretrain(
    generator,
    discriminator,
    embedding_layer,
    dataloader: DataLoader,
    vocab_size: int,
    mask_token_id: int,
    special_token_ids: list[int],
    device: torch.device,
    *,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    rtd_loss_weight: float = 1.0,
):
    """
    Pretraining loop dengan objective MLM (Generator) + RTD (Discriminator).

    Parameter utama
    ---------------
    generator : nn.Module
        Model generator (MLM)

    discriminator : nn.Module
        Model discriminator (RTD)

    embedding_layer : nn.Embedding
        Embedding token (shared oleh generator & discriminator)

    dataloader : DataLoader
        DataLoader berisi batch input_ids & attention_mask

    vocab_size : int
        Ukuran vocabulary

    mask_token_id : int
        Token ID untuk [MASK]

    special_token_ids : list[int]
        Token spesial yang tidak boleh dimask

    device : torch.device
        CPU / CUDA

    num_epochs : int
        Jumlah epoch pretraining

    learning_rate : float
        Learning rate AdamW

    weight_decay : float
        Weight decay AdamW

    warmup_steps : int
        Jumlah warmup step scheduler

    rtd_loss_weight : float
        Bobot loss RTD (λ)
    """

    generator.to(device)
    discriminator.to(device)
    embedding_layer.to(device)
    # .to(device): pindahkan semua parameter & buffer model ke device (CPU/GPU)
    # Penting: data dan model harus di device yang sama untuk komputasi

    generator.train()
    discriminator.train()
    # .train(): set model ke training mode
    # - Aktifkan Dropout & BatchNorm (behavior berbeda dari eval mode)
    # - Di training: Dropout random drop neurons, BatchNorm update statistics
    # - Di eval: Dropout tidak aktif, BatchNorm pakai running statistics

    # ===== Loss Functions =====
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # CrossEntropyLoss: untuk multi-class classification (prediksi vocab token)
    # ignore_index=-100: abaikan target bernilai -100 (token non-masked)
    
    rtd_criterion = nn.BCEWithLogitsLoss()
    # BCEWithLogitsLoss: Binary Cross-Entropy untuk binary classification (replaced/not)
    # Penggabungan fungsi sigmoid + BCELoss dalam satu langkah agar komputasi lebih stabil

    # ===== Optimizer =====
    optimizer = torch.optim.AdamW(
        list(generator.parameters()) +
        list(discriminator.parameters()) +
        list(embedding_layer.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # AdamW (Adam dengan Weight Decay):
    # - Adaptive Moment Estimation: update parameter dengan adaptive learning rate per parameter
    # - m (first moment): running average dari gradients (momentum)
    # - v (second moment): running average dari squared gradients (RMSprop)
    # - Parameter update: param -= lr * m / (sqrt(v) + eps)
    # - Weight decay: L2 regularization untuk mencegah overfitting (decay = 0.01 umum)

    total_steps = num_epochs * len(dataloader)

    # ===== Scheduler =====
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    # get_cosine_schedule_with_warmup: 
    # - LR turun mengikuti cosine curve dari learning_rate → 0
    # - Bentuk: lr = learning_rate * (1 + cos(π * t / T)) / 2
    # - t: current step, T: total steps
    # - Lebih smooth daripada linear decay
    # - Umum di NLP/Vision models (BERT, GPT, ViT)

    global_step = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch in dataloader:
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # =========================================================
            # 1. Masked Language Modeling (Generator)
            # =========================================================
            masked_input_ids, mlm_labels = create_mlm_inputs(
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                special_token_ids=special_token_ids,
            )

            masked_embeddings = embedding_layer(masked_input_ids)

            gen_logits = generator(
                input_embeddings=masked_embeddings,
                attention_mask=attention_mask,
            )

            mlm_loss = mlm_criterion(
                gen_logits.view(-1, vocab_size),
                mlm_labels.view(-1),
            )

            # =========================================================
            # 2. Replaced Token Detection (Discriminator)
            # =========================================================
            replaced_input_ids, rtd_labels = replace_with_generator(
                input_ids=input_ids,
                masked_input_ids=masked_input_ids,
                mlm_labels=mlm_labels,
                generator_logits=gen_logits.detach(),  # stop grad ke generator
            )
            # .detach(): putus gradient flow dari gen_logits
            # - Discriminator training hanya meng-update discriminator weights
            # - Generator sudah diupdate di step MLM loss
            # - .detach() mencegah double-backprop ke generator

            replaced_embeddings = embedding_layer(replaced_input_ids)

            disc_logits = discriminator(
                input_embeddings=replaced_embeddings,
                attention_mask=attention_mask,
            )

            rtd_loss = rtd_criterion(
                disc_logits.view(-1),
                rtd_labels.float().view(-1),
            )
            # .view(-1): reshape tensor menjadi 1D (flatten) untuk perhitungan loss
            # (-1 berarti: hitung dimensi otomatis agar total elemen tetap sama)
            # .float(): konversi int64 ke float32 (BCEWithLogitsLoss butuh float)

            # ===== Total Loss + Backward =====
            total_loss = mlm_loss + rtd_loss_weight * rtd_loss

            optimizer.zero_grad()
            # .zero_grad(): reset gradient accumulator ke 0
            # Penting: tanpa ini, gradients dari step sebelumnya akan terakumulasi

            total_loss.backward()
            # .backward(): compute gradients untuk semua parameter (backpropagation)
            # Gradients disimpan di parameter.grad

            # ===== Gradient Clipping =====
            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters()) +
                list(discriminator.parameters()) +
                list(embedding_layer.parameters()),
                max_norm=1.0
            )
            # clip_grad_norm_: batasi magnitude gradients agar tidak terlalu besar
            # - Hitung norm total dari semua gradients: ||g|| = sqrt(sum(g_i^2))
            # - Jika ||g|| > max_norm: scale semua gradients: g_i = g_i * (max_norm / ||g||)
            # - Jika ||g|| <= max_norm: gradients tidak berubah
            # max_norm=1.0: umum untuk NLP models (BERT, GPT)

            optimizer.step()
            # .step(): update semua parameter berdasarkan gradients
            # AdamW 

            scheduler.step()
            # .step(): update learning rate sesuai schedule (cosine decay)

            if global_step % 100 == 0:
                print(
                    f"Step {global_step} | "
                    f"MLM Loss: {mlm_loss.item():.4f} | "
                    f"RTD Loss: {rtd_loss.item():.4f} | "
                    f"Total: {total_loss.item():.4f}"
                )

    print("\n✅ Pretraining selesai.")
