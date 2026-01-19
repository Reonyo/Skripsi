import os
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


# ======================================================
# UTILITAS CSV LOGGING
# ======================================================

def _init_csv_logger(path, header):
    """
    Membuat file CSV dan menuliskan header jika file belum ada.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def _append_csv(path, row):
    """
    Menambahkan satu baris ke file CSV.
    """
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ======================================================
# FINETUNING LOOP
# ======================================================

def finetune(
    encoder,                 # Discriminator encoder hasil pretraining
    task_head,               # Classification / Regression head
    embedding_layer,         # Embedding layer (dipakai encoder)
    dataloader,
    val_dataloader,
    task_name: str,
    num_labels: int,
    device: torch.device,
    *,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    validate_every: int = 1,     # validasi per epoch (default GLUE)
    output_dir: str = "outputs/finetune",
):
    """
    Finetuning loop untuk task GLUE.

    Encoder:
    - Menggunakan Discriminator hasil pretraining
    - Diupdate selama finetuning (tidak di-freeze)

    Task Head:
    - Linear layer di atas [CLS] representation
    """

    os.makedirs(output_dir, exist_ok=True)

    # ==================================================
    # CSV LOGGER
    # ==================================================
    train_csv = os.path.join(output_dir, f"{task_name}_train_log.csv")
    val_csv = os.path.join(output_dir, f"{task_name}_val_log.csv")

    _init_csv_logger(
        train_csv,
        ["epoch", "step", "loss"],
    )
    _init_csv_logger(
        val_csv,
        ["epoch", "val_loss"],
    )

    # ==================================================
    # DEVICE
    # ==================================================
    encoder.to(device)
    task_head.to(device)
    embedding_layer.to(device)

    encoder.train()
    task_head.train()

    # ==================================================
    # LOSS FUNCTION
    # ==================================================
    # Klasifikasi → CrossEntropy
    # Regresi (STS-B) → MSE
    if num_labels == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # ==================================================
    # OPTIMIZER
    # ==================================================
    optimizer = AdamW(
        list(encoder.parameters())
        + list(task_head.parameters())
        + list(embedding_layer.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    global_step = 0

    # ==================================================
    # TRAINING LOOP
    # ==================================================
    for epoch in range(num_epochs):
        encoder.train()
        task_head.train()

        pbar = tqdm(dataloader, desc=f"[{task_name}] Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            global_step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # ==================================================
            # FORWARD
            # ==================================================
            embeddings = embedding_layer(input_ids)

            encoder_outputs = encoder(
                embeddings,
                attention_mask=attention_mask,
            )

            # Ambil representasi token [CLS]
            # Asumsi: token pertama = CLS
            cls_rep = encoder_outputs[:, 0, :]

            logits = task_head(cls_rep)

            # ==================================================
            # LOSS
            # ==================================================
            if num_labels == 1:
                loss = criterion(logits.squeeze(-1), labels.float())
            else:
                loss = criterion(logits, labels)

            # ==================================================
            # BACKPROP
            # ==================================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ==================================================
            # LOG TRAIN
            # ==================================================
            _append_csv(
                train_csv,
                [epoch, global_step, loss.item()],
            )

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ==================================================
        # VALIDATION (PER EPOCH)
        # ==================================================
        if (epoch + 1) % validate_every == 0:
            encoder.eval()
            task_head.eval()

            total_val_loss = 0.0
            steps = 0

            with torch.no_grad():
                for vbatch in val_dataloader:
                    v_input_ids = vbatch["input_ids"].to(device)
                    v_attention_mask = vbatch["attention_mask"].to(device)
                    v_labels = vbatch["labels"].to(device)

                    v_embeds = embedding_layer(v_input_ids)
                    v_out = encoder(v_embeds, attention_mask=v_attention_mask)
                    v_cls = v_out[:, 0, :]
                    v_logits = task_head(v_cls)

                    if num_labels == 1:
                        v_loss = criterion(
                            v_logits.squeeze(-1),
                            v_labels.float(),
                        )
                    else:
                        v_loss = criterion(v_logits, v_labels)

                    total_val_loss += v_loss.item()
                    steps += 1

            avg_val_loss = total_val_loss / steps

            _append_csv(
                val_csv,
                [epoch, avg_val_loss],
            )

            # ==================================================
            # SAVE CHECKPOINT
            # ==================================================
            ckpt_path = os.path.join(
                output_dir,
                f"{task_name}_epoch_{epoch+1}.pt",
            )
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "task_head": task_head.state_dict(),
                    "embedding": embedding_layer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

            print(
                f"[VAL] Epoch {epoch+1} | "
                f"Loss: {avg_val_loss:.4f}"
            )

    print(f"\n✅ Finetuning selesai untuk task {task_name}")
