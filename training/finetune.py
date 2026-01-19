import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from training.utils.csv_logger import init_csv_logger, append_csv
from training.utils.metrics import compute_metrics


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
    log_every: int = 50,         # simpan loss train per N step
    early_stopping_patience: int = 3,  # early stopping dengan 3 epoch patience
    output_dir: str = "outputs/finetune",
):
    """
    Finetuning loop untuk task GLUE.

    Encoder:
    - Menggunakan Discriminator hasil pretraining
    - Diupdate selama finetuning (tidak di-freeze)

    Task Head:
    - Linear layer di atas [CLS] representation
    
    Early Stopping:
    - Monitor validation loss
    - Stop jika loss tidak improve selama `early_stopping_patience` epoch
    - Max epoch tetap num_epochs
    """

    os.makedirs(output_dir, exist_ok=True)

    # ==================================================
    # EARLY STOPPING TRACKING
    # ==================================================
    best_val_loss = float('inf')
    patience_counter = 0

    # ==================================================
    # CSV LOGGER
    # ==================================================
    train_csv = os.path.join(output_dir, f"{task_name}_train_log.csv")
    val_csv = os.path.join(output_dir, f"{task_name}_val_log.csv")

    init_csv_logger(
        train_csv,
        ["epoch", "step", "loss"],
    )
    init_csv_logger(
        val_csv,
        ["epoch", "val_loss", "val_accuracy", "val_f1", "val_mcc", "val_pearson", "val_spearman"],
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

            encoder_outputs = encoder.encoder(
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
            if global_step % log_every == 0:
                append_csv(
                    train_csv,
                    [epoch + 1, global_step, loss.item()],
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
            all_logits = []
            all_labels = []

            with torch.no_grad():
                for vbatch in val_dataloader:
                    v_input_ids = vbatch["input_ids"].to(device)
                    v_attention_mask = vbatch["attention_mask"].to(device)
                    v_labels = vbatch["labels"].to(device)

                    v_embeds = embedding_layer(v_input_ids)
                    v_out = encoder.encoder(v_embeds, attention_mask=v_attention_mask)
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

                    all_logits.append(v_logits.detach().cpu())
                    all_labels.append(v_labels.detach().cpu())

            avg_val_loss = total_val_loss / steps

            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            metrics = compute_metrics(task_name, num_labels, logits_cat, labels_cat)

            append_csv(
                val_csv,
                [
                    epoch + 1,
                    avg_val_loss,
                    metrics["val_accuracy"],
                    metrics["val_f1"],
                    metrics["val_mcc"],
                    metrics["val_pearson"],
                    metrics["val_spearman"],
                ],
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

            # ==================================================
            # EARLY STOPPING CHECK
            # ==================================================
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"  [IMPROVE] Val loss improved to {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  [NO IMPROVE] Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"[EARLY STOP] No improvement for {early_stopping_patience} epochs. Stopping.")
                    break

    print(f"\n[OK] Finetuning selesai untuk task {task_name}")
