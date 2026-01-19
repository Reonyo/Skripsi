import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from training.masking import create_mlm_inputs, replace_with_generator
from training.utils.csv_logger import init_csv_logger, append_csv


def pretrain(
    generator,
    discriminator,
    gen_embedding,
    disc_embedding,
    dataloader,
    val_dataloader,
    vocab_size,
    mask_token_id,
    special_token_ids,
    device,
    max_steps,
    learning_rate,
    weight_decay,
    warmup_steps,
    rtd_loss_weight,
    validate_every=10_000,
    log_every=100,
    output_dir="outputs/pretrain",
):
    """
    Fungsi utama pretraining (MLM + RTD)

    Semua mekanisme:
    - training loop
    - validation loop
    - logging (CSV)
    - saving checkpoint
    DIPUSATKAN DI SINI
    """

    os.makedirs(output_dir, exist_ok=True)

    # ======================================================
    # CSV LOGGER
    # ======================================================
    train_csv = os.path.join(output_dir, "train_log.csv")
    val_csv = os.path.join(output_dir, "val_log.csv")

    init_csv_logger(
        train_csv,
        ["global_step", "mlm_loss", "rtd_loss", "total_loss"],
    )
    init_csv_logger(
        val_csv,
        ["global_step", "mlm_loss", "rtd_loss", "total_loss", "mlm_acc", "rtd_acc"],
    )

    # ======================================================
    # DEVICE
    # ======================================================
    generator.to(device)
    discriminator.to(device)
    gen_embedding.to(device)
    disc_embedding.to(device)

    # ======================================================
    # OPTIMIZER + LOSS
    # ======================================================
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    rtd_criterion = nn.BCEWithLogitsLoss()
    
    optimizer = AdamW(
        list(generator.parameters())
        + list(discriminator.parameters())
        + list(gen_embedding.parameters())
        + list(disc_embedding.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    global_step = 0

    # ======================================================
    # TRAINING LOOP (STEP-BASED)
    # ======================================================
    dataloader_iter = iter(dataloader)
    
    while global_step < max_steps:
        generator.train()
        discriminator.train()
        
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Restart dataloader when epoch ends
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        global_step += 1
        
        # Show progress
        if global_step == 1 or global_step % log_every == 0 or global_step == max_steps:
            print(f"[Step {global_step}/{max_steps}]", end=" ")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # --------------------------------------------------
            # 1. CREATE MLM INPUTS
            # --------------------------------------------------
            masked_input_ids, mlm_labels = create_mlm_inputs(
                input_ids,
                mask_token_id,
                special_token_ids,
                mlm_probability=0.15,
            )

            # --------------------------------------------------
            # 2. GENERATOR (MLM)
            # --------------------------------------------------
            gen_embeds = gen_embedding(masked_input_ids)
            gen_logits = generator(
                gen_embeds,
                attention_mask=attention_mask,
            )

            mlm_loss = mlm_criterion(
                gen_logits.view(-1, vocab_size),
                mlm_labels.view(-1),
            )

            # --------------------------------------------------
            # 3. REPLACE WITH GENERATOR PREDICTIONS
            # --------------------------------------------------
            replaced_input_ids, rtd_labels = replace_with_generator(
                input_ids,
                masked_input_ids,
                mlm_labels,
                gen_logits,
            )

            # --------------------------------------------------
            # 4. DISCRIMINATOR (RTD)
            # --------------------------------------------------
            disc_embeds = disc_embedding(replaced_input_ids)
            disc_logits = discriminator(
                disc_embeds,
                attention_mask=attention_mask,
            )

            rtd_loss = rtd_criterion(
                disc_logits.view(-1),
                rtd_labels.float().view(-1),
            )

            # --------------------------------------------------
            # 5. TOTAL LOSS
            # --------------------------------------------------
            total_loss = mlm_loss + rtd_loss_weight * rtd_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --------------------------------------------------
            # 6. LOG TRAIN (periodic)
            # --------------------------------------------------
            if global_step % log_every == 0:
                append_csv(
                    train_csv,
                    [
                        global_step,
                        f"{mlm_loss.item():.4f}",
                        f"{rtd_loss.item():.4f}",
                        f"{total_loss.item():.4f}",
                    ],
                )
                print(f"MLM: {mlm_loss.item():.4f} | RTD: {rtd_loss.item():.4f} | Total: {total_loss.item():.4f}")

            # Continue to next step

            # --------------------------------------------------
            # 7. VALIDATION
            # --------------------------------------------------
            if global_step % validate_every == 0:
                generator.eval()
                discriminator.eval()

                val_mlm, val_rtd, val_total = 0.0, 0.0, 0.0
                val_mlm_correct, val_rtd_correct = 0, 0
                val_mlm_total, val_rtd_total = 0, 0
                n_val = 0

                with torch.no_grad():
                    for vbatch in val_dataloader:
                        v_input_ids = vbatch["input_ids"].to(device)
                        v_attention_mask = vbatch["attention_mask"].to(device)

                        # MLM
                        v_masked_input_ids, v_mlm_labels = create_mlm_inputs(
                            v_input_ids,
                            mask_token_id,
                            special_token_ids,
                            mlm_probability=0.15,
                        )

                        v_gen_embeds = gen_embedding(v_masked_input_ids)
                        v_gen_logits = generator(
                            v_gen_embeds,
                            attention_mask=v_attention_mask,
                        )

                        v_mlm = mlm_criterion(
                            v_gen_logits.view(-1, vocab_size),
                            v_mlm_labels.view(-1),
                        )

                        # MLM Accuracy: only count masked positions (label != -100)
                        mlm_preds = v_gen_logits.argmax(dim=-1).view(-1)  # Flatten predictions
                        mlm_labels_flat = v_mlm_labels.view(-1)
                        mlm_mask = mlm_labels_flat != -100
                        val_mlm_correct += (mlm_preds[mlm_mask] == mlm_labels_flat[mlm_mask]).sum().item()
                        val_mlm_total += mlm_mask.sum().item()

                        # RTD
                        v_rep_ids, v_rtd_labels = replace_with_generator(
                            v_input_ids,
                            v_masked_input_ids,
                            v_mlm_labels,
                            v_gen_logits,
                        )

                        v_disc_embeds = disc_embedding(v_rep_ids)
                        v_disc_logits = discriminator(
                            v_disc_embeds,
                            attention_mask=v_attention_mask,
                        )

                        v_rtd = rtd_criterion(
                            v_disc_logits.view(-1),
                            v_rtd_labels.float().view(-1),
                        )

                        # RTD Accuracy: binary classification (replaced=1, original=0)
                        rtd_preds = (v_disc_logits > 0).long()
                        val_rtd_correct += (rtd_preds == v_rtd_labels).sum().item()
                        val_rtd_total += v_rtd_labels.numel()

                        v_total = v_mlm + rtd_loss_weight * v_rtd

                        val_mlm += v_mlm.item()
                        val_rtd += v_rtd.item()
                        val_total += v_total.item()
                        n_val += 1

                val_mlm /= n_val
                val_rtd /= n_val
                val_total /= n_val
                
                # Calculate accuracies
                mlm_acc = val_mlm_correct / val_mlm_total if val_mlm_total > 0 else 0.0
                rtd_acc = val_rtd_correct / val_rtd_total if val_rtd_total > 0 else 0.0

                append_csv(
                    val_csv,
                    [global_step, f"{val_mlm:.4f}", f"{val_rtd:.4f}", f"{val_total:.4f}", 
                     f"{mlm_acc:.4f}", f"{rtd_acc:.4f}"],
                )
                
                print(f"[VAL] Step {global_step} | MLM Loss: {val_mlm:.4f} Acc: {mlm_acc:.4f} | RTD Loss: {val_rtd:.4f} Acc: {rtd_acc:.4f}")

                # --------------------------------------------------
                # SAVE CHECKPOINT
                # --------------------------------------------------
                ckpt_path = os.path.join(
                    output_dir, f"checkpoint_step_{global_step}.pt"
                )
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "gen_embedding": gen_embedding.state_dict(),
                        "disc_embedding": disc_embedding.state_dict(),
                        "step": global_step,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint: {ckpt_path}")

                generator.train()
                discriminator.train()
        
        # Stop training if we've reached max steps
        if global_step >= max_steps:
            break

    print(f"[DONE] Pretraining completed at step {global_step}/{max_steps}")
