"""
RTD-only training loop (baseline without MLM/Generator).
Tokens are randomly corrupted from vocabulary.
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from training.utils.csv_logger import init_csv_logger, append_csv


def create_rtd_inputs(input_ids, special_token_ids, corruption_rate=0.15):
    """
    Randomly replace tokens with random vocab tokens (not using generator).
    
    Args:
        input_ids: (batch_size, seq_len)
        special_token_ids: List of special tokens to not corrupt
        corruption_rate: Fraction of tokens to corrupt
    
    Returns:
        corrupted_ids: (batch_size, seq_len)
        labels: (batch_size, seq_len) - 1 if replaced, 0 if original
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    vocab_size = input_ids.max().item() + 1
    
    # Create mask for which tokens to corrupt (excluding special tokens)
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in special_token_ids:
        special_mask |= (input_ids == special_id)
    
    # Random selection of tokens to corrupt
    corrupt_mask = torch.rand(batch_size, seq_len, device=device) < corruption_rate
    corrupt_mask &= ~special_mask  # Don't corrupt special tokens
    
    # Generate random replacements
    random_tokens = torch.randint(5, vocab_size, (batch_size, seq_len), device=device)
    
    # Create corrupted input
    corrupted_ids = input_ids.clone()
    corrupted_ids[corrupt_mask] = random_tokens[corrupt_mask]
    
    # Labels: 1 if replaced, 0 if original
    labels = corrupt_mask.long()
    
    return corrupted_ids, labels


def train_rtd_only(
    discriminator,
    embedding_layer,
    dataloader,
    val_dataloader,
    vocab_size,
    special_token_ids,
    device,
    max_steps,
    learning_rate,
    weight_decay,
    warmup_steps,
    corruption_rate=0.15,
    validate_every=10_000,
    log_every=100,
    output_dir="outputs/baselines/rtd_only",
):
    """RTD-only training loop (no MLM/Generator)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV Logger
    train_csv = os.path.join(output_dir, "train_log.csv")
    val_csv = os.path.join(output_dir, "val_log.csv")
    
    init_csv_logger(train_csv, ["global_step", "rtd_loss"])
    init_csv_logger(val_csv, ["global_step", "rtd_loss", "rtd_acc"])
    
    # Device
    discriminator.to(device)
    embedding_layer.to(device)
    
    # Loss & Optimizer
    rtd_criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        list(discriminator.parameters()) + list(embedding_layer.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    global_step = 0
    dataloader_iter = iter(dataloader)
    
    # Training loop (step-based)
    while global_step < max_steps:
        discriminator.train()
        
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        global_step += 1
        
        if global_step == 1 or global_step % log_every == 0 or global_step == max_steps:
            print(f"[Step {global_step}/{max_steps}]", end=" ")
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Create corrupted inputs
        corrupted_ids, rtd_labels = create_rtd_inputs(
            input_ids, special_token_ids, corruption_rate
        )
        
        # Discriminator forward
        disc_embeds = embedding_layer(corrupted_ids)
        disc_logits = discriminator(disc_embeds, attention_mask=attention_mask)
        
        # RTD loss
        rtd_loss = rtd_criterion(
            disc_logits.view(-1),
            rtd_labels.float().view(-1),
        )
        
        # Backprop
        optimizer.zero_grad()
        rtd_loss.backward()
        optimizer.step()
        
        # Log train
        if global_step % log_every == 0:
            append_csv(train_csv, [global_step, f"{rtd_loss.item():.4f}"])
            print(f"RTD: {rtd_loss.item():.4f}")
        
        # Validation
        if global_step % validate_every == 0:
            discriminator.eval()
            
            val_rtd = 0.0
            val_rtd_correct, val_rtd_total = 0, 0
            n_val = 0
            
            with torch.no_grad():
                for vbatch in val_dataloader:
                    v_input_ids = vbatch["input_ids"].to(device)
                    v_attention_mask = vbatch["attention_mask"].to(device)
                    
                    v_corrupted_ids, v_rtd_labels = create_rtd_inputs(
                        v_input_ids, special_token_ids, corruption_rate
                    )
                    
                    v_disc_embeds = embedding_layer(v_corrupted_ids)
                    v_disc_logits = discriminator(v_disc_embeds, attention_mask=v_attention_mask)
                    
                    v_rtd = rtd_criterion(
                        v_disc_logits.view(-1),
                        v_rtd_labels.float().view(-1),
                    )
                    
                    # RTD accuracy
                    rtd_preds = (v_disc_logits > 0).long()
                    val_rtd_correct += (rtd_preds == v_rtd_labels).sum().item()
                    val_rtd_total += v_rtd_labels.numel()
                    
                    val_rtd += v_rtd.item()
                    n_val += 1
            
            val_rtd /= n_val
            rtd_acc = val_rtd_correct / val_rtd_total if val_rtd_total > 0 else 0.0
            
            append_csv(val_csv, [global_step, f"{val_rtd:.4f}", f"{rtd_acc:.4f}"])
            print(f"[VAL] Step {global_step} | RTD Loss: {val_rtd:.4f} Acc: {rtd_acc:.4f}")
            
            # Save checkpoint
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
            torch.save({
                "discriminator": discriminator.state_dict(),
                "disc_embedding": embedding_layer.state_dict(),
                "step": global_step,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            
            discriminator.train()
        
        if global_step >= max_steps:
            break
    
    print(f"[DONE] RTD-only training completed at step {global_step}/{max_steps}")
