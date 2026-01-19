"""
MLM-only training loop (baseline without RTD/Discriminator).
Standard BERT-style masked language modeling with discriminator-size model.
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from training.masking import create_mlm_inputs
from training.utils.csv_logger import init_csv_logger, append_csv


def train_mlm_only(
    encoder,
    embedding_layer,
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
    mlm_probability=0.15,
    validate_every=10_000,
    log_every=100,
    output_dir="outputs/baselines/mlm_only",
):
    """MLM-only training loop (BERT-style, large model)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # CSV Logger
    train_csv = os.path.join(output_dir, "train_log.csv")
    val_csv = os.path.join(output_dir, "val_log.csv")
    
    init_csv_logger(train_csv, ["global_step", "mlm_loss"])
    init_csv_logger(val_csv, ["global_step", "mlm_loss", "mlm_acc"])
    
    # Device
    encoder.to(device)
    embedding_layer.to(device)
    
    # MLM prediction head
    mlm_head = nn.Linear(encoder.layers[0].norm_attn.normalized_shape[0], vocab_size).to(device)
    
    # Loss & Optimizer
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(
        list(encoder.parameters()) + list(embedding_layer.parameters()) + list(mlm_head.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    global_step = 0
    dataloader_iter = iter(dataloader)
    
    # Training loop (step-based)
    while global_step < max_steps:
        encoder.train()
        mlm_head.train()
        
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
        
        # Create MLM inputs
        masked_input_ids, mlm_labels = create_mlm_inputs(
            input_ids,
            mask_token_id,
            special_token_ids,
            mlm_probability=mlm_probability,
        )
        
        # Encoder forward
        embeds = embedding_layer(masked_input_ids)
        hidden_states = encoder(embeds, attention_mask=attention_mask)
        
        # MLM prediction
        mlm_logits = mlm_head(hidden_states)
        
        # MLM loss
        mlm_loss = mlm_criterion(
            mlm_logits.view(-1, vocab_size),
            mlm_labels.view(-1),
        )
        
        # Backprop
        optimizer.zero_grad()
        mlm_loss.backward()
        optimizer.step()
        
        # Log train
        if global_step % log_every == 0:
            append_csv(train_csv, [global_step, f"{mlm_loss.item():.4f}"])
            print(f"MLM: {mlm_loss.item():.4f}")
        
        # Validation
        if global_step % validate_every == 0:
            encoder.eval()
            mlm_head.eval()
            
            val_mlm = 0.0
            val_mlm_correct, val_mlm_total = 0, 0
            n_val = 0
            
            with torch.no_grad():
                for vbatch in val_dataloader:
                    v_input_ids = vbatch["input_ids"].to(device)
                    v_attention_mask = vbatch["attention_mask"].to(device)
                    
                    v_masked_input_ids, v_mlm_labels = create_mlm_inputs(
                        v_input_ids,
                        mask_token_id,
                        special_token_ids,
                        mlm_probability=mlm_probability,
                    )
                    
                    v_embeds = embedding_layer(v_masked_input_ids)
                    v_hidden = encoder(v_embeds, attention_mask=v_attention_mask)
                    v_mlm_logits = mlm_head(v_hidden)
                    
                    v_mlm = mlm_criterion(
                        v_mlm_logits.view(-1, vocab_size),
                        v_mlm_labels.view(-1),
                    )
                    
                    # MLM accuracy
                    mlm_preds = v_mlm_logits.argmax(dim=-1).view(-1)
                    mlm_labels_flat = v_mlm_labels.view(-1)
                    mlm_mask = mlm_labels_flat != -100
                    val_mlm_correct += (mlm_preds[mlm_mask] == mlm_labels_flat[mlm_mask]).sum().item()
                    val_mlm_total += mlm_mask.sum().item()
                    
                    val_mlm += v_mlm.item()
                    n_val += 1
            
            val_mlm /= n_val
            mlm_acc = val_mlm_correct / val_mlm_total if val_mlm_total > 0 else 0.0
            
            append_csv(val_csv, [global_step, f"{val_mlm:.4f}", f"{mlm_acc:.4f}"])
            print(f"[VAL] Step {global_step} | MLM Loss: {val_mlm:.4f} Acc: {mlm_acc:.4f}")
            
            # Save checkpoint
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "embedding": embedding_layer.state_dict(),
                "mlm_head": mlm_head.state_dict(),
                "step": global_step,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            
            encoder.train()
            mlm_head.train()
        
        if global_step >= max_steps:
            break
    
    print(f"[DONE] MLM-only training completed at step {global_step}/{max_steps}")
