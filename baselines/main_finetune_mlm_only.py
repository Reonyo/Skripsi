"""
Baseline Finetune: MLM-Only
Fine-tune MLM-only baseline on GLUE tasks.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from training.finetune import finetune
from models.discriminator import Discriminator
from data_loader.glue_dataset import GlueDataset


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*60)
    print("BASELINE FINETUNE: MLM-Only")
    print("="*60)
    
    # Load configs
    model_cfg = load_yaml("baselines/configs/mlm_only.yaml")
    finetune_cfg = load_yaml("baselines/configs/finetune_mlm_only.yaml")
    
    disc_cfg = model_cfg["model"]["discriminator"]
    train_params = finetune_cfg["training"]
    data_params = finetune_cfg["data"]
    task_params = finetune_cfg["task"]
    tasks_cfg = finetune_cfg.get("tasks", {})
    token_params = finetune_cfg["token"]
    checkpoint_cfg = finetune_cfg.get("checkpoint", {})
    
    task_name = task_params["name"]
    if task_name not in tasks_cfg:
        raise ValueError(f"Task {task_name} not found in finetune config")
    
    task_meta = tasks_cfg[task_name]
    num_labels = task_meta["num_labels"]
    
    # Dataset
    train_dataset = GlueDataset(
        data_path=f"data/glue/tokenized/{task_name}/train.pt",
        task_name=task_name,
        max_length=data_params["max_length"],
    )
    
    val_dataset = GlueDataset(
        data_path=f"data/glue/tokenized/{task_name}/validation.pt",
        task_name=task_name,
        max_length=data_params["max_length"],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Embedding
    vocab_size = train_dataset.vocab_size
    embedding_layer = torch.nn.Embedding(
        vocab_size,
        disc_cfg["d_model"],
        padding_idx=token_params["special_token_ids"][0],
    )
    
    # Load pretrained discriminator (used as MLM encoder)
    discriminator = Discriminator(**disc_cfg)
    
    print("Loading pretrained MLM-only weights...")
    pretrained_path = checkpoint_cfg.get(
        "pretrained_path", 
        "outputs/baselines/mlm_only/checkpoint_step_100000.pt"
    )
    
    if os.path.exists(pretrained_path):
        pretrained_ckpt = torch.load(pretrained_path, map_location="cpu")
        discriminator.load_state_dict(pretrained_ckpt["mlm_model"])
        embedding_layer.load_state_dict(pretrained_ckpt["embedding"])
        print(f"Loaded checkpoint from {pretrained_path}")
    else:
        print(f"Warning: No pretrained checkpoint found at {pretrained_path}")
        print("Training from scratch...")
    
    # Task head
    task_head = torch.nn.Linear(disc_cfg["d_model"], num_labels)
    
    # Output directory
    output_dir = Path(f"outputs/baselines/mlm_only_finetune/{task_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Finetune
    finetune(
        encoder=discriminator,
        task_head=task_head,
        embedding_layer=embedding_layer,
        dataloader=train_loader,
        val_dataloader=val_loader,
        task_name=task_name,
        num_labels=num_labels,
        device=device,
        num_epochs=train_params["num_epochs"],
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        validate_every=train_params.get("validate_every", 1),
        log_every=train_params.get("log_every", 50),
        early_stopping_patience=train_params.get("early_stopping_patience", 3),
        output_dir=str(output_dir),
    )
    
    print(f"\n[OK] MLM-only baseline finetuning complete for {task_name}")


if __name__ == "__main__":
    main()
