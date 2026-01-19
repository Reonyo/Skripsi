"""
Baseline 2: RTD Only
Train discriminator with RTD task only (no generator/MLM).
Tokens are randomly replaced from vocabulary.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import yaml
import torch
from torch.utils.data import DataLoader

from baselines.training.rtd_only import train_rtd_only
from models.discriminator import Discriminator
from data_loader.pretrain_dataset import PretrainDataset


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*60)
    print("BASELINE 2: RTD Only (No Generator/MLM)")
    print("="*60)
    
    # Load config
    cfg = load_yaml("baselines/configs/rtd_only.yaml")
    
    disc_cfg = cfg["discriminator"]
    train_params = cfg["training"]
    data_params = cfg["data"]
    token_params = cfg["token"]
    
    # Dataset
    train_dataset = PretrainDataset(
        data_path="data/pretrain/processed/train.pt",
        max_length=data_params["max_length"],
    )
    
    val_dataset = PretrainDataset(
        data_path="data/pretrain/processed/val.pt",
        max_length=data_params["max_length"],
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_params["batch_size"],
        shuffle=False,
    )
    
    # Embedding
    vocab_size = train_dataset.vocab_size
    
    disc_embedding = torch.nn.Embedding(
        vocab_size,
        disc_cfg["d_model"],
        padding_idx=token_params["special_token_ids"][0],
    )
    
    # Discriminator only
    discriminator = Discriminator(**disc_cfg)
    
    print(f"Discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M params")
    print(f"Corruption rate: {train_params['corruption_rate']}")
    
    # Train RTD only
    train_rtd_only(
        discriminator=discriminator,
        embedding_layer=disc_embedding,
        dataloader=train_loader,
        val_dataloader=val_loader,
        vocab_size=vocab_size,
        special_token_ids=token_params["special_token_ids"],
        device=device,
        max_steps=train_params["max_steps"],
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        warmup_steps=train_params["warmup_steps"],
        corruption_rate=train_params["corruption_rate"],
        validate_every=train_params.get("validate_every", 10_000),
        log_every=train_params.get("log_every", 100),
        output_dir="outputs/baselines/rtd_only",
    )


if __name__ == "__main__":
    main()
