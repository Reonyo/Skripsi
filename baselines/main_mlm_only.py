"""
Baseline 3: MLM Only
Train large encoder with MLM task only (BERT-style).
Uses discriminator-size model (12 layers, 768 dim).
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import yaml
import torch
from torch.utils.data import DataLoader

from baselines.training.mlm_only import train_mlm_only
from models.discriminator import Discriminator
from data_loader.pretrain_dataset import PretrainDataset


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*60)
    print("BASELINE 3: MLM Only (BERT-style, Large Model)")
    print("="*60)
    
    # Load config
    cfg = load_yaml("baselines/configs/mlm_only.yaml")
    
    enc_cfg = cfg["encoder"]
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
    
    embedding = torch.nn.Embedding(
        vocab_size,
        enc_cfg["d_model"],
        padding_idx=token_params["special_token_ids"][0],
    )
    
    # Encoder (same size as discriminator)
    encoder = Discriminator(**enc_cfg)
    
    print(f"Encoder (MLM): {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params")
    print(f"MLM probability: {train_params['mlm_probability']}")
    
    # Train MLM only
    train_mlm_only(
        encoder=encoder,
        embedding_layer=embedding,
        dataloader=train_loader,
        val_dataloader=val_loader,
        vocab_size=vocab_size,
        mask_token_id=token_params["mask_token_id"],
        special_token_ids=token_params["special_token_ids"],
        device=device,
        max_steps=train_params["max_steps"],
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        warmup_steps=train_params["warmup_steps"],
        mlm_probability=train_params["mlm_probability"],
        validate_every=train_params.get("validate_every", 10_000),
        log_every=train_params.get("log_every", 100),
        output_dir="outputs/baselines/mlm_only",
    )


if __name__ == "__main__":
    main()
