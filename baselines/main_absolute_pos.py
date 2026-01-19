"""
Baseline 1: Absolute Positional Encoding
Train ELECTRA model with standard absolute PE instead of RoPE.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import yaml
import torch
from torch.utils.data import DataLoader

from training.pretrain import pretrain
from models.generator import Generator
from baselines.models.absolute_pos_encoder import AbsolutePosEncoder
from data_loader.pretrain_dataset import PretrainDataset


class DiscriminatorAbsolutePos(torch.nn.Module):
    """Discriminator wrapper using Absolute PE encoder."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = AbsolutePosEncoder(**kwargs)
        d_model = kwargs['d_model']
        self.output_layer = torch.nn.Linear(d_model, 1)
    
    def forward(self, x, attention_mask=None):
        hidden = self.encoder(x, attention_mask)
        return self.output_layer(hidden).squeeze(-1)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("="*60)
    print("BASELINE 1: Absolute Positional Encoding")
    print("="*60)
    
    # Load config
    cfg = load_yaml("baselines/configs/absolute_pos.yaml")
    
    gen_cfg = cfg["model"]["generator"]
    disc_cfg = cfg["model"]["discriminator"]
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
    
    gen_embedding = torch.nn.Embedding(
        vocab_size,
        gen_cfg["d_model"],
        padding_idx=token_params["special_token_ids"][0],
    )
    
    disc_embedding = torch.nn.Embedding(
        vocab_size,
        disc_cfg["d_model"],
        padding_idx=token_params["special_token_ids"][0],
    )
    
    # Models - Generator uses RoPE (standard), Discriminator uses Absolute PE
    generator = Generator(vocab_size=vocab_size, **gen_cfg)
    discriminator = DiscriminatorAbsolutePos(**disc_cfg)
    
    print(f"Generator: {sum(p.numel() for p in generator.parameters())/1e6:.2f}M params")
    print(f"Discriminator (Absolute PE): {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M params")
    
    # Pretrain
    pretrain(
        generator=generator,
        discriminator=discriminator,
        gen_embedding=gen_embedding,
        disc_embedding=disc_embedding,
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
        rtd_loss_weight=train_params["rtd_loss_weight"],
        validate_every=train_params.get("validate_every", 10_000),
        log_every=train_params.get("log_every", 100),
        output_dir="outputs/baselines/absolute_pos",
    )


if __name__ == "__main__":
    main()
