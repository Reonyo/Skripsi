import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from training.pretrain import pretrain
from models.generator import Generator
from models.discriminator import Discriminator
from data_loader.pretrain_dataset import PretrainDataset


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ======================================================
    # DEVICE
    # ======================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======================================================
    # LOAD CONFIG
    # ======================================================
    model_cfg = load_yaml("configs/model.yaml")
    train_cfg = load_yaml("configs/pretrain.yaml")

    gen_cfg = model_cfg["generator"]
    disc_cfg = model_cfg["discriminator"]

    train_params = train_cfg["training"]
    data_params = train_cfg["data"]
    token_params = train_cfg["token"]

    # ======================================================
    # DATASET
    # ======================================================
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

    # ======================================================
    # EMBEDDING
    # ======================================================
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

    # ======================================================
    # MODEL
    # ======================================================
    generator = Generator(vocab_size=vocab_size, **gen_cfg)
    discriminator = Discriminator(**disc_cfg)

    # ======================================================
    # PRETRAIN
    # ======================================================
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
        max_steps=train_params["max_steps"],  # Step-based training
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        warmup_steps=train_params["warmup_steps"],
        rtd_loss_weight=train_params["rtd_loss_weight"],
        validate_every=train_params.get("validate_every", 10_000),
        log_every=train_params.get("log_every", 100),  # Log every 100 steps by default
        output_dir="outputs/pretrain",
    )


if __name__ == "__main__":
    main()
