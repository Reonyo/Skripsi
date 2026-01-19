import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from training.pretrain import pretrain
from models.generator import Generator
from models.discriminator import Discriminator
from data_loader.pretrain_dataset import PretrainDataset


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=== TEST PRETRAIN MODE ===")

    device = torch.device("cpu")
    print(f"Using device: {device}")

    model_cfg = load_yaml(
        r"C:\Users\MyBook Hype AMD\Documents\GitHub\Skripsi\configs\model_test.yaml"
    )
    train_cfg = load_yaml(
        r"C:\Users\MyBook Hype AMD\Documents\GitHub\Skripsi\configs\pretrain_test.yaml"
    )

    gen_cfg = model_cfg["generator"]
    disc_cfg = model_cfg["discriminator"]

    train_params = train_cfg["training"]
    data_params = train_cfg["data"]
    
    # Extract token params from data_params
    mask_token_id = data_params.get("mask_token_id", 4)
    special_token_ids = data_params.get("special_token_ids", [0, 1, 2, 3])

    train_dataset = PretrainDataset(
        data_path="data/pretrain/processed/train.pt",
        max_length=data_params["max_length"],
    )

    val_dataset = PretrainDataset(
        data_path="data/pretrain/processed/val.pt",
        max_length=data_params["max_length"],
    )

    # kecilkan data
    train_dataset.input_ids = train_dataset.input_ids[:50]
    train_dataset.attention_mask = train_dataset.attention_mask[:50]
    val_dataset.input_ids = val_dataset.input_ids[:20]
    val_dataset.attention_mask = val_dataset.attention_mask[:20]

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

    vocab_size = train_dataset.vocab_size

    gen_embedding = torch.nn.Embedding(
        vocab_size,
        gen_cfg["d_model"],
        padding_idx=special_token_ids[0],
    )

    disc_embedding = torch.nn.Embedding(
        vocab_size,
        disc_cfg["d_model"],
        padding_idx=special_token_ids[0],
    )

    generator = Generator(vocab_size=vocab_size, **gen_cfg)
    discriminator = Discriminator(**disc_cfg)

    pretrain(
        generator=generator,
        discriminator=discriminator,
        gen_embedding=gen_embedding,
        disc_embedding=disc_embedding,
        dataloader=train_loader,
        val_dataloader=val_loader,
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        special_token_ids=special_token_ids,
        device=device,
        max_steps=train_params["max_steps"],  # Step-based training
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        warmup_steps=train_params["warmup_steps"],
        rtd_loss_weight=train_params["rtd_loss_weight"],
        validate_every=train_params["validate_every"],
        log_every=2,  # Log every 2 steps for testing (main will use 100)
        output_dir="test/outputs/test_pretrain",
    )

    print("✅ TEST PRETRAIN SELESAI")


if __name__ == "__main__":
    main()
