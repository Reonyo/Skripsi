import yaml
import torch
from torch.utils.data import DataLoader

from training.pretrain import pretrain
from models.generator import Generator
from models.discriminator import Discriminator


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ======================================================
    # 1. DEVICE
    # ======================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======================================================
    # 2. LOAD CONFIG
    # ======================================================
    model_cfg = load_yaml("configs/model.yaml")
    train_cfg = load_yaml("configs/pretrain.yaml")

    gen_cfg = model_cfg["generator"]
    disc_cfg = model_cfg["discriminator"]

    train_params = train_cfg["training"]
    data_params = train_cfg["data"]
    token_params = train_cfg["token"]

    # ======================================================
    # 3. DATASET & DATALOADER
    # ======================================================
    from data_loader.pretrain_dataset import PretrainDataset
    # Dataset diasumsikan mengembalikan:
    # {"input_ids": Tensor, "attention_mask": Tensor}

    train_dataset = PretrainDataset(
        data_dir="data/pretrain/processed/train",
        max_length=data_params["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ======================================================
    # 4. EMBEDDING LAYER
    # ======================================================
    vocab_size = train_dataset.vocab_size

    embedding_layer = torch.nn.Embedding(
        vocab_size,
        disc_cfg["d_model"],  # embedding mengikuti discriminator
        padding_idx=token_params["special_token_ids"][0],
    )

    # ======================================================
    # 5. MODEL INITIALIZATION
    # ======================================================
    generator = Generator(
        vocab_size=vocab_size,
        **gen_cfg,
    )

    discriminator = Discriminator(
        **disc_cfg,
    )

    # ======================================================
    # 6. PRETRAINING
    # ======================================================
    pretrain(
        generator=generator,
        discriminator=discriminator,
        embedding_layer=embedding_layer,
        dataloader=train_loader,
        vocab_size=vocab_size,
        mask_token_id=token_params["mask_token_id"],
        special_token_ids=token_params["special_token_ids"],
        device=device,
        num_epochs=train_params["num_epochs"],
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        warmup_steps=train_params["warmup_steps"],
        rtd_loss_weight=train_params["rtd_loss_weight"],
    )


if __name__ == "__main__":
    main()
