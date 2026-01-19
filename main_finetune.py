import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tokenizers import Tokenizer

# Training loop
from training.finetune import finetune

# Model
from models.discriminator import Discriminator

# Dataset GLUE
from data_loader.glue_dataset import GlueDataset


def load_yaml(path: str):
    """
    Membaca file konfigurasi YAML dan mengembalikannya sebagai dictionary.
    """
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
    finetune_cfg = load_yaml("configs/finetune.yaml")

    disc_cfg = model_cfg["discriminator"]

    train_params = finetune_cfg["training"]
    data_params = finetune_cfg["data"]
    task_params = finetune_cfg["task"]
    tasks_cfg = finetune_cfg.get("tasks", {})
    token_params = finetune_cfg["token"]
    checkpoint_cfg = finetune_cfg.get("checkpoint", {})

    task_name_cfg = task_params["name"]  # default task name or "ALL"

    # Helper to run a single task end-to-end
    def run_task(task_name: str):
        if task_name not in tasks_cfg:
            raise ValueError(f"Task {task_name} tidak ditemukan di finetune.yaml -> tasks")

        task_meta = tasks_cfg[task_name]
        num_labels = task_meta["num_labels"]

        # ======================================================
        # 3. DATASET & DATALOADER (GLUE)
        # ======================================================
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

        # ======================================================
        # 4. EMBEDDING
        # ======================================================
        tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
        vocab_size = tokenizer.get_vocab_size()

        embedding_layer = torch.nn.Embedding(
            vocab_size,
            disc_cfg["d_model"],
            padding_idx=token_params["special_token_ids"][0],  # PAD
        )

        # ======================================================
        # 5. LOAD DISCRIMINATOR (PRETRAINED)
        # ======================================================
        discriminator = Discriminator(**disc_cfg)

        print(f"\nLoading pretrained weights for {task_name}...")
        pretrained_path = checkpoint_cfg.get(
            "pretrained_path", "outputs/pretrain/pretrained_model.pt"
        )
        if not Path(pretrained_path).exists():
            print(f"[WARN] Pretrained checkpoint tidak ditemukan: {pretrained_path}. Lanjut finetune dari scratch.")
        else:
            pretrained_ckpt = torch.load(pretrained_path, map_location="cpu")
            # Expect keys: discriminator, disc_embedding
            if "discriminator" in pretrained_ckpt:
                discriminator.load_state_dict(pretrained_ckpt["discriminator"])
            if "disc_embedding" in pretrained_ckpt:
                embedding_layer.load_state_dict(pretrained_ckpt["disc_embedding"])

        # ======================================================
        # 6. TASK HEAD
        # ======================================================
        # For both regression and classification, use regular Linear layer
        # Regression outputs will be clamped to 0-5 range during validation
        task_head = torch.nn.Linear(
            disc_cfg["d_model"],
            num_labels,
        )

        # ======================================================
        # 7. OUTPUT DIRECTORY
        # ======================================================
        output_dir = Path(f"outputs/finetune/{task_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ======================================================
        # 8. FINETUNING
        # ======================================================
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

    # Determine tasks to run
    if isinstance(task_name_cfg, str) and task_name_cfg.upper() in {"ALL", "*"}:
        tasks_to_run = list(tasks_cfg.keys())
    elif isinstance(task_name_cfg, str) and "," in task_name_cfg:
        tasks_to_run = [t.strip() for t in task_name_cfg.split(",") if t.strip()]
    else:
        tasks_to_run = [task_name_cfg]

    # Run tasks sequentially
    print("\nMenjalankan finetune untuk task: ", ", ".join(tasks_to_run))
    for tname in tasks_to_run:
        run_task(tname)

    print(f"\n✅ Finetuning selesai untuk semua task yang dipilih.")


if __name__ == "__main__":
    main()
