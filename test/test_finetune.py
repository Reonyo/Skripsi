import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from training.finetune import finetune
from models.discriminator import Discriminator
import yaml


class SyntheticGlueDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, num_labels: int, task_type: str):
        self.input_ids = torch.randint(5, vocab_size, (num_samples, seq_len))
        self.attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
        if num_labels == 1:
            # regression (STS-B) range ~[0,5]
            self.labels = torch.rand(num_samples) * 5.0
        else:
            self.labels = torch.randint(0, num_labels, (num_samples,))
        self.task_type = task_type
        self.vocab_size = vocab_size

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_task(task_name: str, num_labels: int, task_type: str, disc_cfg: dict, checkpoint_path: Path, output_root: Path):
    device = torch.device("cpu")
    seq_len = 16

    # Load pretrained weights (from test pretrain) and align vocab size
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    vocab_size = ckpt["disc_embedding"]["weight"].shape[0]

    # Build datasets using the checkpoint vocab size
    train_ds = SyntheticGlueDataset(num_samples=8, seq_len=seq_len, vocab_size=vocab_size, num_labels=num_labels, task_type=task_type)
    val_ds = SyntheticGlueDataset(num_samples=4, seq_len=seq_len, vocab_size=vocab_size, num_labels=num_labels, task_type=task_type)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # Models
    encoder = Discriminator(**disc_cfg)
    embedding_layer = torch.nn.Embedding(vocab_size, disc_cfg["d_model"], padding_idx=0)
    head_out = num_labels if num_labels > 1 else 1
    task_head = torch.nn.Linear(disc_cfg["d_model"], head_out)

    encoder.load_state_dict(ckpt["discriminator"], strict=False)
    embedding_layer.load_state_dict(ckpt["disc_embedding"], strict=False)

    # Output dir per task
    out_dir = output_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    finetune(
        encoder=encoder,
        task_head=task_head,
        embedding_layer=embedding_layer,
        dataloader=train_loader,
        val_dataloader=val_loader,
        task_name=task_name,
        num_labels=num_labels,
        device=device,
        num_epochs=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        validate_every=1,
        log_every=1,
        output_dir=str(out_dir),
    )


def main():
    # Use small model config
    model_cfg = load_yaml(str(ROOT / "configs/model_test.yaml"))
    disc_cfg = model_cfg["discriminator"]

    # Use latest checkpoint from test_pretrain outputs
    pretrain_dir = ROOT / "test/outputs/test_pretrain"
    ckpts = sorted(pretrain_dir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in test/outputs/test_pretrain. Run test_pretrain.py first.")
    checkpoint_path = ckpts[-1]
    output_root = ROOT / "test/outputs/test_finetune"
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = {
        "SST2": {"num_labels": 2, "type": "classification"},
        "MNLI": {"num_labels": 3, "type": "classification"},
        "RTE": {"num_labels": 2, "type": "classification"},
        "QNLI": {"num_labels": 2, "type": "classification"},
        "CoLA": {"num_labels": 2, "type": "classification"},
        "QQP": {"num_labels": 2, "type": "classification"},
        "MRPC": {"num_labels": 2, "type": "classification"},
        "STSB": {"num_labels": 1, "type": "regression"},
    }

    print("Starting smoke finetune for GLUE tasks (synthetic data)...")
    for name, meta in tasks.items():
        print(f"\n--- Task: {name} ---")
        run_task(name, meta["num_labels"], meta["type"], disc_cfg, checkpoint_path, output_root)
        print(f"✅ Finished {name}")

    print("\n✅ Smoke finetune completed for all tasks.")


if __name__ == "__main__":
    main()
