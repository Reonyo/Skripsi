import os
import argparse
from typing import Iterable
from datasets import load_dataset

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_lines(path: str, lines: Iterable[str], overwrite: bool):
    if os.path.exists(path) and not overwrite:
        print(f"[SKIP] {path} already exists")
        return

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.replace("\n", " ") + "\n")


def download_c4(train_dir: str, val_dir: str,
                n_train: int, n_val: int, overwrite: bool):
    """
    C4 from HuggingFace Hub (allenai/c4), streaming.
    """
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    train_file = os.path.join(train_dir, "c4.txt")
    val_file = os.path.join(val_dir, "c4.txt")

    if (os.path.exists(train_file) and os.path.exists(val_file)) and not overwrite:
        print("[SKIP] C4 train/val already exists")
        return

    print(f"Downloading C4 (train={n_train}, val={n_val})")

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True
    )

    train_texts, val_texts = [], []

    for i, item in enumerate(dataset):
        if i < n_train:
            train_texts.append(item["text"])
        elif i < n_train + n_val:
            val_texts.append(item["text"])
        else:
            break

    write_lines(train_file, train_texts, overwrite)
    write_lines(val_file, val_texts, overwrite)

def download_wikipedia(train_dir: str, val_dir: str,
                       n_train: int, n_val: int, overwrite: bool):
    ensure_dir(train_dir)
    ensure_dir(val_dir)

    train_file = os.path.join(train_dir, "wikipedia.txt")
    val_file = os.path.join(val_dir, "wikipedia.txt")

    if (os.path.exists(train_file) and os.path.exists(val_file)) and not overwrite:
        print("[SKIP] Wikipedia train/val already exists")
        return

    print(f"Downloading Wikipedia (streaming, train={n_train}, val={n_val})")

    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )

    train_texts, val_texts = [], []

    for item in dataset:
        text = item.get("text", "").strip() # type: ignore
        if not text:
            continue

        if len(train_texts) < n_train:
            train_texts.append(text)
        elif len(val_texts) < n_val:
            val_texts.append(text)
        else:
            break

    write_lines(train_file, train_texts, overwrite)
    write_lines(val_file, val_texts, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # TOTAL sizes (will be split 95/5)
    parser.add_argument("--c4_total", type=int, default=125_000)
    parser.add_argument("--wiki_total", type=int, default=5_000)

    args = parser.parse_args()

    if args.debug:
        args.c4_total = 1_000
        args.wiki_total = 1_000

    # 95% train, 5% val
    c4_train = int(args.c4_total * 0.95)
    c4_val = args.c4_total - c4_train

    wiki_train = int(args.wiki_total * 0.95)
    wiki_val = args.wiki_total - wiki_train

    base_dir = "data/pretrain/raw"

    download_c4(
        train_dir=os.path.join(base_dir, "train"),
        val_dir=os.path.join(base_dir, "val"),
        n_train=c4_train,
        n_val=c4_val,
        overwrite=args.overwrite
    )

    download_wikipedia(
        train_dir=os.path.join(base_dir, "train"),
        val_dir=os.path.join(base_dir, "val"),
        n_train=wiki_train,
        n_val=wiki_val,
        overwrite=args.overwrite
    )

    print("Pretraining data (train/val) ready.")
