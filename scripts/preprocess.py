import os
import argparse
from pathlib import Path
from typing import List

import torch
from tokenizers import Tokenizer
from tqdm import tqdm


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_texts(paths: List[str]):
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def preprocess_split(
    texts,
    tokenizer: Tokenizer,
    max_length: int,
):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Tokenizing"):
        enc = tokenizer.encode(text)

        ids = enc.ids[:max_length]
        mask = [1] * len(ids)

        # padding
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids += [tokenizer.token_to_id("[PAD]")] * pad_len
            mask += [0] * pad_len

        input_ids.append(ids)
        attention_masks.append(mask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tokenizer_dir", default="data/tokenizer")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(
        os.path.join(args.tokenizer_dir, "tokenizer.json")
    )

    if args.task == "pretrain":
        raw_train = [
            "data/pretrain/raw/train/c4.txt",
            "data/pretrain/raw/train/wikipedia.txt",
        ]
        raw_val = [
            "data/pretrain/raw/val/c4.txt",
            "data/pretrain/raw/val/wikipedia.txt",
        ]
        out_base = "data/pretrain/processed"

    else:  # finetune (GLUE etc.)
        raw_train = ["data/glue/raw/train.txt"]
        raw_val = ["data/glue/raw/val.txt"]
        out_base = "data/glue/processed"

    ensure_dir(out_base)

    print("Processing TRAIN split...")
    train_data = preprocess_split(
        load_texts(raw_train),
        tokenizer,
        args.max_length,
    )
    torch.save(train_data, os.path.join(out_base, "train.pt"))

    print("Processing VAL split...")
    val_data = preprocess_split(
        load_texts(raw_val),
        tokenizer,
        args.max_length,
    )
    torch.save(val_data, os.path.join(out_base, "val.pt"))

    print("✅ Preprocessing finished")
    print(f"Saved to {out_base}")

if __name__ == "__main__":
    main()
