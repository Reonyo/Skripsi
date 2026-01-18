import os
import argparse
from typing import Iterable, List

import numpy as np
from tokenizers import Tokenizer


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_text(files: List[str]) -> Iterable[str]:
    for path in files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield text


def tokenize_and_write(
    tokenizer: Tokenizer,
    texts: Iterable[str],
    out_prefix: str,
    max_length: int,
):
    """
    Writes:
      - out_prefix.bin : int32 token ids
      - out_prefix.idx : document boundaries
    """
    all_ids = []
    doc_idx = []

    for text in texts:
        enc = tokenizer.encode(text)
        ids = enc.ids[:max_length]

        if len(ids) == 0:
            continue

        doc_idx.append(len(all_ids))
        all_ids.extend(ids)

    all_ids = np.array(all_ids, dtype=np.int32)
    doc_idx = np.array(doc_idx, dtype=np.int64)

    all_ids.tofile(out_prefix + ".bin")
    doc_idx.tofile(out_prefix + ".idx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tokenizer_path = "data/tokenizer/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    raw_dir = "data/pretrain/raw"
    out_dir = "data/pretrain/tokenized"
    ensure_dir(out_dir)

    # ---------- TRAIN ----------
    train_files = [
        os.path.join(raw_dir, "train", "c4.txt"),
        os.path.join(raw_dir, "train", "wikipedia.txt"),
    ]

    print("Tokenizing TRAIN data...")
    tokenize_and_write(
        tokenizer=tokenizer,
        texts=iter_text(train_files),
        out_prefix=os.path.join(out_dir, "train"),
        max_length=args.max_length,
    )

    # ---------- VAL ----------
    val_files = [
        os.path.join(raw_dir, "val", "c4.txt"),
        os.path.join(raw_dir, "val", "wikipedia.txt"),
    ]

    print("Tokenizing VAL data...")
    tokenize_and_write(
        tokenizer=tokenizer,
        texts=iter_text(val_files),
        out_prefix=os.path.join(out_dir, "val"),
        max_length=args.max_length,
    )

    print("Tokenization finished.")
