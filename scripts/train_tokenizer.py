import os
import argparse
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, NFD, StripAccents


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_corpus(files: list[str]) -> Iterable[str]:
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield line.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.vocab_size = 5_000

    c4_path = "data/pretrain/raw/train/c4.txt"
    wiki_path = "data/pretrain/raw/train/wikipedia.txt"
    out_dir = "data/tokenizer"

    ensure_dir(out_dir)

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalizer & pre-tokenizer
    tokenizer.normalizer = Sequence([NFD(), StripAccents()]) # type: ignore
    tokenizer.pre_tokenizer = Whitespace() # type: ignore

    # Trainer (positional args to satisfy Pylance)
    trainer = BpeTrainer(
    vocab_size=args.vocab_size,  # type: ignore
    min_frequency=args.min_frequency, # type: ignore
    special_tokens=[ # type: ignore
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ],
)

    print("Training tokenizer...")
    tokenizer.train_from_iterator(
        load_corpus([c4_path, wiki_path]),
        trainer=trainer
    )

    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    tokenizer.model.save(out_dir)

    print("Tokenizer training finished.")
    print(f"Saved to {out_dir}")
