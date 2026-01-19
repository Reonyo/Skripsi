import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import ast

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


def parse_glue_line(line: str, task_name: str) -> Tuple[str, int | float] | None:
    """Parse GLUE task line (Python dict format) and return (text, label) based on task format."""
    try:
        obj = ast.literal_eval(line.strip())
    except:
        return None
    
    try:
        if task_name.upper() == "SST2":
            sentence = obj.get("sentence", "")
            label = obj.get("label")
            if sentence and label is not None:
                return sentence, int(label)
        
        elif task_name.upper() == "COLA":
            sentence = obj.get("sentence", "")
            label = obj.get("label")
            if sentence and label is not None:
                return sentence, int(label)
        
        elif task_name.upper() == "MRPC":
            text1 = obj.get("sentence1", "")
            text2 = obj.get("sentence2", "")
            label = obj.get("label")
            if text1 and text2 and label is not None:
                return f"{text1} [SEP] {text2}", int(label)
        
        elif task_name.upper() == "QQP":
            text1 = obj.get("question1", "")
            text2 = obj.get("question2", "")
            label = obj.get("label")
            if text1 and text2 and label is not None:
                return f"{text1} [SEP] {text2}", int(label)
        
        elif task_name.upper() == "QNLI":
            text1 = obj.get("question", "")
            text2 = obj.get("sentence", "")
            label = obj.get("label")
            if text1 and text2 and label is not None:
                return f"{text1} [SEP] {text2}", int(label)
        
        elif task_name.upper() == "RTE":
            text1 = obj.get("sentence1", "")
            text2 = obj.get("sentence2", "")
            label = obj.get("label")
            if text1 and text2 and label is not None:
                return f"{text1} [SEP] {text2}", int(label)
        
        elif task_name.upper() == "MNLI":
            text1 = obj.get("premise", "")
            text2 = obj.get("hypothesis", "")
            label = obj.get("label")
            if text1 and text2 and label is not None:
                # Label bisa int atau string
                if isinstance(label, int):
                    label_val = label
                elif isinstance(label, str):
                    label_map = {"contradiction": 0, "neutral": 1, "entailment": 2}
                    label_val = label_map.get(label.lower())
                else:
                    label_val = None
                
                if label_val is not None:
                    return f"{text1} [SEP] {text2}", int(label_val)
        
        elif task_name.upper() == "STSB":
            text1 = obj.get("sentence1", "")
            text2 = obj.get("sentence2", "")
            score = obj.get("label")  # STSB uses 'label' key, not 'score'
            if text1 and text2 and score is not None:
                return f"{text1} [SEP] {text2}", float(score)
    
    except (KeyError, ValueError, TypeError):
        pass
    
    return None


def preprocess_glue_task(
    tokenizer: Tokenizer,
    task_name: str,
    raw_dir: str,
    out_dir: str,
    max_length: int,
):
    """Preprocess single GLUE task with labels."""
    task_raw_dir = os.path.join(raw_dir, task_name)
    task_out_dir = os.path.join(out_dir, task_name)
    ensure_dir(task_out_dir)
    
    # Handle special case for MNLI which has matched/mismatched variants
    if task_name.upper() == "MNLI":
        splits_to_process = [
            "train", 
            "validation_matched", "validation_mismatched",
            "test_matched", "test_mismatched"
        ]
    else:
        splits_to_process = ["train", "validation", "test"]
    
    for split in splits_to_process:
        split_file = os.path.join(task_raw_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            print(f"  Warning: {split_file} not found, skipping {split}")
            continue
        
        input_ids = []
        attention_masks = []
        labels = []
        skipped = 0
        
        print(f"  Processing {split}...")
        with open(split_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  {split}"):
                line = line.strip()
                if not line:
                    continue
                
                result = parse_glue_line(line, task_name)
                if result is None:
                    skipped += 1
                    continue
                
                text, label = result
                try:
                    enc = tokenizer.encode(text)
                    ids = enc.ids[:max_length]
                    mask = [1] * len(ids)
                    
                    # padding
                    pad_len = max_length - len(ids)
                    if pad_len > 0:
                        ids += [0] * pad_len  # Use 0 (PAD token id)
                        mask += [0] * pad_len
                    
                    input_ids.append(ids)
                    attention_masks.append(mask)
                    labels.append(label)
                except Exception as e:
                    skipped += 1
                    if skipped <= 3:
                        print(f"    Warning: Failed to encode line: {e}")
        
        if len(input_ids) > 0:
            label_dtype = torch.float if isinstance(labels[0], float) else torch.long
            data = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=label_dtype),
            }
            out_path = os.path.join(task_out_dir, f"{split}.pt")
            torch.save(data, out_path)
            print(f"  Saved {out_path} ({len(input_ids)} samples, skipped {skipped})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--max_length", type=int, default=256)
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

        print("Preprocessing finished")
        print(f"Saved to {out_base}")

    else:  # finetune (GLUE tasks)
        raw_dir = "data/glue/raw"
        out_dir = "data/glue/tokenized"
        
        glue_tasks = [
            "SST2", "CoLA", "MRPC", "QQP", "STSB", "MNLI", "QNLI", "RTE"
        ]
        
        for task_name in glue_tasks:
            task_raw_dir = os.path.join(raw_dir, task_name)
            if not os.path.exists(task_raw_dir):
                print(f"Warning: {task_raw_dir} not found, skipping {task_name}")
                continue
            
            print(f"\nProcessing GLUE task: {task_name}")
            preprocess_glue_task(
                tokenizer=tokenizer,
                task_name=task_name,
                raw_dir=raw_dir,
                out_dir=out_dir,
                max_length=args.max_length,
            )
        
        print("\nGLUE preprocessing finished")
        print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
