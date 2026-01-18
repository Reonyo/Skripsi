import os
import argparse
from datasets import load_dataset, Dataset

GLUE_TASKS = [
    "sst2", "mnli", "qqp", "mrpc",
    "qnli", "rte", "cola", "stsb"
]

SPLITS = ["train", "validation", "test"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_split(ds: Dataset, path: str, overwrite: bool):
    if os.path.exists(path) and not overwrite:
        print(f"[SKIP] {path} already exists")
        return

    with open(path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(str(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    if args.debug:
        args.max_samples = 100

    base_dir = "data/glue/raw"

    for task in GLUE_TASKS:
        print(f"Downloading {task.upper()}...")
        task_dir = os.path.join(base_dir, task.upper())
        ensure_dir(task_dir)

        dataset = load_dataset("glue", task)

        for split in SPLITS:
            if split not in dataset:
                continue

            ds: Dataset = dataset[split] # type: ignore

            if args.max_samples is not None:
                ds = ds.select(range(min(args.max_samples, ds.num_rows)))

            out_path = os.path.join(task_dir, f"{split}.txt")
            save_split(ds, out_path, args.overwrite)

    print("GLUE data ready.")
