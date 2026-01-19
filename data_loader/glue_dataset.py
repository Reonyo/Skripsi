import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path


class GlueDataset(Dataset):
    """Dataset sederhana untuk GLUE yang sudah ditokenisasi."""

    def __init__(self, data_path: str, task_name: str, max_length: int = 512):
        # Check if file exists, print helpful error if not
        if not Path(data_path).exists():
            print(f"\n[ERROR] File not found: {data_path}")
            print(f"  Task: {task_name}")
            print(f"  Current working directory: {Path.cwd()}")
            print(f"  Absolute path: {Path(data_path).absolute()}")
            
            # Check if parent directory exists
            parent_dir = Path(data_path).parent
            if parent_dir.exists():
                print(f"  Files in {parent_dir}:")
                for f in parent_dir.iterdir():
                    print(f"    - {f.name}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Handle special case for MNLI which has matched and mismatched splits
        if task_name.upper() == "MNLI" and "validation" in data_path:
            # Load both matched and mismatched for validation
            matched_path = data_path.replace("validation.pt", "validation_matched.pt")
            mismatched_path = data_path.replace("validation.pt", "validation_mismatched.pt")
            
            if Path(matched_path).exists() and Path(mismatched_path).exists():
                data_matched = torch.load(matched_path)
                data_mismatched = torch.load(mismatched_path)
                
                # Concatenate matched and mismatched data
                self.input_ids = torch.cat([
                    data_matched["input_ids"][:, :max_length],
                    data_mismatched["input_ids"][:, :max_length]
                ], dim=0)
                self.attention_mask = torch.cat([
                    data_matched["attention_mask"][:, :max_length],
                    data_mismatched["attention_mask"][:, :max_length]
                ], dim=0)
                self.labels = torch.cat([
                    data_matched["labels"],
                    data_mismatched["labels"]
                ], dim=0)
            else:
                raise FileNotFoundError(f"MNLI validation splits not found at {matched_path} or {mismatched_path}")
        else:
            # Regular loading for other tasks
            data = torch.load(data_path)
            
            # Expect keys: input_ids (LongTensor), attention_mask (LongTensor), labels (Long/Float)
            self.input_ids = data["input_ids"][:, :max_length]
            self.attention_mask = data["attention_mask"][:, :max_length]
            self.labels = data["labels"]

        # Derive vocab size from token ids
        self.vocab_size = int(self.input_ids.max().item()) + 1

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
