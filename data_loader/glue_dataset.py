import torch
from torch.utils.data import Dataset


class GlueDataset(Dataset):
    """Dataset sederhana untuk GLUE yang sudah ditokenisasi."""

    def __init__(self, data_path: str, task_name: str, max_length: int = 512):
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
