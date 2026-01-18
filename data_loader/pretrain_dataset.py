import os
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    Dataset untuk pretraining (MLM + RTD).
    Data diasumsikan sudah ditokenisasi & dipadding.
    """

    def __init__(self, data_dir: str, max_length: int):
        self.data_dir = data_dir
        self.max_length = max_length

        # Load data
        self.input_ids = torch.load(os.path.join(data_dir, "input_ids.pt"))
        self.attention_mask = torch.load(os.path.join(data_dir, "attention_mask.pt"))

        assert len(self.input_ids) == len(self.attention_mask)

        self.vocab_size = int(self.input_ids.max()) + 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
