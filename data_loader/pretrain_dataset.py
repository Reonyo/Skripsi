import os
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    Dataset untuk pretraining (MLM + RTD).
    Data diasumsikan sudah ditokenisasi & dipadding dalam format .pt file.
    
    File .pt harus berisi dictionary dengan keys: 'input_ids' dan 'attention_mask'
    
    Args:
        data_path (str): Path ke file .pt yang berisi {input_ids, attention_mask}
        max_length (int): Maximum sequence length (tidak digunakan saat ini, untuk kompatibilitas)
    """

    def __init__(self, data_path: str, max_length: int):
        self.data_path = data_path
        self.max_length = max_length

        # Load data dari file .pt
        data = torch.load(data_path)
        
        # Extract input_ids dan attention_mask dari dictionary
        if isinstance(data, dict):
            self.input_ids = data["input_ids"]
            self.attention_mask = data["attention_mask"]
        else:
            raise ValueError(f"Expected dict in {data_path}, got {type(data)}")

        assert len(self.input_ids) == len(self.attention_mask), \
            f"Mismatch: input_ids ({len(self.input_ids)}) vs attention_mask ({len(self.attention_mask)})"

        self.vocab_size = int(self.input_ids.max()) + 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
