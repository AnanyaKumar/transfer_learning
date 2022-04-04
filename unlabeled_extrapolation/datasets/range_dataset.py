from torch.utils.data import Dataset
import numpy as np

class RangeDataset(Dataset):
    '''
    Takes a range over another dataset
    '''
    def __init__(self, dataset, start_idx, end_idx):
        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = end_idx
        if start_idx >= len(self.dataset):
            raise ValueError(f"start index must be less than length of dataset {len(self.dataset)}")
        if end_idx > len(self.dataset):
            raise ValueError(f"end index must be less than or equal to length of dataset {len(self.dataset)}")

    def __getitem__(self, idx):
        idx += self.start_idx
        return self.dataset[idx]

    def __len__(self):
        return self.end_idx - self.start_idx
