import numpy as np
from torch.utils.data import Dataset

class LoopingDataset(Dataset):
    """
    Dataset class to handle indices going out of bounds when training
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= len(self.dataset):
            index = np.random.choice(len(self.dataset))
        item, label = self.dataset[index]
        return item, label