import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CINICImNet(Dataset):

    def __init__(self, root, mode='train', transform=None):
        super().__init__()
        if mode not in ['train', 'valid', 'test']:
            raise ValueError(f'mode was {mode} but must be train, valid, test.')
        self._cinic_data = torchvision.datasets.ImageFolder(
            root + '/' + mode, transform=transform)

    def __getitem__(self, i):
        return self._cinic_data[i]

    def __len__(self) -> int:
        return len(self._cinic_data)

