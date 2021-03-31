import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.datasets import DatasetFolder


def loader(path):
    img = Image.open(path)
    rgb_img = img.convert('RGB')
    return rgb_img


class ImNetnC10(Dataset):

    def __init__(self, root, transform=None):
        super().__init__()
        self._dataset = DatasetFolder(
                root=root, loader=loader, extensions=('jpeg',), transform=transform)

    def __getitem__(self, i):
        return self._dataset[i]

    def __len__(self) -> int:
        return len(self._dataset)

