# CIFAR-10.1 dataset, by Rebecca Roelofs and Ludwig Schmidt
# Copying the utils from there for convenience.

import os
import pathlib
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np

# Map from ImageNet renditions indices to ImageNet indices.
r_indices = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]

class ImageNet(Dataset):

    def __init__(self, root, split='train', num_examples=None, transform=None, seed=0):
        super().__init__()
        self.data = datasets.ImageFolder(root=root + '/' + split, transform=None)
        self._split = split
        self._num_examples = num_examples
        self._transform = transform
        if self._num_examples is not None:
            if self._num_examples > len(self.data):
                raise ValueError('num_examples can be at most the dataset size {len(self.data)}')
            rng = np.random.RandomState(seed=seed)
            self._data_indices = rng.permutation(len(self.data))[:num_examples]
        
    def __getitem__(self, i):
        if self._num_examples is not None:
            i = self._data_indices[i]
        x, y = self.data[i]
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        if self._split == 'renditions' or self._split == 'imagenet-r':
            y = r_indices[y]
        return x, y

    def __len__(self) -> int:
        return len(self.data) if self._num_examples is None else self._num_examples
