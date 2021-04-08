import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

class CIFAR10C(Dataset):

    def __init__(self, root, corruption, severity, transform=None):
        super().__init__()
        if corruption not in CORRUPTIONS:
            raise ValueError(f"{corruption} is not a valid corruption.")
        if not(0 <= severity <= 4):
            raise ValueError(f"Severity was {severity} but must be 0, 1, 2, 3, 4.")
        num_examples = 10000
        start_idx = num_examples * severity
        end_idx = num_examples * (severity + 1)
        self._xs = np.load(root + '/' + corruption + '.npy')[start_idx:end_idx]
        self._ys = torch.LongTensor(np.load(root + 'labels.npy'))[start_idx:end_idx]
        assert(len(self._xs) == len(self._ys) == num_examples)
        self._transform = transform

    def __getitem__(self, i):
        x, y = self._xs[i], self._ys[i]
        x = Image.fromarray(np.uint8(x))
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self._xs)

