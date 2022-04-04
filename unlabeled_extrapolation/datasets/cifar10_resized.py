from torchvision.datasets import CIFAR10
import numpy as np

class CIFAR10Resized(CIFAR10):

    def __init__(self, root, train, train_len, seed=0, transform=None, target_transform=None, download=False):
        super().__init__(
            root=root, train=train, transform=transform, target_transform=target_transform,
            download=download)
        cifar_len = super().__len__()
        if train:
            assert train_len <= cifar_len
            rng = np.random.default_rng(seed=seed)
            self._indices = rng.choice(cifar_len, train_len)
        else:
            self._indices = np.arange(cifar_len)

    def __getitem__(self, i):
        super_index = self._indices[i]
        return super().__getitem__(super_index)

    def __len__(self) -> int:
        return len(self._indices)
