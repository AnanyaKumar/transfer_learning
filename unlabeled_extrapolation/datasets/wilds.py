
from wilds import get_dataset

import torch
from torch.utils.data import Dataset
import numpy as np

class WILDS(Dataset):

    def __init__(self, dataset_name, split, root, meta_selector=None, transform=None, download=False, return_meta=False):
        # Split can be train, id_val, id_test, val, test.
        super().__init__()
        full_dataset = get_dataset(dataset=dataset_name, download=download, root_dir=root)
        dataset = full_dataset.get_subset(split, transform=None)
        self._transform = transform
        self._dataset = dataset
        self._indices = None
        self._meta_selector = None
        self._return_meta = return_meta
        if meta_selector is not None:
            self._meta_selector = tuple(meta_selector)
            super_indices = self._dataset.indices
            subset_metadata = self._dataset.dataset.metadata_array[super_indices].numpy()
            mask = np.all(subset_metadata == np.array(meta_selector), axis=-1)
            # For some reason the indices is 2d  (each index is in its own list), so need [:. 0] below.
            self._indices = np.argwhere(mask)[:, 0]


    def __getitem__(self, i):
        if self._indices is None:
            x, y, z = self._dataset[i]
        else:
            x, y, z = self._dataset[self._indices[i]]
            assert(tuple(z) == self._meta_selector)
        x = x.convert('RGB') 
        x = self._transform(x)
        if self._return_meta:
            return x, y, z
        return x, y

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._dataset)

