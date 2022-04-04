
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

import torch
from torch.utils.data import Dataset
import numpy as np

# Mapping from continent names to region codes:
# {'Asia': 0, 'Europe': 1, 'Africa': 2, 'Americas': 3, 'Oceania': 4, 'Other': 5}

class Fmow(Dataset):

    def __init__(self, root, regions, split='train', transform=None):
        # Split can be train, id_val, id_test, val, test.
        # regions is a lister of integers between 0 and 4 denoting the regions, don't use 'Other'.
        # For fmow root is the directory that contains fmow (and other wilds datasets).
        super().__init__()
        super_dataset = get_dataset(dataset='fmow', download=False, root_dir=root)
        self._subset = super_dataset.get_subset(split, transform=transform)
        self._regions = regions
        if 'all' not in self._regions:
            super_indices = self._subset.indices
            subset_metadata = self._subset.dataset.metadata_array[super_indices].numpy()
            self._indices = np.argwhere([(a in regions) for a in subset_metadata[:, 0]])[:,0]
        
    def __getitem__(self, i):
        if 'all' not in self._regions:
            x, y, _ = self._subset[self._indices[i]]
        else:
            x, y, _ = self._subset[i]
        return x, y

    def __len__(self) -> int:
        if 'all' not in self._regions:
            return len(self._indices)
        else:
            return len(self._subset)

