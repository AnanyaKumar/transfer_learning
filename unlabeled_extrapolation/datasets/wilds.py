
from wilds import get_dataset

from collections import defaultdict
import torch
from torch.utils.data import Dataset
import numpy as np


def get_indices_by_value(values):
    # Returns list_of_lists, where list_of_lists[i] corresponds to all indices
    # i with equal values[i].
    index_dict = defaultdict(list)
    for i, v in zip(range(len(values)), values):
        index_dict[v].append(i)
    return list(index_dict.values())

def equal_subsample(list_of_lists, rng):
    min_size = np.min([len(l) for l in list_of_lists])
    subsampled = [rng.choice(l, size=min_size, replace=False) for l in list_of_lists]
    subsampled = np.concatenate(subsampled)
    return subsampled


class WILDS(Dataset):

    def __init__(self, dataset_name, split, root, meta_selector=None, transform=None, download=True, return_meta=False,
                 subsampled_y=False, subsampled_meta=False, seed=0):
        # Split can be train, id_val, id_test, val, test.
        super().__init__()
        parent_dataset_name = dataset_name
        if 'waterbirds' in dataset_name:
            parent_dataset_name = 'waterbirds'
        full_dataset = get_dataset(dataset=parent_dataset_name, download=download, root_dir=root)
        dataset = full_dataset.get_subset(split, transform=None)
        self._dataset_name = dataset_name
        self._transform = transform
        self._dataset = dataset
        self._indices = None
        self._meta_selector = None
        self._return_meta = return_meta
        self._rng = np.random.default_rng(seed=seed)
        if meta_selector is not None:
            self._meta_selector = tuple(meta_selector)
            super_indices = self._dataset.indices
            subset_metadata = self._dataset.dataset.metadata_array[super_indices].numpy()
            mask = np.all(subset_metadata == np.array(meta_selector), axis=-1)
            # For some reason the indices is 2d  (each index is in its own list), so need [:. 0] below.
            self._indices = np.argwhere(mask)[:, 0]
        if subsampled_y or subsampled_meta:
            if meta_selector is not None or (subsampled_y and subsampled_meta):
                raise ValueError(f"subsampled_y ({subsampled_y}), subsampled_meta ({subsampled_meta}), "
                                 f"meta_selector ({meta_selector}) but two of these three must be None.")
            super_indices = self._dataset.indices
            if subsampled_y:
                subset_ys = self._dataset.dataset._y_array[super_indices].numpy()
                list_of_lists = get_indices_by_value(subset_ys)
            if subsampled_meta:
                subset_metas = self._dataset.dataset.metadata_array[super_indices].numpy()
                subset_metas = [tuple(l) for l in subset_metas]
                list_of_lists = get_indices_by_value(subset_metas)
            self._indices = equal_subsample(list_of_lists, self._rng)


    def __getitem__(self, i):
        if self._indices is None:
            x, y, z = self._dataset[i]
        else:
            x, y, z = self._dataset[self._indices[i]]
            if self._meta_selector is not None:
                assert(tuple(z) == self._meta_selector)
        x = x.convert('RGB') 
        x = self._transform(x)
        if 'waterbirds' in self._dataset_name and 'background' in self._dataset_name:
            y = z[0]
        if self._return_meta:
            return x, y, z
        return x, y

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self._dataset)

