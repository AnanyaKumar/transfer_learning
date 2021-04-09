from torch.utils.data import Dataset
import os
from PIL import Image

from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from pl_bolts.datasets import ConcatDataset
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg('torchvision')

VALID_DOMAINS = [
    'clipart',
    'infograph',
    'painting',
    'quickdraw',
    'real',
    'sketch'
]

VALID_SPLITS = ['train', 'test']

ROOT = '/u/scr/nlp/domainnet'

def load_dataset(data_dir, domains, split):
    if len(domains) == 1 and domains[0] == 'all':
        domains = VALID_DOMAINS

    data = []
    for domain in domains:
        idx_file = os.path.join(data_dir, f'{domain}_{split}.txt')
        with open(idx_file, 'r') as f:
            data += [line.split() for line in f]
    return data

class DomainNet(Dataset):
    def __init__(self, domain, split='train', root=ROOT,
                 transform=None, unlabeled=False):
        super().__init__()

        domain_list = domain.split(',')
        for domain in domain_list:
            if domain != 'all' and domain not in VALID_DOMAINS:
                raise ValueError(f'domain must be in {VALID_DOMAINS} but was {domain}')
        if split not in VALID_SPLITS:
            raise ValueError(f'split must be in {VALID_SPLITS} but was {split}')
        self._root_data_dir = root
        self._domain = domain
        self._split = split
        self._transform = transform
        if transform is None:
            self._transform = transform_lib.Compose([
                transform_lib.Resize(224),
                transform_lib.ToTensor(),
                transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self._unlabeled = unlabeled
        self.data = load_dataset(root, domain_list, split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, y = self.data[idx]
        x = Image.open(os.path.join(self._root_data_dir, path))
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        # if self._unlabeled:
        #     return x, -1
        # else:
        return x, int(y)


class DomainNetDataModule(LightningDataModule):  # pragma: no cover
    """
    Example::

        from pl_bolts.datamodules import STL10DataModule

        dm = DomainNetDataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = 'domainnet'

    def __init__(
        self,
        data_dir: Optional[str] = ROOT,
        unlabeled_val_split: int = 1000,
        train_val_split: int = 500,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_domain: str = 'all',
        test_domain: str = 'all',
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            unlabeled_val_split: how many images from the unlabeled training split to use for validation
            train_val_split: how many images from the labeled training split to use for validation
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        self.dims = (3, 224, 224)
        self.data_dir = data_dir
        self.unlabeled_val_split = unlabeled_val_split
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_domain = train_domain
        self.test_domain = test_domain
        # a default value
        self.num_unlabeled_samples = \
            len(DomainNet(domain=self.train_domain, split='train', transform=transforms, unlabeled=True, root=self.data_dir)) - unlabeled_val_split

    @property
    def num_classes(self) -> int:
        return 345

    def prepare_data(self) -> None:
        """
        Downloads the unlabeled, train and test split
        """
        pass

    def train_dataloader(self) -> DataLoader:
        """
        Loads the 'unlabeled' split minus a portion set aside for validation via `unlabeled_val_split`.
        """
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = DomainNet(domain=self.train_domain, split='train', transform=transforms, unlabeled=True, root=self.data_dir)
        train_length = len(dataset)

        dataset_train, _ = random_split(
            dataset, [train_length - self.unlabeled_val_split, self.unlabeled_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def train_dataloader_mixed(self) -> DataLoader:
        return self.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        """
        Loads a portion of the 'unlabeled' training data set aside for validation
        The val dataset = (unlabeled - train_val_split)

        Args:

            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = DomainNet(domain=self.train_domain, split='train', transform=transforms, unlabeled=True, root=self.data_dir)
        train_length = len(dataset)

        _, dataset_val = random_split(
            dataset, [train_length - self.unlabeled_val_split, self.unlabeled_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader_mixed(self) -> DataLoader:
        """
        For returning a mix between unlabeled train and unlabeled val
        """
        return self.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        """
        Loads the test split

        Args:
            batch_size: the batch size
            transforms: the transforms
        """

        transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = DomainNet(domain=self.test_domain, split='test', transform=transforms, unlabeled=False, root=self.data_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def train_dataloader_labeled(self) -> DataLoader:
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = DomainNet(domain=self.train_domain, split='train', transform=transforms, unlabeled=False, root=self.data_dir)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.train_val_split, self.train_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader_labeled(self) -> DataLoader:
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
        dataset = DomainNet(domain=self.train_domain, split='train', transform=transforms, unlabeled=False, root=self.data_dir)
        labeled_length = len(dataset)
        _, labeled_val = random_split(
            dataset, [labeled_length - self.train_val_split, self.train_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        loader = DataLoader(
            labeled_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def _default_transforms(self) -> Callable:
        return None

def verify_class_mapping(root='/u/scr/nlp/domainnet'):
    mapping = {}
    with open(os.path.join(root, VALID_DOMAINS[0] + '_train.txt'), 'r') as f:
        for line in f:
            class_name = line.split('/')[1]
            class_idx = line.split()[-1]
            mapping[class_name] = class_idx
    for domain in VALID_DOMAINS:
        train_file = os.path.join(root, f'{domain}_train.txt')
        test_file = os.path.join(root, f'{domain}_test.txt')
        with open(train_file, 'r') as f:
            for line in f:
                class_name = line.split('/')[1]
                class_idx = line.split()[-1]
                if class_name not in mapping or mapping[class_name] != class_idx:
                    raise ValueError(f'existing map is {class_name} --> {mapping[class_name]} but encountered {class_idx}')
    print('All mappings are valid')

if __name__ == '__main__':
    verify_class_mapping()
