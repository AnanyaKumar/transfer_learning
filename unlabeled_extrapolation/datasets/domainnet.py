from torch.utils.data import Dataset
import os
from PIL import Image


VALID_DOMAINS = [
    'clipart',
    'infograph',
    'painting',
    'quickdraw',
    'real',
    'sketch'
]

SENTRY_DOMAINS = [
    'clipart',
    'painting',
    'real',
    'sketch'
]

NUM_CLASSES_DICT = {
    'full': 345,
    'sentry': 40
}

VALID_SPLITS = ['train', 'test']

VALID_VERSIONS = ['full', 'sentry']

ROOT = '/u/scr/nlp/domainnet'
SENTRY_SPLITS_ROOT = '/u/scr/nlp/domainnet/SENTRY_splits'


def load_dataset(root, domains, split, version):
    if len(domains) == 1 and domains[0] == 'all':
        if version == 'sentry':
            domains = SENTRY_DOMAINS
        else:
            domains = VALID_DOMAINS

    data = []
    for domain in domains:
        if version == 'sentry':
            if os.path.isdir(root + '/SENTRY_splits'):
                idx_file = os.path.join(root, f'SENTRY_splits/{domain}_{split}_mini.txt')
            else:
                idx_file = os.path.join(SENTRY_SPLITS_ROOT, f'{domain}_{split}_mini.txt')
        else:
            if os.path.isfile(root + f'/{domain}_{split}.txt'):
                idx_file = os.path.join(root, f'{domain}_{split}.txt')
            else:
                idx_file = os.path.join(ROOT, f'{domain}_{split}.txt')
        with open(idx_file, 'r') as f:
            data += [line.split() for line in f]
    return data


class DomainNet(Dataset):
    def __init__(self, domain, split='train', root=ROOT,
                 transform=None, unlabeled=False, verbose=True,
                 version='sentry'):
        super().__init__()

        if version not in VALID_VERSIONS:
            raise ValueError(f'dataset version must be in {VALID_VERSIONS} but was {version}')
        domain_list = domain.split(',')
        for domain in domain_list:
            if domain != 'all' and version == 'full' and domain not in VALID_DOMAINS:
                raise ValueError(f'domain must be in {VALID_DOMAINS} but was {domain}')
            if domain != 'all' and version == 'sentry' and domain not in SENTRY_DOMAINS:
                raise ValueError(f'domain must be in {SENTRY_DOMAINS} but was {domain}')
        if split not in VALID_SPLITS:
            raise ValueError(f'split must be in {VALID_SPLITS} but was {split}')
        self._root_data_dir = root
        self._domain_list = domain_list
        self._split = split
        self._transform = transform
        self._version = version

        self._unlabeled = unlabeled
        self.data = load_dataset(root, domain_list, split, version)
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.228, 0.224, 0.225]
        if verbose:
            print(f'Loaded domains {", ".join(domain_list)}, split is {split}')
            print(f'Total number of images: {len(self.data)}')
            print(f'Total number of classes: {self.get_num_classes()}')

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

    def get_num_classes(self):
        return len(set([self.data[idx][1] for idx in range(len(self.data))]))
