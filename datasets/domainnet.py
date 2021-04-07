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

VALID_SPLITS = ['train', 'val']

def load_dataset(data_dir, domain, split):
    idx_file = os.path.join(data_dir, f'{domain}_{split}.txt')
    with open(idx_file, 'r') as f:
        data = [line.split() for line in f]
    return data

class DomainNet(Dataset):
    def __init__(self, domain, split='train', root='/u/scr/nlp/domainnet',
                 transform=None):
        super().__init__()
        if domain not in VALID_DOMAINS:
            raise ValueError(f'domain must be in {VALID_DOMAINS} but was {domain}')
        if split not in VALID_SPLITS:
            raise ValueError(f'split must be in {VALID_SPLITS} but was {split}')
        self._root_data_dir = root
        self._domain = domain
        self._split = split
        self._transform = transform
        self.data = load_dataset(root, domain, split)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, y = self.data[idx]
        x = Image.open(os.path.join(self._root_data_dir, path))
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        return x, int(y)
