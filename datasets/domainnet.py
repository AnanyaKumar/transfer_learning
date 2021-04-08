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

