import sys
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26

BREEDS_SPLITS_TO_FUNC = {
    'entity13': make_entity13,
    'entity30': make_entity30,
    'living17': make_living17,
    'nonliving26': make_nonliving26,
}

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

SPLITS = ['train', 'val']

MIN_NUM_TRAIN_PER_CLASS = 100

NUM_VAL_PER_CLASS = 50

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def get_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def get_image_paths_breeds_class(class_dir, breeds_class):
    image_paths_breeds_class = []
    for root, _, fnames in sorted(os.walk(class_dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions=IMG_EXTENSIONS):
                path = os.path.join(root, fname)
                image_paths_breeds_class.append((path, breeds_class))
    return image_paths_breeds_class

def get_image_paths_by_class(data_dir, idx_to_class_id, subclasses, split):
    image_paths_and_class = []
    for idx in range(len(subclasses)):
        for subclass in subclasses[idx]:
            subclass_image_paths_breeds_class = get_image_paths_breeds_class(
                data_dir + '/' + idx_to_class_id[subclass] + '/', idx)
            image_paths_and_class.extend(subclass_image_paths_breeds_class)
            # print(data_dir + '/' + idx_to_class_id[subclass] + '/', len(subclass_image_names))
            if split == 'train':
                assert(len(subclass_image_paths_breeds_class) >= MIN_NUM_TRAIN_PER_CLASS)
            else:
                assert(len(subclass_image_paths_breeds_class) == NUM_VAL_PER_CLASS)
    return image_paths_and_class

class Breeds(Dataset):
    def __init__(self, root, breeds_name,
                 info_dir='/juice/scr/ananya/cifar_experiments/BREEDS-Benchmarks/imagenet_class_hierarchy/modified',
                 source=True, target=False, split='train', transform=None):
        super().__init__()
        if breeds_name not in BREEDS_SPLITS_TO_FUNC.keys():
            raise ValueError(f'breeds_name must be in {BREEDS_SPLITS_TO_FUNC.keys()} but was {breeds_name}')
        if split not in SPLITS:
            raise ValueError(f'split must be in {SPLITS} but was {split}')
        if not source and not target:
            raise ValueError('At least one of "source" and "target" must be True!')

        self._breeds_name = breeds_name
        self._source = source
        self._split = split
        self._transform = transform
        if os.path.isdir(root + '/BREEDS-Benchmarks'):
            self._info_dir = root + '/BREEDS-Benchmarks/imagenet_class_hierarchy/modified'
        else:
            self._info_dir = info_dir
        self._data_dir = root + '/' + split
        self._idx_to_class_id, self._class_to_idx = get_classes(self._data_dir)
        breeds_func = BREEDS_SPLITS_TO_FUNC[breeds_name]
        self._superclasses, self._subclass_split, self._label_map = breeds_func(self._info_dir, split="rand")
        self._subclasses = []

        if source:
            self._subclasses.extend(self._subclass_split[0])
        if target:
            self._subclasses.extend(self._subclass_split[1])

        self._image_paths_by_class = get_image_paths_by_class(
            self._data_dir, self._idx_to_class_id, self._subclasses, split)

        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.228, 0.224, 0.225]

    def __getitem__(self, i):
        path, y = self._image_paths_by_class[i]
        x = Image.open(path)
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self._image_paths_by_class)

    def get_num_classes(self):
        return len(self._idx_to_class_id)
