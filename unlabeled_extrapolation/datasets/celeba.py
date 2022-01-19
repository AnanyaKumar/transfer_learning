from functools import partial
import numpy as np
import os
import pandas
from pathlib import Path
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
# from innout.datasets import RangeDataset


JUICE_CELEBA_ROOT = '/juice/scr/ananya/celeba'

SPLIT_TO_IDX = {
    'train': 0,
    'val': 1,
    'test': 2,
}


attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                   'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                   'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                   'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                   'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                   'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                   'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def resize_transform():
    return transforms.Compose([
        transforms.Resize((64,64), interpolation=2),
        transforms.ToTensor()])


def tensor_transform():
    return transforms.ToTensor()


def invert_list(l):
    """Return a list that inverts the mapping of l."""
    dict = {}
    for i, x in zip(range(len(l)), l):
        dict[x] = i
    return dict


attr_name_to_idx = invert_list(attribute_names)


domain_selectors = {
    'wearing_hat': lambda a: a[attr_name_to_idx['Wearing_Hat']],
    'not_wearing_hat': lambda a: 1 - a[attr_name_to_idx['Wearing_Hat']],
}


def select_indices(attributes, attribute_selector):
    """Return a list of indices i where attribute_selector(attributes[i]) is True."""
    mask = np.apply_along_axis(attribute_selector, axis=1, arr=attributes)
    return np.argwhere(mask)


def sample_indices(indices, target, pos_fraction, num_points, rng):
    # First split into two groups, positive and negative
    num_pos = round(num_points * pos_fraction)
    num_neg = num_points - num_pos
    pos_indices = indices[target == 1]
    neg_indices = indices[target == 0]
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)
    indices = np.concatenate([pos_indices[:num_pos], neg_indices[:num_neg]])
    rng.shuffle(indices)
    return indices


# TODO: call this file and produce a pickle.
def celeba_to_pickle(save_file, celeba_root=JUICE_CELEBA_ROOT):
    """Save the CelebA dataset into a pickle file."""
    fn = partial(os.path.join, celeba_root)
    celeba_splits = pandas.read_csv(
            fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
#     mask = (celeba_splits[1] == 0)  # Use celeba training set.
    filenames = celeba_splits.index.values
    images = []
    counter = 0
    for filename in filenames:
        if counter % 1000 == 0:
            print(counter)
        counter += 1
        img = Image.open(os.path.join(celeba_root, "img_align_celeba", filename))
        images.append(np.array(img))
    file = open(save_file, "wb")
    pickle.dump(images, file)
    file.close()

    
    
class CelebA(Dataset):
    def __init__(self, target_attribute, split, supergroup_n=None, num_subgroups=None, subgroup_idx=None,
                 seed=None, transform=None, celeba_root=JUICE_CELEBA_ROOT, pickle_file_path=None,
                 verbose=False):
        if split not in SPLIT_TO_IDX:
            raise ValueError(f'Split must be in {SPLIT_TO_IDX} but was {split}')
        self._rng = np.random.RandomState(seed)
        self._celeba_root = celeba_root
        fn = partial(os.path.join, self._celeba_root)
        self._split = split
        celeba_splits = pandas.read_csv(
            fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        mask = (celeba_splits[1] == SPLIT_TO_IDX[split])  # Use appropriate split.
        self._pickle_file_path = pickle_file_path
        if pickle_file_path is None:
            self._filenames = celeba_splits[mask].index.values
        else:
            self._images = pickle.load(open(pickle_file_path, "rb"))
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        self._attr = torch.as_tensor(attr[mask].values)
        self._attr = (self._attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self._attr_names = list(attr.columns)
        assert self._attr_names == attribute_names
        self._transform = transform

        # Convert meta-attributes and target attribute to indices if they are strings.
        def convert_attr_desc_to_index(attr_desc):
            if type(attr_desc) == str:
                if not attr_desc in attribute_names:
                    raise ValueError('Attributes specified must be a valid string or integer between 0 and'
                                     ' 39 inclusive but was {}'.format(attr_desc))
                return attr_name_to_idx[attr_desc]
            else:
                if not(0 <= attr_desc <= 39):
                        raise ValueError('Attributes specified must be a valid string or integer between 0 and'
                                         ' 39 inclusive but was {}'.format(attr_desc))
                return attr_desc
        self._target_attribute = convert_attr_desc_to_index(target_attribute)
        
        # For now just return all the indices
        self._indices = np.arange(len(self._attr))
        
        if split == 'train':
            # Populate default values, which is to use the entire training set, one subgroup, and pick the first subgroup.
            if subgroup_idx is None:
                subgroup_idx = 0
            if num_subgroups is None:
                num_subgroups = 1
            if supergroup_n is None:
                supergroup_n = len(self._indices)
            # Select a random supergroup_n number of indices.
            self._indices = self._rng.permutation(self._indices)[:supergroup_n]
            if not(0 <= subgroup_idx < num_subgroups):
                raise ValueError(f'{subgroup_idx} should be in between 0 and {num_subgroups-1} inclusive')
            # Get the indices for this subgroup.
            lower_i = int(np.floor(float(subgroup_idx) / num_subgroups * supergroup_n))
            upper_i = int(np.floor(float(subgroup_idx+1) / num_subgroups * supergroup_n))
            if verbose:
                print('celeba index splits: ', lower_i, upper_i)
            if subgroup_idx == num_subgroups - 1 and upper_i != supergroup_n:
                print(f'floating point issue, got {upper_i} for upper index with subgroup idx {subgroup_idx}, converting to {supergroup_n}')
                upper_i = supergroup_n
            self._indices = sorted(self._indices[lower_i:upper_i])
            if verbose:
                print('first 20 indices: ', self._indices[:20])
        else:
            # Use the entire set of indices for val and test set.
            pass
        
    def __getitem__(self, i):
        index = self._indices[i]
        if self._pickle_file_path is None:
            x = Image.open(os.path.join(self._celeba_root, "img_align_celeba", self._filenames[index]))
        else:
            x = Image.fromarray(self._images[index])
        if self._transform is not None:
            x = self._transform(x)
        y = self._attr[index][self._target_attribute]
        return x, y

    def __len__(self) -> int:
        return len(self._indices)
