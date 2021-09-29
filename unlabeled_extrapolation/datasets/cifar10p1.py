# CIFAR-10.1 dataset, by Rebecca Roelofs and Ludwig Schmidt
# Copying the utils from there for convenience.

import os
import pathlib
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np


def load_new_test_data(root, version_string=''):
    data_path = root
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021
    return imagedata, labels


class CIFAR10p1(Dataset):

    def __init__(self, root, split='train', version='v6', transform=None):
        super().__init__()
        imagedata, labels = load_new_test_data(root=root, version_string=version)
        self._transform = transform
        self._imagedata = imagedata
        self._labels = labels
        
    def __getitem__(self, i):
        x = Image.fromarray(np.uint8(self._imagedata[i]))
        if self._transform is not None:
            x = self._transform(x)
        y = self._labels[i]
        return x, y

    def __len__(self) -> int:
        return len(self._imagedata)
