import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2

import os
from PIL import Image

from . import breeds, domainnet
VALID_BREEDS_DOMAINS = breeds.BREEDS_SPLITS_TO_FUNC.keys()
VALID_DOMAINNET_DOMAINS = domainnet.SENTRY_DOMAINS

#########################
#### DATA AUG STUFF #####
#########################

def get_transforms(transform_scheme):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if transform_scheme == 'imagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif transform_scheme == 'simclr':
        # adapted from
        # https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/transforms.py
        size = 224
        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=kernel_size, p=0.5),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError('Transformation scheme not supported')
    return transform

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    # https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/transforms.py
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

#####################
### DATASET STUFF ###
#####################

def validate_dataset(dataset_name, source, target):
    if dataset_name == 'breeds':
        if source not in VALID_BREEDS_DOMAINS:
            raise ValueError(f'Valid Breeds domains are {VALID_BREEDS_DOMAINS} but received '
                             f'source {source}.')
        if target is not None and source != target:
            raise ValueError(f'Must use the same task for source and target: tried to set '
                             f'source {source} and target {target}.')
        num_classes = {
            'entity30': 30,
            'living17': 17
        }[source]
    elif dataset_name == 'domainnet':
        if (source not in VALID_DOMAINNET_DOMAINS) or (target not in VALID_DOMAINNET_DOMAINS):
            raise ValueError(f'Valid DomainNet domains are {VALID_DOMAINNET_DOMAINS} but '
                             f'received source {source} and target {target}.')
        if source == target:
            raise ValueError(f'Should not use the same domain as source and target: {source}.')
        num_classes = domainnet.NUM_CLASSES_DICT['sentry']
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}.')
    return num_classes

class DomainClassificationDataset(Dataset):
    def __init__(self, ds1, ds2, data_attr_name, dataset_name, transform=None,
                 data_path=None):
        super().__init__()
        ds1_samples = getattr(ds1, data_attr_name)
        ds2_samples = getattr(ds2, data_attr_name)
        self.samples = [(item[0], 0) for item in ds1_samples] + \
            [(item[0], 1) for item in ds2_samples]
        self._transform = transform
        self._dataset_name = dataset_name
        self._data_path = data_path # only necessary for domainnet

    def __getitem__(self, i):
        if self._dataset_name == 'breeds':
            path, y = self.samples[i]
            x = Image.open(path)
        elif self._dataset_name == 'domainnet':
            path, y = self.samples[i]
            x = Image.open(os.path.join(self._data_path, path))
            y = int(y)
        else:
            raise NotImplementedError('Only breeds and domainnet are supported.')
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def __len__(self):
        return len(self.samples)

def filter_to_single_class(dataset, class_to_use, data_attr_name):
    setattr(dataset, data_attr_name,
            list(filter(lambda item: int(item[1]) == class_to_use, getattr(dataset, data_attr_name))))

def get_class_datasets(dataset_name, domain_name, class_1, class_2, transform,
                       data_path, use_source):
    if dataset_name == 'breeds':
        data_attr = '_image_paths_by_class'
        train_1 = breeds.Breeds(data_path, domain_name, source=use_source,
                                target=(not use_source), split='train')
        test_1 = breeds.Breeds(data_path, domain_name, source=use_source,
                               target=(not use_source), split='val')
        train_2 = breeds.Breeds(data_path, domain_name, source=use_source,
                                target=(not use_source), split='train')
        test_2 = breeds.Breeds(data_path, domain_name, source=use_source,
                               target=(not use_source), split='val')
    elif dataset_name == 'domainnet':
        data_attr = 'data'
        train_1 = domainnet.DomainNet(domain_name, split='train', root=data_path)
        test_1 = domainnet.DomainNet(domain_name, split='test', root=data_path)
        train_2 = domainnet.DomainNet(domain_name, split='train', root=data_path)
        test_2 = domainnet.DomainNet(domain_name, split='test', root=data_path)
    else:
        raise ValueError('Only supports Breeds and domainnet currently')
    filter_to_single_class(train_1, class_1, data_attr)
    filter_to_single_class(test_1, class_1, data_attr)
    filter_to_single_class(train_2, class_2, data_attr)
    filter_to_single_class(test_2, class_2, data_attr)
    train_ds = DomainClassificationDataset(train_1, train_2, data_attr, dataset_name,
                                           transform=transform, data_path=data_path)
    test_ds = DomainClassificationDataset(test_1, test_2, data_attr, dataset_name,
                                          transform=transform, data_path=data_path)
    return train_ds, test_ds

def get_domain_datasets(dataset_name, source, target, data_path, class_idx, transform):
    if dataset_name == 'breeds':
        data_attr = '_image_paths_by_class'
        source_train = breeds.Breeds(data_path, source, source=True, target=False, split='train')
        target_train = breeds.Breeds(data_path, target, source=False, target=True, split='train')
        source_test = breeds.Breeds(data_path, source, source=True, target=False, split='val')
        target_test = breeds.Breeds(data_path, target, source=False, target=True, split='val')
    elif dataset_name == 'domainnet':
        data_attr = 'data'
        source_train = domainnet.DomainNet(source, split='train', root=data_path)
        target_train = domainnet.DomainNet(target, split='train', root=data_path)
        source_test = domainnet.DomainNet(source, split='test', root=data_path)
        target_test = domainnet.DomainNet(target, split='test', root=data_path)
    else:
        raise ValueError('Only supports Breeds and DomainNet currently')
    filter_to_single_class(source_train, class_idx, data_attr)
    filter_to_single_class(target_train, class_idx, data_attr)
    filter_to_single_class(source_test, class_idx, data_attr)
    filter_to_single_class(target_test, class_idx, data_attr)
    train_ds = DomainClassificationDataset(source_train, target_train, data_attr, dataset_name,
                                           transform=transform, data_path=data_path)
    test_ds = DomainClassificationDataset(source_test, target_test, data_attr, dataset_name,
                                          transform=transform, data_path=data_path)
    return train_ds, test_ds

def get_pooled_datasets(dataset_name, source, target, data_path, transform):
    if dataset_name == 'breeds':
        data_attr = '_image_paths_by_class'
        source_train = breeds.Breeds(data_path, source, source=True, target=False, split='train')
        target_train = breeds.Breeds(data_path, target, source=False, target=True, split='train')
        source_test = breeds.Breeds(data_path, source, source=True, target=False, split='val')
        target_test = breeds.Breeds(data_path, target, source=False, target=True, split='val')
    elif dataset_name == 'domainnet':
        data_attr = 'data'
        source_train = domainnet.DomainNet(source, split='train', root=data_path)
        target_train = domainnet.DomainNet(target, split='train', root=data_path)
        source_test = domainnet.DomainNet(source, split='test', root=data_path)
        target_test = domainnet.DomainNet(target, split='test', root=data_path)
    else:
        raise ValueError('Only supports Breeds and DomainNet currently')
    train_ds = DomainClassificationDataset(source_train, target_train, data_attr, dataset_name,
                                           transform=transform, data_path=data_path)
    test_ds = DomainClassificationDataset(source_test, target_test, data_attr, dataset_name,
                                          transform=transform, data_path=data_path)
    return train_ds, test_ds

######################
### TRAINING STUFF ###
######################

def get_classes_to_compare(num_classes, num, seed):
    prng = np.random.RandomState(seed)
    classes = []
    class_1, class_2 = None, None # for proper scope
    for _ in range(num):
        while True:
            class_1, class_2 = prng.choice(num_classes, size=2, replace=False)
            curr_pair = sorted([class_1, class_2])
            if curr_pair not in classes:
                classes.append(curr_pair)
                break
    return classes

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
