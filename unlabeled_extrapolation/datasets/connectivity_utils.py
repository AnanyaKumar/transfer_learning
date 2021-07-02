import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from PIL import Image, ImageFilter
import random

from . import breeds, domainnet
VALID_BREEDS_DOMAINS = breeds.BREEDS_SPLITS_TO_FUNC.keys()
VALID_DOMAINNET_DOMAINS = domainnet.SENTRY_DOMAINS

#####################
### DATASET STUFF ###
#####################

def infer_dataset(source, target):
    if target is None:
        target = source
    if source in VALID_BREEDS_DOMAINS:
        dataset = 'breeds'
        if source != target:
            raise ValueError('For Breeds, must use the same source and target.')
        num_classes = {
            'entity30': 30,
            'living17': 17
        }[source]
        return dataset, num_classes
    if (source in VALID_DOMAINNET_DOMAINS and target in VALID_DOMAINNET_DOMAINS):
        dataset = 'domainnet'
        num_classes = domainnet.NUM_CLASSES_DICT['sentry']
        return dataset, num_classes
    raise ValueError('Must provide valid breeds task or domainnet domain pair.')

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
            list(filter(lambda item: item[1] == class_to_use, getattr(dataset, data_attr_name))))

def get_transforms(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.transform == 'imagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.transform == 'simclr':
        # adapted from
        # https://github.com/facebookresearch/moco/blob/master/main_moco.py
        augmentation = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize
        ]
        transform = transforms.Compose(augmentation)
    else:
        raise ValueError('Transformation scheme not supported')
    return transform

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

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

######################
### TRAINING STUFF ###
######################

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
