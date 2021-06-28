import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image, ImageFilter
import random

from . import breeds
VALID_BREEDS_DOMAINS = breeds.BREEDS_SPLITS_TO_FUNC.keys()

class DomainClassificationDataset(Dataset):
    def __init__(self, ds1, ds2, data_attr_name, transform=None):
        super().__init__()
        ds1_samples = getattr(ds1, data_attr_name)
        ds2_samples = getattr(ds2, data_attr_name)
        self.samples = [(item[0], 0) for item in ds1_samples] + \
            [(item[0], 1) for item in ds2_samples]
        self._transform = transform

    def __getitem__(self, i):
        path, y = self.samples[i]
        x = Image.open(path)
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

def get_class_datasets(args, class_1, class_2, transform, dataset_name, use_source):
    if dataset_name in VALID_BREEDS_DOMAINS and args.target in VALID_BREEDS_DOMAINS:
        data_attr = '_image_paths_by_class'
        
        train_1 = breeds.Breeds(args.data_path, dataset_name, source=use_source,
                                target=(not use_source), split='train')
        filter_to_single_class(train_1, class_1, data_attr)
        test_1 = breeds.Breeds(args.data_path, dataset_name, source=use_source,
                               target=(not use_source), split='val')
        filter_to_single_class(test_1, class_1, data_attr)
        
        train_2 = breeds.Breeds(args.data_path, dataset_name, source=use_source,
                                target=(not use_source), split='train')
        filter_to_single_class(train_2, class_2, data_attr)
        test_2 = breeds.Breeds(args.data_path, dataset_name, source=use_source,
                               target=(not use_source), split='val')
        filter_to_single_class(test_2, class_2, data_attr)
    else:
        raise ValueError('Only supports Breeds currently')
    train_ds = DomainClassificationDataset(train_1, train_2, data_attr, transform=transform)
    test_ds = DomainClassificationDataset(test_1, test_2, data_attr, transform=transform)
    return train_ds, test_ds

def get_domain_datasets(args, class_idx, transform):
    if args.source in VALID_BREEDS_DOMAINS and args.target in VALID_BREEDS_DOMAINS:
        data_attr = '_image_paths_by_class'
        source_train = breeds.Breeds(args.data_path, args.source, source=True, target=False, split='train')
        filter_to_single_class(source_train, class_idx, data_attr)
        target_train = breeds.Breeds(args.data_path, args.target, source=False, target=True, split='train')
        filter_to_single_class(target_train, class_idx, data_attr)
        
        source_test = breeds.Breeds(args.data_path, args.source, source=True, target=False, split='val')
        filter_to_single_class(source_test, class_idx, data_attr)
        target_test = breeds.Breeds(args.data_path, args.target, source=False, target=True, split='val')
        filter_to_single_class(target_test, class_idx, data_attr)
    else:
        raise ValueError('Only supports Breeds currently')
    train_ds = DomainClassificationDataset(source_train, target_train, data_attr, transform=transform)
    test_ds = DomainClassificationDataset(source_test, target_test, data_attr, transform=transform)
    return train_ds, test_ds

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
