# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger
import os

from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from unlabeled_extrapolation.datasets.domainnet import DomainNet, SENTRY_DOMAINS, VALID_DOMAINS
from unlabeled_extrapolation.datasets.breeds import Breeds, BREEDS_SPLITS_TO_FUNC

BREEDS_TASKS = BREEDS_SPLITS_TO_FUNC.keys()

logger = getLogger()


def parse_splits(dataset_kwargs):
    source_amount = process_data_amount(dataset_kwargs, 'source_amount')
    target_amount = process_data_amount(dataset_kwargs, 'target_amount')
    related_amount = process_data_amount(dataset_kwargs, 'related_amount')
    if source_amount > 0 and related_amount > 0:
        raise ValueError('Should not try to add both extra source and related data. Please '
                         'choose one.')
    return source_amount, target_amount, related_amount


def process_data_amount(args, key):
    size = args.get(key, 0)
    if size == -1:
        size = 1000 # something larger than any single dataset, integer version of "infinity"
    return size


class CustomSplitDataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset_name,
        dataset_kwargs,
        transform,
        seed
    ):
        super().__init__()
        self._transform = transform
        self.prng = np.random.RandomState(seed) # deterministic subsampling

        if dataset_name == 'imagenet':
            # standard ImageNet, no subsampling
            self.means = [0.485, 0.456, 0.406]
            self.stds = [0.228, 0.224, 0.225]
            dataset = datasets.ImageFolder(os.path.join(data_path, 'train'))
            self.samples = dataset.samples

        elif dataset_name == 'breeds':
            breeds_name = dataset_kwargs.get('breeds_name')
            if breeds_name not in BREEDS_TASKS:
                raise ValueError(f'Must provide a task in {BREEDS_TASKS}, {breeds_name} provided.')
            source_amount, target_amount, related_amount = parse_splits(dataset_kwargs)

            # calculate the reference "100%" dataset size, min of source and target sizes
            source_ds = Breeds(data_path, breeds_name, source=True, target=False, split='train')
            source_samples = source_ds._image_paths_by_class
            target_ds = Breeds(data_path, breeds_name, source=False, target=True, split='train')
            target_samples = target_ds._image_paths_by_class
            reference_size = min(len(source_samples), len(target_samples))

            self.means = target_ds.means # same means/stds is used for all Breeds tasks
            self.stds = target_ds.stds

            # fill in the proper proportions of samples
            self.samples = []
            self.add_data(reference_size, target_amount, target_samples, 'target')
            self.add_data(reference_size, source_amount, source_samples, 'source')
            if related_amount > 0:
                # collect all ImageNet images
                all_imagenet = datasets.ImageFolder(os.path.join(data_path, 'train'))
                all_imagenet_images = all_imagenet.samples
                # collect all Breeds target images
                target_images = set([item[0] for item in target_samples])
                # identify non-target ImageNet images
                non_target_imagenet = list(filter(lambda item: item[0] not in target_images,
                                                  all_imagenet_images))
                self.add_data(reference_size, related_amount, non_target_imagenet, 'related')

        elif dataset_name == 'domainnet':
            use_sentry = dataset_kwargs.get('use_sentry', True)
            version = 'sentry' if use_sentry else 'full'
            source_domain = dataset_kwargs.get('source_domain')
            target_domain = dataset_kwargs.get('target_domain')
            if source_domain is not None and target_domain is not None: # for UDA
                valid_domainset = SENTRY_DOMAINS if use_sentry else VALID_DOMAINS
                if (source_domain not in valid_domainset) or (target_domain not in valid_domainset):
                    raise ValueError(f'Valid domains are {valid_domainset}, but source was {source_domain} '
                                    f'and target was {target_domain}.')
                source_amount, target_amount, related_amount = parse_splits(dataset_kwargs)
                # calculate the reference "100%" dataset size, min of source and target sizes
                source_ds = DomainNet(source_domain, root=data_path, split='train', version=version)
                source_samples = source_ds.data
                target_ds = DomainNet(target_domain, root=data_path, split='train', version=version)
                target_samples = target_ds.data
                reference_size = min(len(source_samples), len(target_samples))

                self.means = target_ds.means # same means/stds is used for all DomainNet domains
                self.stds = target_ds.stds

                # fill in the proper proportions of samples
                self.samples = []
                self.add_data(reference_size, target_amount, target_samples, 'target')
                self.add_data(reference_size, source_amount, source_samples, 'source')
                if related_amount > 0:
                    # identify all non-target domains
                    non_target_domains = list(set(valid_domainset) - set([target_domain]))
                    non_target_domains = ','.join(non_target_domains)
                    # collect all DomainNet non-target images
                    non_target_domainnet = DomainNet(non_target_domains, root=data_path,
                                                    split='train', version=version)
                    self.add_data(reference_size, related_amount, non_target_domainnet.data, 'related')
            else: # not UDA
                domains = dataset_kwargs.get('domains')
                ds = DomainNet(domains, root=data_path, split='train', version=version)
                self.samples = ds.data
                self.means = ds.means # same means/stds is used for all DomainNet domains
                self.stds = ds.stds

            # for DomainNet, edit all the samples so the __getitem__ function is consistent
            func = lambda inp: (os.path.join(data_path, inp[0]), int(inp[1]))
            self.samples = list(map(func, self.samples))

        else:
            raise ValueError('Unsupported dataset name: must be one of [imagenet, breeds, domainnet].')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, y = self.samples[index]
        x = Image.open(path)
        x = x.convert('RGB')
        if self._transform is not None:
            x = self._transform(x)
        return x, y

    def add_data(self, reference_size, fraction, samples_to_add, label):
        size_to_use = int(fraction * reference_size)
        permutation = self.prng.permutation(len(samples_to_add))
        new_samples = [samples_to_add[i] for i in permutation[:size_to_use]]
        logger.info(f'Using {len(new_samples)} examples from {label}.')
        self.samples.extend(new_samples)


class CustomSplitMultiCropDataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset_name,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        seed=None,
        dataset_kwargs={},
        size_dataset=-1,
        return_index=False
    ):
        super().__init__()

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        if size_dataset >= 0:
            raise NotImplementedError('Overridden by the custom splitting functionality.')
        if len(dataset_kwargs) > 0 and seed is None:
            raise ValueError('If subsampling, must provide a seed.')

        self.ds = CustomSplitDataset(data_path, dataset_name, dataset_kwargs, None, seed)
        self.return_index = return_index
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.ds.means, std=self.ds.stds)])
            ] * nmb_crops[i])
        self.trans = trans
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        image, _ = self.ds[index]
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
