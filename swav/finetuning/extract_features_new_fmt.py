import argparse
import os
import copy
import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
import pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from unlabeled_extrapolation.models.imnet_resnet import ResNet50

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
)
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])

# Breeds-specific values
import unlabeled_extrapolation.datasets.breeds as breeds
BREEDS_ROOT = '/scr/biggest/imagenet'
VALID_BREEDS_TASKS = breeds.BREEDS_SPLITS_TO_FUNC.keys()

# DomainNet-specific values
import unlabeled_extrapolation.datasets.domainnet as domainnet
VALID_FULL_DOMAINNET_DOMAINS = domainnet.VALID_DOMAINS
VALID_SENTRY_DOMAINNET_DOMAINS = domainnet.SENTRY_DOMAINS
DOMAINNET_ROOT = '/scr/biggest/domainnet'


def load_feat_model(args):
    ckpt_path = os.path.join(args.run_dir, 'checkpoints', args.ckpt_name)
    model = ResNet50(pretrained=True, pretrain_style='swav', checkpoint_path=ckpt_path)
    feat_model = model.get_feature_extractor()
    feat_model = feat_model.cuda()
    return feat_model


def get_data_loader(dataset, domain, use_train, batch_size, num_workers, domainnet_version):
    if dataset == 'breeds':
        domain_name, use_source = domain
        ds = breeds.Breeds(root=BREEDS_ROOT, breeds_name=domain_name, source=use_source,
                           target=(not use_source), split=('train' if use_train else 'val'),
                           transform=TRANSFORM)
    elif dataset == 'domainnet':
        ds = domainnet.DomainNet(domain, split=('train' if use_train else 'test'),
                                 version=domainnet_version, root=DOMAINNET_ROOT, transform=TRANSFORM)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    return loader


def get_features_labels(feat_model, loader):
    feat_model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.cuda()
            features = feat_model(images)
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(labels.detach().numpy())
    features = np.squeeze(np.concatenate(features_list))
    labels = np.concatenate(labels_list)
    return [features, labels]


def get_acc(preds, labels):
    return np.mean(preds == labels)


def get_model_representations(args, previous_data):
    feat_model = load_feat_model(args)
    data = copy.deepcopy(previous_data)
    for domain in args.domains:
        if domain in data.keys():
            continue
        curr_domain = {}
        for split, use_train in zip(['train', 'test'], [True, False]):
                loader = get_data_loader(args.dataset, domain, use_train, args.batch_size,
                                         args.num_workers, args.domainnet_version)
                curr_domain[split] = get_features_labels(feat_model, loader)
        data[domain] = curr_domain
    return data


def main():
    parser = argparse.ArgumentParser(description='Extract features from model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers.')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='The (outer) run directory to use.')
    parser.add_argument('--ckpt_epoch', type=int, required=True,
                        help='The epoch of the checkpoint in the checkpoints/ folder.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset on which to calculate features.')
    parser.add_argument('--domains', type=str, required=True,
                        help='For DomainNet, the domain(s) to extract features for (comma-separated).')
    parser.add_argument('--domainnet_version', choices=['full', 'sentry'], default='sentry')
    parser.add_argument('--overwrite', action='store_true',
                        help='If set, will overwrite pre-existing files.')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.run_dir, 'finetuning'), exist_ok=True)
    file_prefix = 'features_and_labels'
    if args.dataset == 'domainnet':
        file_prefix += f'_{args.domainnet_version}'
    file_prefix += f'_{args.ckpt_epoch}'
    file_path = os.path.join(args.run_dir, 'finetuning', f'{file_prefix}_new_fmt.pickle')

    if (not args.overwrite) and (os.path.exists(file_path)):
        with open(file_path, 'rb') as f:
            previous_data, previous_args = pickle.load(f)
    else:
        previous_data = {}

    # check if done already
    args.domains = args.domains.split(',')
    if all((d in previous_data.keys()) for d in args.domains):
        print(f'All desired features have been extracted at {file_path}. Exiting...')
        exit(0)

    # input validation
    if args.dataset == 'breeds':
        if any((d not in VALID_BREEDS_TASKS) for d in args.domains):
            raise ValueError(f'Invalid Breeds task name: provided domains {",".join(args.domains)}.')
        domains = []
        for d in args.domains:
            domains.extend([(d, True), (d, False)]) # source and target
        args.domains = domains
    elif args.dataset == 'domainnet':
        valid = VALID_FULL_DOMAINNET_DOMAINS if args.domainnet_version == 'full' else VALID_SENTRY_DOMAINNET_DOMAINS
        if any((d not in valid) for d in args.domains):
            raise ValueError(f'Invalid DomainNet domain name: provided domains {",".join(args.domains)}.')
    else:
        raise ValueError(f'The dataset {args.dataset} is not currently supported.')

    data = get_model_representations(args, previous_data)
    pickle.dump((data, args), open(file_path, 'wb'))


if __name__ == "__main__":
    main()

