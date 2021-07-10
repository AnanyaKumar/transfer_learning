import argparse
from multiprocessing import Value
import os
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
# TODO: same for DomainNet as well?
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
BREEDS_NUM_CLASSES_DICT = {
    'entity30': 30,
    'living17': 17
}

# DomainNet-specific values
import unlabeled_extrapolation.datasets.domainnet as domainnet
VALID_DOMAINNET_DOMAINS = domainnet.VALID_SENTRY_DOMAINS
DOMAINNET_NUM_CLASSES = domainnet.NUM_CLASSES_DICT['sentry']

def load_model(ckpt_path, args):
    model = ResNet50(pretrained=True, pretrain_style='swav', checkpoint_path=ckpt_path)
    num_classes = get_num_classes(args)
    model.new_last_layer(num_classes)
    return model


def get_num_classes(args):
    if args.dataset == 'breeds':
        return BREEDS_NUM_CLASSES_DICT[args.dataset_name]
    elif args.dataset == 'domainnet':
        return DOMAINNET_NUM_CLASSES
    else:
        raise ValueError(f'Currently, only Breeds and DomainNet are supported, but received {args.dataset}.')


def get_data_loader(dataset, source, target, use_source, split,
                    batch_size=64, num_workers=2):
    if dataset == 'breeds':
        ds = breeds.Breeds(root=BREEDS_ROOT, breeds_name=source, source=use_source,
            target=(not use_source), split=split, transform=TRANSFORM)
    elif dataset == 'domainnet':
        domain_to_use = source if use_source else target
        ds = domainnet.DomainNet(domain_to_use, split=split, version='sentry')
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)
    return loader


def get_features_labels(net, loader, use_cuda=True):
    if use_cuda:
        net = net.cuda()
    net.eval()
    feature_model = nn.Sequential(*list(net._model.children())[:-1])
    features_list, labels_list = [], []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if use_cuda:
                images = images.cuda()
            features = feature_model(images)
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(labels.detach().numpy())
    features = np.squeeze(np.concatenate(features_list))
    labels = np.concatenate(labels_list)
    return features, labels


def get_acc(preds, labels):
    return np.mean(preds == labels)


# Given a network how to get representations
def get_model_representations(args):
    model = load_model(args.checkpoint_path, args)

    source_train_loader = get_data_loader(args.dataset, args.source, args.target, True, 'train',
                                          args.batch_size, args.num_workers)
    target_train_loader = get_data_loader(args.dataset, args.source, args.target, False, 'train',
                                          args.batch_size, args.num_workers)
    source_test_loader = get_data_loader(args.dataset, args.source, args.target, True, 'val',
                                         args.batch_size, args.num_workers)
    target_test_loader = get_data_loader(args.dataset, args.source, args.target, False, 'val',
                                         args.batch_size, args.num_workers)

    features, labels = [None] * 4, [None] * 4
    features[0], labels[0] = get_features_labels(model, source_train_loader, use_cuda=args.use_cuda) 
    features[1], labels[1] = get_features_labels(model, target_train_loader, use_cuda=args.use_cuda) 
    features[2], labels[2] = get_features_labels(model, source_test_loader, use_cuda=args.use_cuda) 
    features[3], labels[3] = get_features_labels(model, target_test_loader, use_cuda=args.use_cuda) 
    return features, labels


def main():
    parser = argparse.ArgumentParser(description='Extract features from model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers.')
    parser.add_argument('--no_cuda', action='store_true', help='If set, will use CPU.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='The saved model checkpoint to use.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset on which to calculate features.')
    parser.add_argument('--source', type=str,
                        help='If using Breeds, the task to use; if using DomainNet, the source domain to use.')
    parser.add_argument('--target', type=str,
                        help='If using DomainNet, the target domain to use.')
    args = parser.parse_args()

    args.use_cuda = not args.no_cuda
    if args.dataset == 'breeds':
        if args.source not in VALID_BREEDS_TASKS:
            raise ValueError(f'Invalid Breeds task name: provided {args.source}.')
        if args.target != args.source:
            print('Breeds uses the same task for source and target; overwriting target arg with source.')
            args.target = args.source
    if args.dataset == 'domainnet':
        if args.source not in VALID_DOMAINNET_DOMAINS or args.target not in VALID_DOMAINNET_DOMAINS:
            raise ValueError(f'Invalid DomainNet domain name: provided source {args.source} and target {args.target}.')
    else:
        raise ValueError(f'The dataset {args.dataset} is not currently supported.')

    features, labels = get_model_representations(args)
    save_path = os.path.join(args.checkpoint_dir, 'features_and_labels.pickle')
    pickle.dump((features, labels, args), open(save_path, 'wb'))


if __name__ == "__main__":
    main()

