import argparse
import os
from numpy.lib.utils import source
import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
import pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from unlabeled_extrapolation.models.imnet_resnet import ResNet50

# Breeds-specific values
import unlabeled_extrapolation.datasets.breeds as breeds
BREEDS_ROOT = '/scr/biggest/imagenet'
VALID_BREEDS_TASKS = breeds.BREEDS_SPLITS_TO_FUNC.keys()
BREEDS_NUM_CLASSES_DICT = {
    'entity30': 30,
    'living17': 17
}
BREEDS_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
)
BREEDS_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    BREEDS_NORMALIZE,
])


def load_model(ckpt_dir, ckpt_name, args):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model = ResNet50(pretrained=True, pretrain_style='swav', checkpoint_path=ckpt_path)
    num_classes = get_num_classes(args)
    model.new_last_layer(num_classes)
    return model


def get_num_classes(args):
    if args.dataset == 'breeds':
        return BREEDS_NUM_CLASSES_DICT[args.dataset_name]
    else:
        raise ValueError('Not supported.')


def get_breeds_data_loader(breeds_name, use_source, split, batch_size=64,
                           num_workers=2):
    dataset = breeds.Breeds(
        root=BREEDS_ROOT, breeds_name=breeds_name, source=use_source,
        target=(not use_source), split=split, transform=BREEDS_TRANSFORM)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
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


def make_none_list(rs, cs):
    return [[None] * cs for _ in range(rs)]


# Given a network how to get representations
def get_model_representations(args):
    M = len(args.checkpoint_names)
    models = []
    for m in range(M):
        model = load_model(args.checkpoint_dir, args.checkpoint_names[m], args)
        models.append(model)

    train_loader = get_breeds_data_loader(args.dataset_name, True, 'train',
                                          args.batch_size, args.num_workers)
    source_test_loader = get_breeds_data_loader(args.dataset_name, True, 'val',
                                                args.batch_size, args.num_workers)
    target_test_loader = get_breeds_data_loader(args.dataset_name, False, 'val',
                                                args.batch_size, args.num_workers)

    features, labels = make_none_list(M, 3), make_none_list(M, 3)
    for m in range(M):
        features[m][0], labels[m][0] = get_features_labels(models[m], train_loader, use_cuda=args.use_cuda) 
        features[m][1], labels[m][1] = get_features_labels(models[m], source_test_loader, use_cuda=args.use_cuda) 
        features[m][2], labels[m][2] = get_features_labels(models[m], target_test_loader, use_cuda=args.use_cuda) 
    return features, labels


def main():
    parser = argparse.ArgumentParser(description='Extract features from model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers.')
    parser.add_argument('--no-cuda', action='store_true', help='If set, will use CPU.')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Outer checkpoint directory')
    parser.add_argument('--checkpoint-names', type=str, nargs='+', required=True,
                        help='The saved model checkpoint(s) desired.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset on which to calculate features.')
    parser.add_argument('--dataset-name', type=str,
                        help='If using Breeds, the task to use.')
    args = parser.parse_args()

    args.use_cuda = not args.no_cuda
    if args.dataset == 'breeds':
        if args.dataset_name not in VALID_BREEDS_TASKS:
            raise ValueError(f'Invalid dataset name: provided {args.dataset_name}.')
    else:
        raise ValueError('Only Breeds is currently supported.')

    features, labels = get_model_representations(args)
    save_path = os.path.join(args.checkpoint_dir, 'features_and_labels.pickle')
    pickle.dump((features, labels, args), open(save_path, 'wb'))


if __name__ == "__main__":
    main()

