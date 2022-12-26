
import argparse
import os
import sys
import torch
from torch import nn
import json
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import preprocessing
import yaml
import quinine
import pickle
import socket

import unlabeled_extrapolation.utils.utils as utils
from unlabeled_extrapolation.baseline_train import build_model
from unlabeled_extrapolation.baseline_train import get_test_loaders
from unlabeled_extrapolation.baseline_train import get_train_loader
from unlabeled_extrapolation.baseline_train import preprocess_config


def load_model(config_path, checkpoint_path, use_cuda=True):
    with open(config_path) as f:
        config = json.load(f)
    net = utils.initialize(config['model'])
    if use_cuda:
        net = net.cuda()
    net.new_last_layer(config['num_classes'])
    utils.load_ckp(checkpoint_path, net)
    return net


# Load datasets.
def load_test_dataset(config, idx, split_arg_name='split', split='val', batch_size=64,
                      num_workers=2):
    test_config = config['test_datasets'][idx]
    test_config['args'][split_arg_name] = split
    print(test_config['name'], test_config)
    if 'transforms' not in test_config:
        test_config['transforms'] = config['default_test_transforms']
    test_data = utils.init_dataset(test_config)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)
    return test_data, test_loader


def get_features_labels(net, loader, use_cuda=True, train_mode=False, use_new_bn_stats=False):
    if use_cuda:
        net.cuda()
    if use_new_bn_stats:
        if train_mode:
            raise ValueError('If use_new_bn_stats, then train_mode must be False.')
        net.train()
        for data in loader:
            images, labels = data
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            net(images)
    if train_mode:
        net.train()
    else:
        net.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            features = net.get_features(images)
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
    features = np.squeeze(np.concatenate(features_list))
    labels = np.concatenate(labels_list)
    return features, labels


def get_acc(preds, labels):
    return np.mean(preds == labels)


def make_none_list(rs, cs):
    return [[None] * cs for _ in range(rs)]


# Given a network how to get representations
def get_model_representations(
    config_paths, checkpoint_paths, model_names, loader_names, loader_indices, split_arg_names, split_names,
    batch_size=64, num_workers=2, use_cuda=True):
    M, L = len(model_names), len(loader_names)
    models = []
    if not type(config_paths) == list:
    # If not a list, then just use the specified config_path and checkpoint path
        if M > 1:
            raise ValueError('Only specified one config path but > 1 models, see config_paths.')
        with open(config_paths) as f:
            config = json.load(f)
        models.append(load_model(config_paths, checkpoint_paths, use_cuda=use_cuda))
    else:
        with open(config_paths[0]) as f:
            config = json.load(f)
        for m in range(M):
            models.append(load_model(config_paths[m], checkpoint_paths[m], use_cuda=use_cuda))

    loaders = []
    for l in range(L):
        _, loader = load_test_dataset(
            config, loader_indices[l], split_arg_names[l], split_names[l], batch_size=batch_size,
            num_workers=num_workers)
        loaders.append(loader)
    features, labels = make_none_list(M, L), make_none_list(M, L)
    for m in range(M):
        for l in range(L):
            features[m][l], labels[m][l] = get_features_labels(models[m], loaders[l],
                                                               use_cuda=use_cuda) 


def main():
    parser = argparse.ArgumentParser(description='Extract features from model.')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--save_path', type=str, metavar='s',
                        help='Path to save extracted features.', required=True)
    parser.add_argument('--use_test_transforms_for_train', type=str,
                        help='no augmentations when training.', required=False,
                        default='False')
    parser.add_argument('--train_mode', type=str,
                        help='Produce features in train mode', required=False,
                        default='False')
    parser.add_argument('--use_new_bn_stats', type=str,
                        help='Update batchnorm stats before producing features.', required=False,
                        default='False')
    parser.add_argument('--no_wandb', action='store_true', help='disable W&B')
    args, unparsed = parser.parse_known_args()
    config = quinine.Quinfig(args.config)
    utils.update_config(unparsed, config) 

    # Check for CUDA.
    print(f'cuda is available: {torch.cuda.is_available()}')
    print('Running on machine ', socket.gethostname())
    # print(config.loader_names)
   
    # This makes specifying certain things more convenient, e.g. don't have to specify a
    # transform for every test datset.
    preprocess_config(config, args.config) 
    # Get network and data loaders.
    net = build_model(config)
    test_loaders, _ = get_test_loaders(config, shuffle=False)
    test_name_loaders = sorted(list(test_loaders.items()))
    test_names = [k for k, v in test_name_loaders]
    loader_names = [config.train_dataset.name] + test_names
    if args.use_test_transforms_for_train == 'True':
        print('Using test transform')
        if 'default_test_transforms' not in config:
            raise ValueError('Specify default test transforms if not using train transform.')
        config['train_dataset']['transforms'] = config['default_test_transforms']
        print(config['train_dataset'])
    elif args.use_test_transforms_for_train != 'False':
        raise ValueError(f'use_test_transforms_for_train must be True or False, but was '
                         f'{args.use_test_transforms_for_train}')
    train_loader = get_train_loader(config)
    
    # Get features and labels.
    train_mode = False
    if args.train_mode == 'True':
        train_mode = True
    elif args.train_mode != 'False':
        raise ValueError(f'train_mode must be True or False but was {args.train_mode}')
    use_new_bn_stats = False
    if args.use_new_bn_stats == 'True':
        use_new_bn_stats = True
    elif args.use_new_bn_stats != 'False':
        raise ValueError(f'use_new_bn_stats must be True or False but was {args.use_new_bn_stats}')
    train_features, train_labels = get_features_labels(net, train_loader, train_mode=train_mode,
                                                       use_new_bn_stats=use_new_bn_stats)
    features, labels = [train_features], [train_labels]
    for _, loader in test_name_loaders:
        cur_features, cur_labels = get_features_labels(net, loader)
        features.append(cur_features)
        labels.append(cur_labels)
    print(f'Processed {len(labels)} loaders')
    print('Output shape for first dataset: ', features[0].shape, labels[0].shape)

    # features, labels = get_model_representations(
    #     config.config_paths, config.checkpoint_paths, config.model_names,
    #     config.loader_names, config.loader_indices, config.split_arg_names,
    #     config.split_names, config.batch_size, config.num_workers, config.use_cuda)
    # print('output shapes: ', features[0][0].shape, labels[0][0].shape)
    # print('save path: ', config.save_path)
    
    # Create save directory.
    save_dir = os.path.dirname(args.save_path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    pickle.dump((features, labels, loader_names), open(args.save_path, 'wb'))
    print('Saved to: ', args.save_path)


if __name__ == "__main__":
    main()

