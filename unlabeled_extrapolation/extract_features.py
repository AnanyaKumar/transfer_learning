
import argparse
import os
import sys
import torch
from torch import nn
import models
import json
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import preprocessing
import yaml
import quinine
import pickle

import unlabeled_extrapolation.utils.utils as utils


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


def get_features_labels(net, loader, use_cuda=True):
    if use_cuda:
        net.cuda()
    net.eval()
    feature_model = nn.Sequential(*list(net._model.children())[:-1])
    features_list, labels_list = [], []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            features = feature_model(images)
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
    return features, labels


def main():
    parser = argparse.ArgumentParser(description='Extract features from model.')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    args, unparsed = parser.parse_known_args()
    config = quinine.Quinfig(args.config)
    utils.update_config(unparsed, config) 

    # Check for CUDA.
    print(f'cuda is available: {torch.cuda.is_available()}')
    print(config.loader_names)

    features, labels = get_model_representations(
        config.config_paths, config.checkpoint_paths, config.model_names,
        config.loader_names, config.loader_indices, config.split_arg_names,
        config.split_names, config.batch_size, config.num_workers, config.use_cuda)
    pickle.dump((features, labels, config.loader_names, config.model_names),
                open(config.save_path, 'wb'))


if __name__ == "__main__":
    main()

