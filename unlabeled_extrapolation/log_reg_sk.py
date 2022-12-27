
import argparse
import copy
from collections import OrderedDict
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
import socket
import yaml
import quinine
import pickle
import pandas as pd


def get_acc(preds, labels):
    return np.mean(preds == labels)


def normalize_features(features, normalize_index):
    # normalize_index is the index to compute mean and std-dev
    # TODO: consider changing to axis=0
    mean = np.mean(features[normalize_index])
    stddev = np.std(features[normalize_index])
    normalized_features = []
    for i in range(len(features)):
        normalized_features.append((features[i] - mean) / stddev)
    return normalized_features


def inv_normalize_weights(weights, intercept, features, normalize_index):
    # mean = np.mean(features[normalize_index], axis=0)
    # stddev = np.std(features[normalize_index], axis=0)
    # new_weights = weights / stddev
    # new_intercept = intercept - np.matmul(weights, mean / stddev)
    # Other version
    mean = np.mean(features[normalize_index])
    stddev = np.std(features[normalize_index])
    new_weights = weights / stddev
    new_intercept = intercept - np.matmul(weights, mean / stddev * np.ones(weights.shape[1]))
    return new_weights, new_intercept


def pad_head(coef, intercept, classes_present):
    # Create a map from class to index in coef
    # Create coef and intercept of the current shape
    # Loop over real number of classes. if index doesn't exist in classes, then initialize to 0. Otherwise use coef intercept.
    num_classes = np.max(classes_present) + 1
    assert num_classes >= len(classes_present)
    assert len(classes_present) == coef.shape[0] == intercept.shape[0]
    new_coefs = np.zeros((num_classes, coef.shape[1]))
    new_intercept = np.zeros((num_classes,))
    for coef_index, present_class in enumerate(list(classes_present)):
        new_coefs[present_class, :] = coef[coef_index, :]
        new_intercept[present_class] = intercept[coef_index]
    return new_coefs, new_intercept


def test_log_reg_warm_starting(features, labels, train_index, test_indices, val_index, loader_names,
                               num_cs=100, start_c=-7, end_c=2, max_iter=200, random_state=0):
    L = len(features)
    # TODO: figure out what this should be based on initial results.
    Cs = np.logspace(start_c, end_c, num_cs)
    clf = LogisticRegression(random_state=random_state, warm_start=True, max_iter=max_iter)
    #.fit(features[m][train_index], labels[m][train_index])
    accs = []
    best_acc = -1.0
    best_clf, best_coef, best_intercept, best_i, best_c = None, None, None, None, None
    for i, C in zip(range(len(Cs)), Cs):
        clf.C = C
        clf.fit(features[train_index], labels[train_index])
        cur_accs = []
        for l in test_indices:
            cur_preds = clf.predict(features[l])
            # These names are selected to be consistent with fine-tuning results.
            # If you update these, please update scripts/run_adaptation_experiments.py
            if l == train_index:
                key = 'train/acc'
            else:
                key = 'test_acc/' + loader_names[l]
            cur_acc = get_acc(cur_preds, labels[l])
            # Don't multiply by 100, we multiply later in summarize_linprobe_results.py
            cur_accs.append((key, cur_acc))
            if l == val_index and cur_acc > best_acc:
                best_acc = cur_acc
                best_clf = copy.deepcopy(clf)
                best_coef = copy.deepcopy(clf.coef_)
                best_intercept = copy.deepcopy(clf.intercept_)
                best_i = i
                best_c = C
        print(cur_accs, flush=True)
        result_row = OrderedDict([('C', C)] + cur_accs)
        accs.append(result_row)
    return best_clf, best_coef, best_intercept, best_c, best_i, accs


def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
    parser.add_argument('--load_path', type=str,
                        help='Pickle file where features, labels are saved', required=True)
    parser.add_argument('--results_save_path', type=str,
                        help='Path to save tsv results file', required=True)
    parser.add_argument('--weights_save_path', type=str,
                        help='Path to save best logistic regression weights', required=True)
    parser.add_argument('--test_indices', type=int, nargs='+',
                        help='Datasets to test on.', required=False)
    parser.add_argument('--train_index', type=int,
                        help='Dataset to train on.', required=False)
    parser.add_argument('--num_reg_values', type=int,
                        help='Number of regularization values to sweep over.', required=False)
    parser.add_argument('--val_metric', type=str,
                        help='Metric to select regularization on.', required=True)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', required=False, default=0)
    parser.add_argument('--no_padding', action='store_true', help='dont pad missing labels')
    args, unparsed = parser.parse_known_args()
    
    print('Running on machine ', socket.gethostname())
    features, labels, loader_names = pickle.load(open(args.load_path, "rb"))
    assert len(features) == len(labels) == len(loader_names)
    if args.train_index is None:
        args.train_index = 0
    if args.test_indices is None:
        args.test_indices = list(range(0, len(loader_names))) 
    print("Training on: ", loader_names[args.train_index])
    print("Testing on: ")
    for idx in args.test_indices:
        print(loader_names[idx])
    # Find the index of dataset to tune logistic regression parameters on.
    val_index = -1
    for i, name in zip(range(len(loader_names)), loader_names):
        if name == args.val_metric or 'test_acc/' + name == args.val_metric:
            val_index = i
            break
    if val_index == -1:
        raise ValueError('Val metric not found in loaders: ', loader_names)
    # Normalize featurs, this makes regularization hyperparameters more consistent across datasets.
    normalized_features = normalize_features(features, args.train_index)
    # Get the best classifier.
    clf, coef, intercept, best_c, best_i, accs = test_log_reg_warm_starting(
            normalized_features, labels, args.train_index, args.test_indices, val_index=val_index,
            loader_names=loader_names, num_cs=args.num_reg_values, random_state=args.seed) 
    # Create parent folders for save paths if needed.
    def make_parent_dir(save_path):
        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    make_parent_dir(args.results_save_path)
    make_parent_dir(args.weights_save_path)
    # Save accuracies and weights to files.
    accs_df = pd.DataFrame(accs)
    accs_df.to_csv(args.results_save_path, sep='\t')
    new_coef, new_intercept = inv_normalize_weights(coef, intercept, features,
                                                    normalize_index=args.train_index)
    # Pad the head if desired, if some classes are missing from the training set, e.g., in FMoW.
    if not(args.no_padding):
        new_coef, new_intercept = pad_head(new_coef, new_intercept, clf.classes_)
    pickle.dump((new_coef, new_intercept, best_c, best_i), open(args.weights_save_path, 'wb'))
    # Just a redundancy check that the best classifier weights, and best weights, are the same.
    assert(np.allclose(clf.coef_, coef))
    assert(np.allclose(clf.intercept_, intercept))


if __name__ == "__main__":
    main()

