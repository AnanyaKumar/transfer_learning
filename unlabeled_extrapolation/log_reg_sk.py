
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
import pandas as pd


def get_acc(preds, labels):
    return np.mean(preds == labels)


def normalize_features(features, normalize_index):
    # normalize_index is the index to compute mean and std-dev
    mean = np.mean(features[0][normalize_index])
    stddev = np.std(features[0][normalize_index])
    for i in range(len(features[0])):
        features[0][i] = (features[0][i] - mean) / stddev


def test_log_reg_warm_starting(features, labels, source_idx, test_indices, num_cs=50):
    M, L = len(features), len(features[0])
    m = 0
    Cs = np.logspace(-7, 0, num_cs)
    clf = LogisticRegression(random_state=0, warm_start=True, max_iter=200)
    #.fit(features[m][source_idx], labels[m][source_idx])
    accs = []
    for C in Cs:
        clf.C = C
        clf.fit(features[m][source_idx], labels[m][source_idx])
        cur_accs = []
        for l in test_indices:
            cur_preds = clf.predict(features[m][l])
            cur_acc = get_acc(cur_preds, labels[m][l])
            cur_accs.append(cur_acc)
        print(cur_accs)
        accs.append(cur_accs)
    return clf, accs


def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
    parser.add_argument('--load_path', type=str,
                        help='Pickle file where features, labels are saved', required=True)
    parser.add_argument('--save_path', type=str,
                        help='Path to save tsv results file', required=True)
    parser.add_argument('--test_indices', type=int, nargs='+',
                        help='Datasets to test on.', required=True)
    parser.add_argument('--train_index', type=int,
                        help='Dataset to train on.', required=True)
    parser.add_argument('--num_reg_values', type=int,
                        help='Number of regularization values to sweep over.', required=False)
    args, unparsed = parser.parse_known_args()
    features, labels, loader_names, model_names = pickle.load(open(args.load_path, "rb"))
    print("Training on: ", loader_names[args.train_index])
    print("Testing on: ")
    for idx in args.test_indices:
        print(loader_names[idx])
    normalize_features(features, args.train_index)
    clf, accs = test_log_reg_warm_starting(
            features, labels, args.train_index, args.test_indices, num_cs=args.num_reg_values)
    accs_df = pd.DataFrame(accs)
    accs_df.to_csv(args.save_path, sep='\t')
    

if __name__ == "__main__":
    main()

