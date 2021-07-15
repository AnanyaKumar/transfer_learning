import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import os

from unlabeled_extrapolation.datasets.connectivity_utils import *

def get_max_iters(dataset, domain):
    if dataset == 'breeds':
        if domain == 'entity30':
            return 100
        if domain == 'living17':
            return 200
        raise NotImplementedError('Other Breeds tasks not supported.')
    if dataset == 'domainnet':
        return 150

def get_num_classes(dataset, domain):
    if dataset == 'breeds':
        if domain == 'entity30':
            return 30
        if domain == 'living17':
            return 17
        raise NotImplementedError('Other Breeds tasks not supported.')
    if dataset == 'domainnet':
        return 40

def get_acc(preds, labels):
    return np.mean(preds == labels)

def ell_2_normalize(features):
    for i in range(len(features)):
        print('features shape', features[i].shape)
        norms = np.linalg.norm(features[i], axis=1)
        print('norms shape', norms.shape)
        features[i] = features[i] / norms

def make_binary_dataset(ds1, ds2):
    features = np.concatenate((ds1, ds2))
    labels = np.concatenate((np.zeros(len(ds1)), np.ones(len(ds2))))
    print('features labels shape', features.shape, labels.shape)
    return features, labels

def get_single_class(features, labels, column, class_idx):
    return features[column][labels[column] == class_idx]

def classify_domain(features, labels, max_iters, num_classes, args):
    source_train_idx, target_train_idx, source_test_idx, target_test_idx = range(4)
    accuracies = []
    for class_idx in range(num_classes):
        # train
        curr_source_train = get_single_class(features, labels, source_train_idx, class_idx)
        curr_target_train = get_single_class(features, labels, target_train_idx, class_idx)
        train_features, train_labels = make_binary_dataset(curr_source_train, curr_target_train)
        # test
        curr_source_test = get_single_class(features, labels, source_test_idx, class_idx)
        curr_target_test = get_single_class(features, labels, target_test_idx, class_idx)
        test_features, test_labels = make_binary_dataset(curr_source_test, curr_target_test)
        accuracies.extend(test_log_reg_warm_starting(
            train_features, train_labels, test_features, test_labels, args.num_reg_values,
            max_iters
        ))
    return accuracies

def class_comparison(features, labels, train_idx, test_idx,
                     class_1, class_2, num_reg_values, max_iters):
    train_1 = get_single_class(features, labels, train_idx, class_1)
    train_2 = get_single_class(features, labels, train_idx, class_2)
    train_features, train_labels = make_binary_dataset(train_1, train_2)
    test_1 = get_single_class(features, labels, test_idx, class_1)
    test_2 = get_single_class(features, labels, test_idx, class_2)
    test_features, test_labels = make_binary_dataset(test_1, test_2)
    result = test_log_reg_warm_starting(train_features, train_labels,
                                        test_features, test_labels,
                                        num_reg_values, max_iters)
    return result

def classify_class(features, labels, max_iters, num_classes, args):
    source_train_idx, target_train_idx, source_test_idx, target_test_idx = range(4)
    accuracies = []
    classes_to_compare = get_classes_to_compare(num_classes, args.num_class_comparisons, args.seed)
    for class_1, class_2 in classes_to_compare:
        # Do the source first
        result = class_comparison(features, labels, source_train_idx, source_test_idx,
                                  class_1, class_2, args.num_reg_values, max_iters)
        for r in result:
            accuracies.append([class_1, class_2] + r)
        # Do the target
        result = class_comparison(features, labels, target_train_idx, target_test_idx,
                                  class_1, class_2, args.num_reg_values, max_iters)
        for r in result:
            accuracies.append([class_1, class_2] + r)
    return accuracies

def test_log_reg_warm_starting(train_features, train_labels, test_features,
                               test_labels, num_reg_values, max_iters):
    Cs = np.logspace(-7, 0, num_reg_values)
    clf = LogisticRegression(random_state=0, warm_start=True, max_iter=max_iters)
    train_accs, test_accs = [], []
    for C in Cs:
        clf.C = C
        clf.fit(train_features, train_labels)
        train_preds = clf.predict(train_features)
        train_acc = get_acc(train_preds, train_labels)
        test_preds = clf.predict(test_features)
        test_acc = get_acc(test_preds, test_labels)
        print('Train accuracy', train_acc, 'and test accuracy', test_acc)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    return [train_accs, test_accs]

def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
    parser.add_argument('--run_dir', type=str,
                        help='Outer run directory to use.', required=True)
    parser.add_argument('--test_between', choices=['classes', 'domains'], required=True,
                        help='Whether to test between domains or classes.')
    parser.add_argument('--num_class_comparisons', type=int, default=20,
                        help='Maximum number of class-wise comparisons to do for class prediction.')
    parser.add_argument('--num_reg_values', type=int, default=20,
                        help='Number of regularization values to sweep over.')
    parser.add_argument('--seed', type=int, default=20,
                        help='Seed for choosing random pairs to compare for test-between-classes.')
    args = parser.parse_args()
    load_path = os.path.join(args.run_dir, 'finetuning', 'features_and_labels.pickle')
    if not os.path.exists(load_path):
        raise ValueError(f'Must run extract_features.py first to get {load_path}. Exiting...')
    save_path = os.path.join(args.run_dir, 'rep_connect',
                             f'sklearn_rep_connectivity_{args.test_between}.tsv')
    if os.path.exists(save_path):
        print(f'Already exists results at {save_path}. Exiting...')
        exit(0)
    features, labels, previous_args = pickle.load(open(load_path, 'rb'))
    print(f'Using representations from {previous_args.dataset}, source {previous_args.source}, '
          f'target {previous_args.target}, using model {previous_args.ckpt_name} from {previous_args.run_dir}.')
    ell_2_normalize(features)
    max_iters = get_max_iters(previous_args.dataset, previous_args.source)
    num_classes = get_num_classes(previous_args.dataset, previous_args.source)
    if args.test_between == 'domains':
        accs = classify_domain(features, labels, max_iters, num_classes, args)
    else: # classes
        accs = classify_class(features, labels, max_iters, num_classes, args)
    accs_df = pd.DataFrame(accs)
    accs_df.to_csv(save_path, sep='\t')

if __name__ == "__main__":
    main()
