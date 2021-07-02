import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import os

def get_max_iters(dataset_name):
    if dataset_name == 'entity30':
        return 100
    elif dataset_name == 'living17':
        return 200
    else:
        raise ValueError('Not supported')

def get_num_classes(dataset_name):
    if dataset_name == 'entity30':
        return 30
    if dataset_name == 'living17':
        return 17
    raise ValueError('Dataset not supported.')


def get_acc(preds, labels):
    return np.mean(preds == labels)


def normalize_features(features, train_col=0):
    # train_col is the index to compute mean and std-dev
    mean = np.mean(features[train_col])
    stddev = np.std(features[train_col])
    for i in range(len(features)):
        features[i] = (features[i] - mean) / stddev

def make_binary_dataset(ds1, ds2):
    features = np.concatenate((ds1, ds2))
    labels = np.concatenate((np.zeros(len(ds1)), np.ones(len(ds2))))
    return features, labels

def get_single_class(features, labels, column, class_idx):
    return features[column][labels[column] == class_idx]

def classify_domain(features, labels, dataset_name, num_reg_values):
    num_classes = get_num_classes(dataset_name)
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
        # concatenate the datasets
        test_features, test_labels = make_binary_dataset(curr_source_test, curr_target_test)
        accuracies.extend(test_log_reg_warm_starting(
            train_features, train_labels, test_features, test_labels, num_reg_values, dataset_name
        ))
    return accuracies

def class_comparison(features, labels, train_idx, test_idx,
                     class_1, class_2, num_reg_values, dataset_name):
    curr_source_train_1 = get_single_class(features, labels, train_idx, class_1)
    curr_source_train_2 = get_single_class(features, labels, train_idx, class_2)
    source_train_features, source_train_labels = make_binary_dataset(curr_source_train_1,
                                                                     curr_source_train_2)
    curr_source_test_1 = get_single_class(features, labels, test_idx, class_1)
    curr_source_test_2 = get_single_class(features, labels, test_idx, class_2)
    source_test_features, source_test_labels = make_binary_dataset(curr_source_test_1,
                                                                   curr_source_test_2)
    result = test_log_reg_warm_starting(source_train_features, source_train_labels,
                                        source_test_features, source_test_labels,
                                        num_reg_values, dataset_name)
    return result

def classify_class(features, labels, dataset_name, num_reg_values, class_comparisons):
    num_classes = get_num_classes(dataset_name)
    source_train_idx, target_train_idx, source_test_idx, target_test_idx = range(4)
    accuracies = []
    already_chosen = set()
    class_1, class_2 = None, None # for proper scope
    for _ in range(class_comparisons):
        while True:
            class_1, class_2 = np.random.choice(num_classes, size=2, replace=False)
            curr_pair = tuple(sorted([class_1, class_2]))
            if curr_pair not in already_chosen:
                already_chosen.add(curr_pair)
                break
        # Do the source first
        result = class_comparison(features, labels, source_train_idx, source_test_idx,
                                  class_1, class_2, num_reg_values, dataset_name)
        for r in result:
            accuracies.append([class_1, class_2] + r)
        # Do the target
        result = class_comparison(features, labels, target_train_idx, target_test_idx,
                                  class_1, class_2, num_reg_values, dataset_name)
        for r in result:
            accuracies.append([class_1, class_2] + r)
    return accuracies

def test_log_reg_warm_starting(train_features, train_labels, test_features,
                               test_labels, num_reg_values, dataset_name):
    max_iters = get_max_iters(dataset_name)
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
    parser.add_argument('--load_path', type=str,
                        help='Pickle file where features, labels are saved', required=True)
    parser.add_argument('--test_between', choices=['classes', 'domains'], required=True,
                        help='Whether to test between domains or classes.')
    parser.add_argument('--class_comparisons', type=int, default=20,
                        help='Maximum number of class-wise comparisons to do for class prediction.')
    parser.add_argument('--num_reg_values', type=int, default=20,
                        help='Number of regularization values to sweep over.')
    args = parser.parse_args()

    features, labels, previous_args = pickle.load(open(args.load_path, 'rb'))
    print(f'Using representations from {previous_args.dataset} {previous_args.dataset_name}, using '
          f'model {previous_args.checkpoint_dir} {previous_args.checkpoint_names[0]}, '
          f'now using a training data fraction {args.train_data_frac}.')

    if previous_args.dataset != 'breeds':
        raise ValueError('Not supported currently.')

    # only use the first model
    features, labels = features[0], labels[0]
    if len(features) != 4:
        raise ValueError('Must re-calculate features so that target train features are present.')
    normalize_features(features)

    if args.test_between == 'domains':
        accs = classify_domain(features, labels, previous_args.dataset_name, args.num_reg_values)
    else: # classes
        accs = classify_class(features, labels, previous_args.dataset_name, args.num_reg_values,
                              args.class_comparisons)
    accs_df = pd.DataFrame(accs)
    directory = os.path.dirname(args.load_path)
    save_path = os.path.join(directory, f'sklearn_representation_between_{args.test_between}.tsv')
    accs_df.to_csv(save_path, sep='\t')


if __name__ == "__main__":
    main()

