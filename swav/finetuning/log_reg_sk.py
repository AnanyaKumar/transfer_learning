import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
import os

def get_max_iters(dataset, domain_task):
    if dataset == 'breeds':
        if domain_task == 'entity30':
            return 100
        if domain_task == 'living17':
            return 200
        raise NotImplementedError('Other Breeds tasks not supported.')
    if dataset == 'domainnet':
        return 100


def get_acc(preds, labels):
    return np.mean(preds == labels)


def get_subsampled_features(features, labels, frac, train_col=0):
    num_samples = len(features[train_col])
    idx = np.random.choice(num_samples, size=int(frac * num_samples), replace=False)
    features[train_col] = features[train_col][idx]
    labels[train_col] = labels[train_col][idx]


def normalize_features(features, train_col=0):
    # train_col is the index to compute mean and std-dev
    mean = np.mean(features[train_col])
    stddev = np.std(features[train_col])
    for i in range(len(features)):
        features[i] = (features[i] - mean) / stddev


def test_log_reg_warm_starting(features, labels, num_cs=50, max_iters=100):
    train_col, source_test_col, target_test_col = 0, 2, 3 # 1 is target train
    Cs = np.logspace(-7, 0, num_cs)
    clf = LogisticRegression(random_state=0, warm_start=True, max_iter=max_iters)
    source_accs, target_accs = [], []
    for C in Cs:
        clf.C = C
        clf.fit(features[train_col], labels[train_col])
        source_preds = clf.predict(features[source_test_col])
        source_acc = get_acc(source_preds, labels[source_test_col])
        target_preds = clf.predict(features[target_test_col])
        target_acc = get_acc(target_preds, labels[target_test_col])
        print('Source accuracy', source_acc, 'and target accuracy', target_acc)
        source_accs.append(source_acc)
        target_accs.append(target_acc)
    return source_accs, target_accs


def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
    parser.add_argument('--load_path', type=str,
                        help='Pickle file where features, labels are saved', required=True)
    parser.add_argument('--train_data_frac', type=float, default=1.0,
                        help='The amount of source data to actually use for fine-tuning.')
    parser.add_argument('--num_reg_values', type=int,
                        help='Number of regularization values to sweep over.', required=False)
    args = parser.parse_args()
    features, labels, previous_args = pickle.load(open(args.load_path, 'rb'))
    print(f'Using representations from {previous_args.dataset}, source {previous_args.source}, '
          f'target {previous_args.target}, using model {previous_args.checkpoint_path} '
          f'now using a training data fraction {args.train_data_frac}.')
    if len(features) == 1: # for backwards compatibility
        features, labels = features[0], labels[0]
    get_subsampled_features(features, labels, args.train_data_frac)
    normalize_features(features)
    max_iters = get_max_iters(previous_args.dataset, previous_args.source)
    accs = test_log_reg_warm_starting(features, labels, num_cs=args.num_reg_values, max_iters=max_iters)
    accs_df = pd.DataFrame(accs)
    directory = os.path.dirname(args.load_path)
    save_path = os.path.join(directory, f'sklearn_finetuning_fraction_{args.train_data_frac}.tsv')
    accs_df.to_csv(save_path, sep='\t')


if __name__ == "__main__":
    main()

