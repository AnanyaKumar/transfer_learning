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
        return 150


def get_acc(preds, labels):
    return np.mean(preds == labels)


def subsample_features(features, labels, frac, train_idx=0):
    num_samples = len(features[train_idx])
    idx = np.random.choice(num_samples, size=int(frac * num_samples), replace=False)
    features[train_idx] = features[train_idx][idx]
    labels[train_idx] = labels[train_idx][idx]


def normalize_features(features, train_idx=0):
    # train_idx is the index to compute mean and std-dev
    mean = np.mean(features[train_idx])
    stddev = np.std(features[train_idx])
    for i in range(len(features)):
        features[i] = (features[i] - mean) / stddev


def test_log_reg_warm_starting(features, labels, num_cs, max_iters):
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
    parser.add_argument('--run_dir', type=str,
                        help='Outer run directory to use.', required=True)
    parser.add_argument('--train_data_fracs', type=float, nargs='+', default=[1.0],
                        help='The amount of source data to actually use for fine-tuning.')
    parser.add_argument('--num_reg_values', type=int, default=50,
                        help='Number of regularization values to sweep over.', required=False)
    args = parser.parse_args()
    load_path = os.path.join(args.run_dir, 'finetuning', 'features_and_labels.pickle')
    if not os.path.exists(load_path):
        raise ValueError(f'Must run extract_features.py first to get {load_path}. Exiting...')
    for train_data_frac in args.train_data_fracs:
        save_path = os.path.join(args.run_dir, 'finetuning',
                                f'sklearn_finetuning_fraction_{train_data_frac}.tsv')
        if os.path.exists(save_path):
            print(f'Already exists results at {save_path}. Skipping...')
            continue
        features, labels, previous_args = pickle.load(open(load_path, 'rb'))
        print(f'Using representations from {previous_args.dataset}, source {previous_args.source}, '
            f'target {previous_args.target}, using model {previous_args.ckpt_name} from {previous_args.run_dir}, '
            f'now using a training data fraction {train_data_frac}.')
        subsample_features(features, labels, train_data_frac)
        normalize_features(features)
        max_iters = get_max_iters(previous_args.dataset, previous_args.source)
        accs = test_log_reg_warm_starting(features, labels, args.num_reg_values, max_iters)
        accs_df = pd.DataFrame(accs)
        accs_df.to_csv(save_path, sep='\t')


if __name__ == "__main__":
    main()

