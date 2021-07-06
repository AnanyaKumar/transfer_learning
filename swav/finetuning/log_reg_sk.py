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


def get_acc(preds, labels):
    return np.mean(preds == labels)


def get_subsampled_features(features, labels, frac, train_col=0):
    for row in range(len(features)):
        num_samples = len(features[row][train_col])
        idx = np.random.choice(num_samples, size=int(frac * num_samples), replace=False)
        features[row][train_col] = features[row][train_col][idx]
        labels[row][train_col] = labels[row][train_col][idx]

def normalize_features(features, train_col=0):
    for row in range(len(features)):
        # train_col is the index to compute mean and std-dev
        mean = np.mean(features[row][train_col])
        stddev = np.std(features[row][train_col])
        for i in range(len(features[row])):
            features[row][i] = (features[row][i] - mean) / stddev


def test_log_reg_warm_starting(features, labels, num_cs=50, max_iters=100):
    if len(features[0]) == 3:
        train_col, source_test_col, target_test_col = range(3)
    elif len(features[0]) == 4:
        train_col, source_test_col, target_test_col = 0, 2, 3 # 1 is target train
    else:
        raise ValueError(f'Not sure why there are {len(features[0])} design matrices, rather than 3 or 4.')
    M = len(features)
    all_accs = []
    for m in range(M):
        print(f'Now doing model number {m} of {M}')
        Cs = np.logspace(-7, 0, num_cs)
        clf = LogisticRegression(random_state=0, warm_start=True, max_iter=max_iters)
        #.fit(features[m][source_idx], labels[m][source_idx])
        source_accs, target_accs = [], []
        for C in Cs:
            clf.C = C
            clf.fit(features[m][train_col], labels[m][train_col])
            source_preds = clf.predict(features[m][source_test_col])
            source_acc = get_acc(source_preds, labels[m][source_test_col])
            target_preds = clf.predict(features[m][target_test_col])
            target_acc = get_acc(target_preds, labels[m][target_test_col])
            print('Source accuracy', source_acc, 'and target accuracy', target_acc)
            source_accs.append(source_acc)
            target_accs.append(target_acc)
        all_accs.append(source_accs)
        all_accs.append(target_accs)
    return all_accs


def main():
    parser = argparse.ArgumentParser(description='Train logistic regression model on features.')
    parser.add_argument('--load_path', type=str,
                        help='Pickle file where features, labels are saved', required=True)
    parser.add_argument('--save_path', type=str,
                        help='Path to save tsv results file')
    parser.add_argument('--train_data_frac', type=float, default=1.0,
                        help='The amount of source data to actually use for fine-tuning.')
    parser.add_argument('--num_reg_values', type=int,
                        help='Number of regularization values to sweep over.', required=False)
    args = parser.parse_args()
    features, labels, previous_args = pickle.load(open(args.load_path, 'rb'))
    print(f'Using representations from {previous_args.dataset} {previous_args.dataset_name}, using '
          f'model {previous_args.checkpoint_dir} {",".join(previous_args.checkpoint_names)}, '
          f'now using a training data fraction {args.train_data_frac}.')
    get_subsampled_features(features, labels, args.train_data_frac)
    normalize_features(features)
    max_iters = get_max_iters(previous_args.dataset_name)
    accs = test_log_reg_warm_starting(features, labels, num_cs=args.num_reg_values, max_iters=max_iters)
    accs_df = pd.DataFrame(accs)
    if args.save_path is not None:
        save_path = args.save_path
    else:
        directory = os.path.dirname(args.load_path)
        save_path = os.path.join(directory, f'sklearn_finetuning_fraction_{args.train_data_frac}.tsv')
    accs_df.to_csv(save_path, sep='\t')


if __name__ == "__main__":
    main()

