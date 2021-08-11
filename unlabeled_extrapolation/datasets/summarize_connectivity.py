import pandas as pd
import torch
import os
import io

from unlabeled_extrapolation.datasets.connectivity_utils import *
from unlabeled_extrapolation.datasets import breeds

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Which dataset on which to test connectivity.')
parser.add_argument('--source', type=str, required=True,
                    help='Name of source dataset')
parser.add_argument('--target', type=str, default=None,
                    help='Name of target dataset. If not provided, will be set to the '
                    'same as --source')
parser.add_argument('--test_between', required=True, choices=['classes', 'domains'])
parser.add_argument('--transform', type=str, choices=['imagenet', 'simclr'], default='simclr')
parser.add_argument('--num_iters', default=15, type=int,
                    help='If doing class-comparison, the number of random pairs to choose.')
parser.add_argument('--seed', default=20, type=int, help='Seed for choosing pairs of classes.')

# if linear probing
parser.add_argument('--swav_dir', type=str,
                    help='If provided, will use the checkpoint in this directory.')
parser.add_argument('--swav_ckpt', type=str, default='ckp-399.pth',
                    help='The name of the checkpoint to use.')
# other arg
parser.add_argument('--fail_if_incomplete', action='store_true')

def get_class_mapping(args):
    if args.dataset_name == 'breeds':
        breeds_dataset = breeds.Breeds('/u/scr/nlp/imagenet', args.source)
        class_mapping = breeds_dataset._label_map
    elif args.dataset_name == 'domainnet':
        with open('/u/scr/nlp/domainnet/SENTRY_splits/classnames.txt', 'r') as f:
            class_names = f.readlines()
        class_mapping = {idx: item.strip() for idx, item in enumerate(class_names)}
    else:
        raise NotImplementedError('Not implemented.')
    return class_mapping

if __name__ == '__main__':
    args = parser.parse_args()
    if args.target is None:
        args.target = args.source
    args.linear_probe = (args.swav_dir is not None) and (args.swav_ckpt is not None)
    save_dir = os.path.join(
        'connectivity_checkpoints',
        f'{args.source}-{args.target}',
        f'{args.test_between}-{args.transform}'
    )
    if args.linear_probe:
        save_dir += f'-{args.swav_dir}-{args.swav_ckpt}'
    class_mapping = get_class_mapping(args)
    results = []
    if args.test_between == 'domains':
        for class_idx in range(len(class_mapping)):
            file_name = os.path.join(save_dir, f'class-{class_idx}-final')
            try:
                data = torch.load(file_name, map_location=torch.device('cpu'))
                final_train_acc = data['train_accs'][-1]
                final_test_acc = data['test_accs'][-1]
                results.append([class_idx, class_mapping[class_idx], final_train_acc, final_test_acc])
            except FileNotFoundError:
                print('Failed at', file_name, '. Skipping...')
                if args.fail_if_incomplete:
                    raise Exception('Incomplete. Exiting since --fail_if_incomplete was used...')
    else:
        identifier = '{}-classes-{}-{}-final'
        class_pairs = get_classes_to_compare(len(class_mapping), args.num_iters, args.seed)
        for class_1, class_2 in class_pairs:
            source_path = os.path.join(save_dir, identifier.format('source', class_1, class_2))
            source_done = os.path.exists(source_path)
            target_path = os.path.join(save_dir, identifier.format('target', class_1, class_2))
            target_done = os.path.exists(target_path)
            if args.fail_if_incomplete and not (source_done and target_done):
                raise Exception('Incomplete. Exiting since --fail_if_incomplete was used...')
            try:
                if source_done:
                    print('Doing', source_path)
                    source_data = torch.load(source_path, map_location=torch.device('cpu'))
                    final_source_train_acc = source_data['train_accs'][-1]
                    final_source_test_acc = source_data['test_accs'][-1]
                else:
                    final_source_train_acc = final_source_test_acc = float('nan')
                if target_done:
                    print('Doing now,', target_path)
                    target_data = torch.load(target_path, map_location=torch.device('cpu'))
                    final_target_train_acc = target_data['train_accs'][-1]
                    final_target_test_acc = target_data['test_accs'][-1]
                else:
                    final_target_train_acc = final_target_test_acc = float('nan')
            except RuntimeError:
                print('Failed at', class_1, class_2, 'for unknown error.')
                final_source_train_acc = final_source_test_acc = \
                    final_target_train_acc = final_target_test_acc = float('nan')
            results.append([class_1, class_mapping[int(class_1)], class_2, class_mapping[int(class_2)],
                            final_source_train_acc, final_source_test_acc, final_target_train_acc, final_target_test_acc])
    df = pd.DataFrame(results)
    df.loc['mean'] = df.mean()
    print('Printing copy-and-pastable csv to the terminal...')
    with io.StringIO() as buffer:
        df.to_csv(buffer, sep='\t')
        print(buffer.getvalue())

