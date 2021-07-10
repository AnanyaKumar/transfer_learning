import pandas as pd
import torch
import os
import io

from unlabeled_extrapolation.datasets import breeds

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test_between', required=True, choices=['classes', 'domains'])
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--target', required=True)
parser.add_argument('--data_aug', required=True, choices=['imagenet', 'simclr'])


def validate_args(args):
    if args.dataset_name == 'breeds':
        if args.source != args.target:
            raise ValueError('Source and target must be the same for Breeds tasks.')
    elif args.dataset== 'domainnet':
        if args.source == args.target:
            raise ValueError('Source and target should be different for DomainNet.')
    else:
        raise NotImplementedError('Unsupported dataset.')


def get_class_mapping(args):
    if args.dataset_name == 'breeds':
        breeds_dataset = breeds.Breeds('/scr/biggest/imagenet', args.source)
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
    validate_args(args)
    class_mapping = get_class_mapping(args)
    results = []
    if args.test_between == 'domains':
        for class_idx in range(len(class_mapping)):
            file_name = f'connectivity_checkpoints/{args.source}-{args.target}-domains-{args.data_aug}-class-idx-{class_idx}-final'
            try:
                data = torch.load(file_name, map_location=torch.device('cpu'))
                final_train_acc = data['train_accs'][-1]
                final_test_acc = data['test_accs'][-1]
                results.append([class_idx, class_mapping[class_idx], final_train_acc, final_test_acc])
            except:
                print('Failed at', file_name, '. Skipping...')
    else:
        this_run = f'{args.source}-{args.target}-classes-{args.data_aug}-class-idxes'
        format_str = '-{}-{}-final'
        for file_name in os.listdir('connectivity_checkpoints'):
            if not (file_name.startswith(this_run + '-source') and file_name.endswith('-final')):
                continue
            tokens = file_name.split('-')
            class_1 = tokens[-3]
            class_2 = tokens[-2]
            print('Doing', file_name)
            source_data = torch.load(os.path.join('connectivity_checkpoints', file_name), map_location=torch.device('cpu'))
            final_source_train_acc = source_data['train_accs'][-1]
            final_source_test_acc = source_data['test_accs'][-1]
            target_file_name = this_run + '-target' + format_str.format(class_1, class_2)
            print('Doing now,', target_file_name)
            target_data = torch.load(os.path.join('connectivity_checkpoints', target_file_name), map_location=torch.device('cpu'))
            final_target_train_acc = target_data['train_accs'][-1]
            final_target_test_acc = target_data['test_accs'][-1]
            results.append([class_1, class_mapping[int(class_1)], class_2, class_mapping[int(class_2)],
                final_source_train_acc, final_source_test_acc, final_target_train_acc, final_target_test_acc])
    df = pd.DataFrame(results)
    print('Printing copy-and-pastable csv to the terminal...')
    with io.StringIO() as buffer:
        df.to_csv(buffer, sep='\t', index=False)
        print(buffer.getvalue())

