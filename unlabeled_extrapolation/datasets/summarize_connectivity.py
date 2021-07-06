import pandas as pd
import torch
import os
import io

from unlabeled_extrapolation.datasets import breeds

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test_between', required=True, choices=['classes', 'domains'])
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--data_aug', required=True, choices=['imagenet', 'simclr'])

if __name__ == '__main__':
    args = parser.parse_args()
    breeds_dataset = breeds.Breeds('/scr/biggest/imagenet', args.dataset_name)
    class_mapping = breeds_dataset._label_map
    results = []
    if args.test_between == 'domains':
        for class_idx in range(len(class_mapping)):
            file_name = f'connectivity_checkpoints/{args.dataset_name}-{args.dataset_name}-domains-{args.data_aug}-class-idx-{class_idx}-final'
            try:
                data = torch.load(file_name, map_location=torch.device('cpu'))
                final_train_acc = data['train_accs'][-1]
                final_test_acc = data['test_accs'][-1]
                results.append([class_idx, class_mapping[class_idx], final_train_acc, final_test_acc])
            except:
                print('Failed at', file_name, '. Skipping...')
    else:
        this_run = args.dataset_name + '-' + args.dataset_name + '-classes-' + args.data_aug + '-class-idxes'
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

