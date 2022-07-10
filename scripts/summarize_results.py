import argparse
import pandas as pd
import numpy as np
import json
import subprocess
from pathlib import Path
from collections import OrderedDict
import os
import shlex
import shutil
import uuid
import re
import glob
from pathlib import Path


def get_parent_folder(file_path):
    parent_path = file_path[:file_path.rfind('/')]
    parent_folder = parent_path[parent_path.rfind('/')+1:]
    return parent_folder


def get_result(file_path, val_metric, output_metrics, take_max=True):
    df = pd.read_csv(file_path, sep='\t')
    # The user probably wants us to output the val metric!
    if val_metric not in output_metrics and val_metric != 'LAST':
        output_metrics += val_metric
    # If needed, compute the worst among all the test_accs.
    # That is, if the user is selecting for this metric or wants it displayed.
    # I guess val_metric == 'WORST' is redundant given the previous line which will
    # add it to output_metrics, but just keeping it here in case I fiddle around with
    # the previous line later.
    if val_metric == 'WORST' or 'WORST' in output_metrics:
        test_acc_column_names = [s for s in df.columns if 'test_acc' in s]
        worst_accs = df[test_acc_column_names].min(axis=1)
        df['WORST'] = worst_accs
    if val_metric is not None and val_metric != 'LAST':
        if val_metric not in df:
            raise ValueError(f'{val_metric} column not in {file_path}')
        if take_max:
            best_idx = df[val_metric].idxmax()
        else:
            best_idx = df[val_metric].idxmin()
        best_value = df[val_metric][best_idx]
    else:
        best_idx = -1
        best_value = -1.0
    best_res = df.iloc[best_idx].to_dict()
    if output_metrics is not None:
        best_res = [(f"{k}", best_res[k]) for k in output_metrics]
    else:
        best_res = [(f"{k}", v) for k,v in best_res.items()]
    # Get run-name and wandb from config file.
    parent_dir = Path(file_path).parent.absolute()
    config_path = parent_dir / 'config.json'
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        run_name = [('name', config['run_name'])]
        wandb_url = [('wandb_url', config['wandb_url'])]
        best_res = run_name + best_res + wandb_url
    else:
        run_name = [('name', file_path[file_path.rfind('/')+1:-4])]
        best_res = run_name + best_res
    return best_res, best_value

def summarize_results(results_dir, val_metric, output_metrics, replication=False, use_all=False, max_num=None):
    # If use_all is True, then look at all stats.tsv files.
    # Otherwise, if replication is True look for replication files, if false ignore replicatin files.
    # max_num is for linear probing, and uses only the first max_num runs.
    # Returns a list of rows, each row is the result for a corresponding file.
    file_paths = glob.glob(results_dir + '/**/*stats*.tsv', recursive=True)
    file_paths = list(set(file_paths))
    # Sort so that the results are deterministic, e.g. if multiple runs have
    # the same validation metric.
    file_paths = sorted(file_paths)
    results_list, val_values_list = [], []
    for file_path in file_paths:
        if max_num is not None and int(file_path[-5]) >= max_num:
            continue
        parent_folder = get_parent_folder(file_path)
        if use_all or (('replication' in parent_folder and replication) or
                       ('replication' not in parent_folder and not replication)):
            res_row, val_value = get_result(file_path, val_metric, output_metrics)
            results_list.append(OrderedDict(res_row))
            val_values_list.append(val_value)
    res = pd.DataFrame(results_list)
    for col in res.columns:
        if 'epoch' in col:
            res = res.astype({col: 'int32'})
        if 'acc' in col:
            res[col] = res[col] * 100
    res = res.round(4)
    return res, val_values_list, file_paths

if __name__ == '__main__':
    # Higher is better when we pick val_metric, so use it for accuracies not losses.
    parser = argparse.ArgumentParser(description='Read logs')
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Folders where we should look for tsv files (glob syntax)")
    parser.add_argument('--val_metric', type=str,
                        help='Metric to use to pick best (higher better) row and checkpoint.')
    parser.add_argument('--secondary_val_metrics', type=str, nargs='+',
                        help='Additional metrics to report best row for.')
    parser.add_argument('--output_metrics', type=str, nargs='+',
                        help='Metrics to output.')
    parser.add_argument('--output_file', type=str,
                        help='Path to output results, should end with .tsv')
    parser.add_argument('-s', action='store_true', help="Short version: Do not show column names")
    args = parser.parse_args()
    # Get results for val_metric.
    results, val_values, file_paths = summarize_results(
        args.results_dir, args.val_metric, args.output_metrics)
    # Get index of the best run, and save the config file.
    best_run_idx = np.argmax(val_values)
    best_config_path = os.path.dirname(file_paths[best_run_idx]) + '/config.json'
    config_save_path = args.results_dir + '/best_config.json'
    shutil.copy(best_config_path, config_save_path)
    # Get replication summary (could be empty).
    replication_results, _, _ = summarize_results(
        args.results_dir, args.val_metric, args.output_metrics,
        replication=True)
    # Get other results.
    secondary_results_list = []
    for secondary_val_metric in args.secondary_val_metrics:
        secondary_results, _, _ = summarize_results(
            args.results_dir, secondary_val_metric, args.output_metrics)
        secondary_results_list.append(secondary_results)
    # Save tsv file.
    output = f'Results for {args.results_dir}, early stopped on {args.val_metric}:\n\n'
    if args.s:
        output += results.to_csv(sep='\t', index=None, header=False)
    else:
        output += results.to_csv(sep='\t', index=None)
    for secondary_metric, secondary_results in zip(
        args.secondary_val_metrics, secondary_results_list):
        output += '\n'
        output += f'Early stopped based on {secondary_metric}:\n'
        output += secondary_results.to_csv(sep='\t', index=None, header=False)
    if len(replication_results) > 0:
        output += '\n'
        output += 'Replication results:\n'
        output += replication_results.to_csv(sep='\t', index=None, header=False)
        output += '\n'
        output += 'Replication average and 90% confidence interval:\n'
        output += pd.DataFrame(replication_results.mean(axis=0)).T.to_csv(
            sep='\t', index=None, header=True)
        multiplier = 1.645 / np.sqrt(len(replication_results))
        output += pd.DataFrame(multiplier * replication_results.std(axis=0)).T.to_csv(
            sep='\t', index=None, header=False)

    if args.output_file is None:
        output_file = args.results_dir + '/results.tsv'
        # output_file = str('/tmp/' + f'res_{str(uuid.uuid4())[:4]}.tsv')
    else:
        if args.output_file[-4:] != '.tsv':
            raise ValueError('--output_file should end with .tsv')
        output_file = args.output_file
    with open(output_file, "w") as write_file:
        write_file.write(output)
    print('Saved results to ' + output_file)

