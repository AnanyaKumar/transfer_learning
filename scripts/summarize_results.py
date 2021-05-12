import argparse
import pandas as pd
import numpy as np
import json
import subprocess
from pathlib import Path
from collections import OrderedDict
import shlex
import uuid
import re
import glob
from pathlib import Path

def get_result(file_path, val_metric, output_metrics, take_max=True):
    df = pd.read_csv(file_path, sep='\t')
    if val_metric is not None:
        if val_metric not in df:
            raise ValueError(f'{val_metric} column not in {file_path}')
        if take_max:
            best_idx = df[val_metric].idxmax()
        else:
            best_idx = df[val_metric].idxmin()
    else:
        best_idx = -1
    best_res = df.iloc[best_idx].to_dict()
    if output_metrics is not None:
        best_res = [(f"{k}", best_res[k]) for k in output_metrics]
    else:
        best_res = [(f"{k}", v) for k,v in best_res.items()]
    # Get run-name and wandb from config file.
    parent_dir = Path(file_path).parent.absolute()
    with open(parent_dir / 'config.json', 'r') as f:
        config = json.load(f)
    run_name = [('name', config['run_name'])]
    wandb_url = [('wandb_url', config['wandb_url'])]
    best_res = run_name + best_res + wandb_url
    return best_res

def summarize_results(args):
    # Returns a list of rows, each row is the result for a corresponding file.
    file_paths = glob.glob(args.results_dirs + '/**/stats.tsv', recursive=True)
    file_paths = list(set(file_paths))
    file_paths = sorted(file_paths)
    results_list = []
    for file_path in file_paths:
        res_row = get_result(file_path, args.val_metric, args.output_metrics)
        results_list.append(OrderedDict(res_row))
    res = pd.DataFrame(results_list)
    for col in res.columns:
        if 'epoch' in col:
            res = res.astype({col: 'int32'})
        if 'acc' in col:
            res[col] = res[col] * 100
    res = res.round(4)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read logs')
    parser.add_argument('--results_dirs', type=str, required=True,
                        help="Folders where we should look for tsv files (glob syntax)")
    parser.add_argument('--val_metric', type=str,
                        help='Metric to use to pick best row, if unspecified return last row.')
    parser.add_argument('--output_metrics', type=str, nargs='+',
                        help='Metrics to output.')
    parser.add_argument('--output_file', type=str,
                        help='Path to output results, should end with .tsv')
    parser.add_argument('-s', action='store_true', help="Short version: Do not show column names")
    args = parser.parse_args()
    results = summarize_results(args)
    if args.output_file is None:
        output_file = str('/tmp/' + f'res_{str(uuid.uuid4())[:4]}.tsv')
    else:
        if args.output_file[-4:] != '.tsv':
            raise ValueError('--output_file should end with .tsv')
        output_file = args.output_file
    if args.s:
        results.to_csv(output_file, sep='\t', index=None, header=False)
    else:
        results.to_csv(output_file, sep='\t', index=None)
    print('Saved results to ' + output_file)

