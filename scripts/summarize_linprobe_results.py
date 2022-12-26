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
import sys
import uuid
import re
import glob
from pathlib import Path

from summarize_results import get_result, summarize_results

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
    parser.add_argument('--max_num', type=int,
                        help='Max number of lin probe runs to use.')
    parser.add_argument('-s', action='store_true', help="Short version: Do not show column names")
    parser.add_argument('--one_experiment_json', action='store_true', help="Summarize single experiment in json")
    args = parser.parse_args()
    metrics, results_list = [], []
    # Get results for val_metric.
    val_results, _, _, _ = summarize_results(
        args.results_dir, args.val_metric, args.output_metrics,
        max_num=args.max_num)
    if args.one_experiment_json:
        def codalab_reformat(s):
          return s.replace('/', '_')
        new_columns = [codalab_reformat(s) for s in val_results.columns]
        val_results.columns = new_columns
        json_output = val_results.iloc[0].to_json()
        json_file = open(args.results_dir + '/results.json', "w")
        json_file.write(json_output)
        sys.exit(0)

    metrics.append(args.val_metric)
    results_list.append(val_results)
    # Get other results.
    if args.secondary_val_metrics is None:
        args.secondary_val_metrics = []
    for secondary_val_metric in args.secondary_val_metrics:
        secondary_results, _, _ = summarize_results(
            args.results_dir, secondary_val_metric, args.output_metrics, use_all=True, max_num=args.max_num)
        metrics.append(secondary_val_metric)
        results_list.append(secondary_results)
    # Save tsv file
    output = ''
    for metric, results in zip(metrics, results_list):
        output += '\n'
        output += f'Replication results when early stopped based on {metric}:\n'
        output += results.to_csv(sep='\t', index=None, header=True)
        output += '\n'
        output += 'Replication average and 90% confidence interval:\n'
        output += pd.DataFrame(results.mean(axis=0)).T.to_csv(
            sep='\t', index=None, header=True)
        multiplier = 1.645 / np.sqrt(len(results))
        output += pd.DataFrame(multiplier * results.std(axis=0)).T.to_csv(
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

