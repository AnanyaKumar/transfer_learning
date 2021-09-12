# Unlike summarize results, this summarizes results for many experiments at once.
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

from summarize_results import get_result, summarize_results

if __name__ == '__main__':
    # Higher is better when we pick val_metric, so use it for accuracies not losses.
    parser = argparse.ArgumentParser(description='Read logs')
    parser.add_argument('--results_dir_glob', type=str, required=True,
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
    # Get all folders satisfying glob.
    
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

