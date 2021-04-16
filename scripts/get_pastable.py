import pandas as pd
import numpy as np
import json
import subprocess
from pathlib import Path
from collections import OrderedDict
import shlex
import uuid
import re


def get_res(model_dir, val_metric='val_accuracy', take_max=True, metrics=None,
            config_keys=None, ignore_missing=False):
    '''
    Args:
        model_dir: str
            where the model is saved
        val_metric: str
            name of the validation metric to do model selection with
        take_max: bool
            take the max of the val_metric. Otherwise take min
        metrics: Optional[List[str]]
            list of metrics to keep from the results. Otherwise keep all.
        config_keys: Optional[List[str]]
            list of keys to keep from the config. Otherwise keep name and wandb_url.
    '''
    model_dir = Path(model_dir)
    with open(model_dir / 'config.json', 'r') as f:
        config = json.load(f)

    if config_keys is None:
        config_keys = ['run_name', 'wandb_url']

    config_list = [(k, config[k]) for k in config_keys]

    res_path = model_dir / 'stats_eval.tsv'
    try:
        df = pd.read_csv(res_path, sep='\t')
    except FileNotFoundError as err:
        if ignore_missing:
            return True
        raise err

    # check if metrics in df
    for k in metrics:
        if k not in df.columns:
            return False

    if take_max:
        best_idx = df[val_metric].idxmax()
    else:
        best_idx = df[val_metric].idxmin()

    best_res = df.iloc[best_idx].to_dict()
    last_res = df.iloc[-1].to_dict()
    # reduce to accuracy metrics
    if metrics is not None:
        best_res = [(f"{k}_best", best_res[k]) for k in metrics]
        last_res = [(f"{k}_last", last_res[k]) for k in metrics]
    else:
        best_res = [(f"{k}_best", v) for k,v in best_res.items()]
        last_res = [(f"{k}_last", v) for k,v in last_res.items()]
    all_res = OrderedDict(config_list + last_res + best_res)

    res.append(all_res)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Read logs')
    parser.add_argument('-s', action='store_true', help="Short version: Do not show column names")
    parser.add_argument('-e', type=str, default=None, help="exclude run names with this suffix")
    parser.add_argument('-i', type=str, default=None, help="include run names with this suffix. includes first, then excludes.")
    parser.add_argument('-a', action='store_true', help='Average over trials')
    parser.add_argument('-d', action='store_true', help='stdev over trials')
    parser.add_argument('-c', action='store_true', help='Include AUC metrics')
    parser.add_argument('--ignore_missing', action='store_true',
                        help='Ignore runs with missing eval stats')
    parser.add_argument('results_dirs', nargs='*')
    args = parser.parse_args()


    res = []
    for results_dir in args.results_dirs:
        metric_type = 'accuracy'
        metrics = ['epoch', f'train_{metric_type}', f'val_{metric_type}', f'test_{metric_type}', f'test2_{metric_type}']
        if args.c:
            metrics.extend(['train_auroc', 'val_auroc', 'test_auroc', 'test2_auroc'])
            metrics.extend(['train_auprc', 'val_auprc', 'test_auprc', 'test2_auprc'])
        if get_res(results_dir, metrics=metrics, config_keys=None,
                   ignore_missing=args.ignore_missing):
            continue
        else:
            try:
                metric_type = 'loss'
                metrics = ['epoch', f'train_{metric_type}', f'val_{metric_type}', f'test_{metric_type}', f'test2_{metric_type}']
                metric_type = 'r2'
                metrics += ['epoch', f'train_{metric_type}', f'val_{metric_type}', f'test_{metric_type}', f'test2_{metric_type}']
                ret = get_res(results_dir, metrics=metrics, config_keys=None, val_metric='val_loss', take_max=False)
            except:
                metric_type = 'loss'
                metrics = ['epoch', f'train_{metric_type}', f'val_{metric_type}', f'test_{metric_type}', f'test2_{metric_type}']
                ret = get_res(results_dir, metrics=metrics, config_keys=None, val_metric='val_loss', take_max=False)
            if not ret:
                raise ValueError(f"Could not read logs at {results_dir}")

    res = pd.DataFrame(res)
    for col in res.columns:
        if 'epoch' in col:
            res = res.astype({col: 'int32'})
        if 'accuracy' in col:
            res[col] = res[col] * 100
    res = res.round(4)

    if args.i:
        include_mask = [run_name.endswith(args.i) for run_name in res['run_name']]
        res = res[include_mask]
    elif args.e:
        include_mask = [(not run_name.endswith(args.e)) for run_name in res['run_name']]
        res = res[include_mask]

    res = res.dropna(how='all', axis=1)
    if args.a or args.d:
        trial_res = res[res.run_name.str.contains('trial\d+', case=False)]
        run_names = trial_res.run_name.str.extract('(.*)trial\d+(.*)', re.I)
        run_names = run_names[0] + run_names[1]
        trial_res = trial_res.assign(run_name=run_names)
        trial_groups = trial_res.groupby(['run_name'])
        if args.a:
            res = trial_groups.mean().reset_index()
        elif args.d:
            res = (trial_groups.std() / np.sqrt(5) * 1.645).reset_index()
        res = res.assign(num_trials=trial_groups.size().reset_index()[0])

    tmp_file_dir = Path('/tmp')
    tmp_file = str(tmp_file_dir / f'res_{str(uuid.uuid4())[:4]}.tsv')

    if args.s:
        res.to_csv(tmp_file, sep='\t', index=None, header=False)
    else:
        res.to_csv(tmp_file, sep='\t', index=None)
    with open(tmp_file, 'r') as f:
        for line in f:
            print(line.strip())
    cmd = f'xclip -sel clip {tmp_file}'
    subprocess.run(shlex.split(cmd))
    print(f"Results saved at {tmp_file}")
    Path(tmp_file).chmod(0o777)
