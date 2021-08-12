from run_clip_experiment import EXPERIMENT_TYPES
from run_clip_experiment import RESULTS_DIR
from run_clip_experiment import parse_args
from run_clip_experiment import init_results_fields
import argparse
import pandas as pd
import shlex


DOMAIN_ORDER = ['real', 'clipart', 'painting', 'sketch']


def main(args):
    results_filepath = RESULTS_DIR / f'{args.experiment_type}.pkl'
    df = pd.read_pickle(results_filepath)
    experiment_args = shlex.split(args.experiment_str)
    parsed_args = parse_args(experiment_args, optionals_only=True)
    results_fields = init_results_fields(parsed_args, args.experiment_type,
                                         include_date=False)
    if 'all' in args.source_domains:
        args.source_domains = DOMAIN_ORDER
    if 'all' in args.target_domains:
        args.target_domains = DOMAIN_ORDER

    df = df[df.source_domain.isin(args.source_domains)]
    df = df[df.target_domain.isin(args.target_domains)]
    df = df[df.model == args.model]
    mask = (df[list(results_fields)] == pd.Series(results_fields)).any(axis=1)
    df = df.loc[mask]
    df = df.loc[df.astype(str).drop_duplicates().index]
    df.source_domain = pd.Categorical(df.source_domain, DOMAIN_ORDER)
    df.target_domain = pd.Categorical(df.target_domain, DOMAIN_ORDER)
    df = df.sort_values(['source_domain', 'target_domain'])
    if not args.include_source_val:
        df = df[df.source_domain != df.target_domain]
    columns = ['per_class_avg_acc']
    if args.include_domain_names:
        columns = ['source_domain', 'target_domain'] + columns
    print(df[columns].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate CLIP Experiments')
    parser.add_argument('experiment_type', type=str, choices=EXPERIMENT_TYPES,
                        help='CLIP experiment type')
    parser.add_argument('model', type=str, help='CLIP model')
    parser.add_argument('experiment_str', type=str,
                        help='Rest of args passed to run_clip_experiment.py')
    parser.add_argument('--source_domains', nargs='*', default=['all'],
                        choices=DOMAIN_ORDER + ['all'],
                        help='Source domains of results to aggregate')
    parser.add_argument('--target_domains', nargs='*', default=['all'],
                        choices=DOMAIN_ORDER + ['all'],
                        help='Target domains of results to aggregate')
    parser.add_argument('--print_folders', action='store_true',
                        help='Print folders of results in aggregation')
    parser.add_argument('--include_source_val', action='store_true',
                        help='Include validation results for source domains')
    parser.add_argument('--include_domain_names', action='store_true',
                        help='Print domain names')

    args = parser.parse_args()
    main(args)
