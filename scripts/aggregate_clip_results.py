from run_clip_experiment import DF_COLUMNS
from run_clip_experiment import make_experiment_dir
from run_clip_experiment import parse_args
import argparse
import pandas as pd
import shlex


DOMAIN_ORDER = ['real', 'clipart', 'painting', 'sketch']


def main(args):
    df = pd.DataFrame(columns=DF_COLUMNS)
    experiment_args = shlex.split(args.experiment_str)
    parsed_args = None
    if 'all' in args.source_domains:
        args.source_domains = DOMAIN_ORDER
    if 'all' in args.target_domains:
        args.target_domains = ['all']
    for src_domain in args.source_domains:
        for tgt_domain in args.target_domains:
            if parsed_args is None:
                experiment_args = [src_domain, tgt_domain] + experiment_args
                parsed_args = parse_args(experiment_args)
            else:
                parsed_args.source_domain = src_domain
                parsed_args.target_domain = tgt_domain

            experiment_dir = make_experiment_dir(parsed_args)
            if args.print_folders:
                print(experiment_dir)
            experiment_df = pd.read_pickle(experiment_dir / 'results.pkl')
            df = pd.concat((df, experiment_df))

    df = df.drop_duplicates()
    df.SourceDomain = pd.Categorical(df.SourceDomain, DOMAIN_ORDER)
    df.TargetDomain = pd.Categorical(df.TargetDomain, DOMAIN_ORDER)
    df = df.sort_values(['SourceDomain', 'TargetDomain'])
    if not args.include_source_val:
        df = df[df.SourceDomain != df.TargetDomain]
    if not args.include_domain_names:
        df = df.drop(columns=['SourceDomain', 'TargetDomain'])
    print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate CLIP Experiments')
    parser.add_argument('source_domains', nargs='+', default='all',
                        choices=DOMAIN_ORDER + ['all'],
                        help='Source domains of results to aggregate')
    parser.add_argument('target_domains', nargs='+', default='all',
                        choices=DOMAIN_ORDER + ['all'],
                        help='Target domains of results to aggregate')
    parser.add_argument('experiment_str', type=str,
                        help='Rest of args passed to run_clip_experiment.py')
    parser.add_argument('--print_folders', action='store_true',
                        help='Print folders of results in aggregation')
    parser.add_argument('--include_source_val', action='store_true',
                        help='Include validation results for source domains')
    parser.add_argument('--include_domain_names', action='store_true',
                        help='Print domain names')

    args = parser.parse_args()
    main(args)
