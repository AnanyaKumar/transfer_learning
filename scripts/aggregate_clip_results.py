from run_clip_experiment import DF_COLUMNS
from run_clip_experiment import RESULTS_DIR
from run_clip_experiment import EXPERIMENT_TYPES
from run_clip_experiment import ParseKwargs
import argparse
import clip
import pandas as pd


DOMAIN_ORDER = ['real', 'clipart', 'painting', 'sketch']


def main(args):
    experiment = args.experiment_type.replace('-', '')
    model = args.model.replace('/', '')
    results_folders = RESULTS_DIR.glob(f'clip_domainnet_{model}_*_{experiment}')
    for key, value in args.experiment_kwargs.items():
        filter_key = f'{key.replace("_", "")}{value}'
        results_folders = filter(
            lambda folder, filter_key=filter_key: filter_key in str(folder),
            results_folders
        )

    domains = args.target_domains
    results_folders = filter(
        lambda folder: any(f'target{domain}' in str(folder) for domain in domains),
        results_folders
    )
    if args.translate_features:
        results_folders = filter(
            lambda folder: '_translatefeats' in str(folder),
            results_folders
        )
    else:
        results_folders = filter(
            lambda folder: '_translatefeats' not in str(folder),
            results_folders
        )

    df = pd.DataFrame(columns=DF_COLUMNS)
    for folder in results_folders:
        if args.print_folders:
            print(folder)
        df = pd.concat((df, pd.read_pickle(folder / 'results.pkl')))

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
    parser.add_argument('experiment_type', choices=EXPERIMENT_TYPES,
                        help='Experiment type of results to aggregate')
    parser.add_argument('model', type=str,
                        choices=clip.available_models(),
                        help='CLIP Model')
    parser.add_argument('target_domains', nargs='*', default='all',
                        choices=DOMAIN_ORDER + ['all'],
                        help='Target domains of results to aggregate')
    parser.add_argument('--print_folders', action='store_true',
                        help='Print folders of results in aggregation')
    parser.add_argument('--translate_features', action='store_true',
                        help='Experiment used feature translation')
    parser.add_argument('--include_source_val', action='store_true',
                        help='Include validation results for source domains')
    parser.add_argument('--include_domain_names', action='store_true',
                        help='Print domain names')
    parser.add_argument('--experiment_kwargs', nargs='*',
                        action=ParseKwargs, default={},
                        help='Other fields to filter results')

    args = parser.parse_args()
    main(args)
