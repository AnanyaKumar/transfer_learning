import pickle
import pandas as pd
import numpy as np
import os
import io

def main(args):
    results = pd.DataFrame()
    ood_domain_string = ",".join(args.ood_domains)
    for data_frac in args.train_data_fracs:
        file_name = f'lin_probe_{args.file_name}_{args.id_domain}_{ood_domain_string}_{data_frac}.pickle'
        path = os.path.join(args.run_dir, 'finetuning', file_name)
        if not os.path.exists(path):
            print(f'No file at {path}. Skipping...')
            continue
        with open(path, 'rb') as f:
            id_accs, ood_accs = pickle.load(f)
        # find best in-domain accuracy
        best_id_reg = np.argmax(id_accs)
        result = {'data_frac': data_frac}
        result[f'id_{args.id_domain}_acc'] = id_accs[best_id_reg]
        print(f'File:\t{path}')
        print(f'ID ({args.id_domain}):\t{result[f"id_{args.id_domain}_acc"]}')
        for ood_domain in args.ood_domains:
            result[f'ood_{ood_domain}_acc'] = ood_accs[ood_domain][best_id_reg]
            print(f'OOD ({ood_domain}):\t{result[f"ood_{ood_domain}_acc"]}')
        results = results.append(result, ignore_index=True)
    print('Printing copy-and-pastable csv to the terminal...')
    with io.StringIO() as buffer:
        results.to_csv(buffer, sep='\t', index=False)
        print(buffer.getvalue())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Result summarizer')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Outer run directory to use.')
    parser.add_argument('--id_domain', type=str, required=True,
                        help='The source domain on which to train')
    parser.add_argument('--ood_domains', type=str, required=True,
                        help='The target domains on which to evaluate (comma-separated).')
    parser.add_argument('--file_name', type=str, required=True,
                        help='Name of the pickle file (without directories, without .pickle).')
    parser.add_argument('--train_data_fracs', type=float, nargs='+',
                        help='Train data fraction checkpoints to summarize.')
    args = parser.parse_args()
    args.ood_domains = args.ood_domains.split(',')
    main(args)
