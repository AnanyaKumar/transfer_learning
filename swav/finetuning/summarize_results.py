import pandas as pd
import os
import io

def main(args):
    results = []
    for data_frac in args.train_data_fracs:
        path = os.path.join(args.tsv_dir, f'sklearn_finetuning_fraction_{data_frac}.tsv')
        result = pd.read_csv(path, sep='\t')
        # find best in-domain accuracy
        id_idx = 0
        ood_idx = 1
        best_id_iterate = result.iloc[id_idx].argmax()
        id_acc = result.iloc[id_idx][best_id_iterate]
        ood_acc = result.iloc[ood_idx][best_id_iterate]
        print(f'TSV: \t {path}')
        print(f'ID: \t {id_acc}')
        print(f'OOD: \t {ood_acc}')
        results.append([id_acc, ood_acc])
    df = pd.DataFrame(results)
    print('Printing copy-and-pastable csv to the terminal...')
    with io.StringIO() as buffer:
        df.to_csv(buffer, sep='\t', index=False)
        print(buffer.getvalue())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Result summarizer')
    parser.add_argument('tsv_dir', type=str,
                        help='Directory of tsv\'s to summarize.')
    parser.add_argument('train_data_fracs', type=float, nargs='+',
                        help='Train data fraction checkpoints to summarize.')
    args = parser.parse_args()
    main(args)
