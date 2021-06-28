import pandas as pd
import os
import io

def main(args):
    results = []
    for tsv_path in args.tsv_list:
        if not tsv_path.endswith('.tsv'):
            tsv_path = os.path.join(tsv_path, 'sklearn_finetuning.tsv')
        result = pd.read_csv(tsv_path, sep='\t')
        # find best in-domain accuracy
        id_idx = 0
        ood_idx = 1
        best_id_iterate = result.iloc[id_idx].argmax()
        id_acc = result.iloc[id_idx][best_id_iterate]
        ood_acc = result.iloc[ood_idx][best_id_iterate]
        print(f'TSV: \t {tsv_path}')
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
    parser.add_argument('tsv_list', type=str, nargs='+',
                        help='List of tsv paths to summarize')
    args = parser.parse_args()
    main(args)
