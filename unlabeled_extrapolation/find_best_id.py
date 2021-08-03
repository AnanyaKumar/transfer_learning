import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--stats_tsvs', type=str, nargs='+', required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    for stats_tsv in args.stats_tsvs:
        print('*' * 30)
        print('Now doing', stats_tsv)
        stats = pd.read_csv(stats_tsv, sep='\t')
        best_epoch = stats['test_acc/id_test'].argmax()
        columns = list(stats.columns)
        print('Best epoch based on ID test accuracy:', best_epoch)
        for col in columns:
            if col.startswith('test_acc/'):
                print(col, '\t', stats[col][best_epoch])