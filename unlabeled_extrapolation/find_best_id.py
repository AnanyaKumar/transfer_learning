import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('stats_tsvs', type=str, nargs='+')

if __name__ == '__main__':
    args = parser.parse_args()
    for stats_tsv in args.stats_tsvs:
        print('*' * 30)
        print('Now doing', stats_tsv)
        print('Command:')
        with open(os.path.join(os.path.dirname(stats_tsv), 'command.txt'), 'r') as f:
            cmd = f.read()
            print(cmd)
        stats = pd.read_csv(stats_tsv, sep='\t')
        best_epoch = stats['test_acc/id_test'].argmax()
        columns = list(stats.columns)
        print('Best epoch based on ID test accuracy:', best_epoch)
        for col in columns:
            if col.startswith('test_acc/'):
                print(col, '\t', stats[col][best_epoch])
