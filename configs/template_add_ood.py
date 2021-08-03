import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, required=True)
parser.add_argument('--ood_domain', type=str, required=True)
parser.add_argument('--ood_template', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.file_name, 'r') as f:
        file_lines = f.read()
    with open(args.ood_template, 'r') as f:
        ood_template = f.read()
    ood_template = ood_template.replace('OOD_DOMAIN_REPLACE', args.ood_domain)
    idx_to_add = file_lines.find('# ADD OOD HERE')
    file_lines = file_lines[:idx_to_add] + ood_template + file_lines[idx_to_add:]
    with open(args.file_name, 'w') as f:
        f.write(file_lines)