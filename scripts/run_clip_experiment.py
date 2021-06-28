import argparse
import clip
import torch
import numpy as np
import pandas as pd
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from unlabeled_extrapolation.datasets.domainnet import DomainNet


DOMAINNET_ROOT = '/scr/biggest/domainnet'
MODELS = ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']
DOMAINS = ['real', 'clipart', 'painting', 'sketch']


def get_features(dataset, model):
    all_features = []
    all_labels = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=128)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    features = torch.cat(all_features).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    return features, labels


def run_exp(model_name, C):
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(model_name, device)
    print(f'Running experiments for model {model_name}')
    # Load the datasets
    train_features_labels = {}
    test_features_labels = {}
    for domain in DOMAINS:
        train_dataset = DomainNet(domain, root=DOMAINNET_ROOT,
                                  transform=preprocess)
        test_dataset = DomainNet(domain, split='test', root=DOMAINNET_ROOT,
                                 transform=preprocess)
        train_features_labels[domain] = get_features(train_dataset, model)
        test_features_labels[domain] = get_features(test_dataset, model)

    columns = ('SourceDomain', 'TargetDomain', 'AvgPerClassAcc', 'Acc')
    df = pd.DataFrame(columns=columns)
    for source_domain in DOMAINS:
        train_features, train_labels = train_features_labels[source_domain]
        classifier = LogisticRegression(random_state=0, C=C, max_iter=1000)
        classifier.fit(train_features, train_labels)
        for target_domain in DOMAINS:
            test_features, test_labels = test_features_labels[target_domain]
            preds = classifier.predict(test_features)
            cm = confusion_matrix(test_labels, preds)
            per_class_correct = np.diag(cm)
            per_class_avg_acc = np.mean(per_class_correct / cm.sum(axis=1))
            accuracy = per_class_correct.sum() / preds.size
            print(f'{source_domain} -> {target_domain}',
                  f' Per-Class Avg. Acc. = {100. * per_class_avg_acc:.3f}',
                  f' Overall Acc. = {100. * accuracy:.3f}')
            df.loc[len(df)] = (source_domain, target_domain,
                               per_class_avg_acc, accuracy)

    results_dir = pathlib.Path(__file__).parent.resolve().parent / 'results'
    experiment_name = f'clip_domainnet_C{C}_{model_name}_probe'
    experiment_name = experiment_name.replace('/', '')
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True)
    df.to_pickle(str(experiment_dir / 'results.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLIP DA Experiments')
    parser.add_argument('model', type=str, choices=MODELS + ['all'],
                        help='CLIP Model')
    parser.add_argument('--C', default=0.316, type=float,
                        help='Inverse regularization for linear probe')

    args = parser.parse_args()
    if args.model == 'all':
        for model_name in MODELS:
            run_exp(model_name, args.C)
    else:
        run_exp(args.model, args.C)
