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
DOMAINNET_CLASSNAMES_FILE = '/u/scr/nlp/domainnet/SENTRY_splits/classnames.txt'
DOMAINS = ['real', 'clipart', 'painting', 'sketch']
EXPERIMENT_TYPES = ['linear-probe', 'zero-shot']
MODELS = ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']


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


def load_classnames():
    with open(DOMAINNET_CLASSNAMES_FILE, 'r') as classnames_file:
        return [classname.strip() for classname in classnames_file]


def linear_probe(model_name, C, num_selftrain_iters):
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

    columns = ['SourceDomain', 'TargetDomain', 'AvgPerClassAcc']
    columns += [f'AvgPerClassAccST{i}' for i in range(num_selftrain_iters)]
    df = pd.DataFrame(columns=columns)
    for source_domain in DOMAINS:
        train_features, train_labels = train_features_labels[source_domain]
        classifier = LogisticRegression(random_state=0, C=C, max_iter=10000,
                                        warm_start=True)
        classifier.fit(train_features, train_labels)
        for target_domain in DOMAINS:
            accuracies = []
            for selftrain_iter in range(num_selftrain_iters + 1):
                test_features, test_labels = test_features_labels[target_domain]
                preds = classifier.predict(test_features)
                per_class_avg_acc = compute_per_class_avg_acc(preds, test_labels)
                accuracies.append(per_class_avg_acc)
                if selftrain_iter < num_selftrain_iters:
                    target_features, _ = train_features_labels[target_domain]
                    target_pseudolabels = classifier.predict(target_features)
                    classifier.fit(
                        np.concatenate((train_features, target_features)),
                        np.concatenate((train_labels, target_pseudolabels))
                    )

            df.loc[len(df)] = [source_domain, target_domain] + accuracies

    results_dir = pathlib.Path(__file__).parent.resolve().parent / 'results'
    experiment_name = f'clip_domainnet_C{C}_{model_name}_probe'
    experiment_name = experiment_name.replace('/', '')
    if num_selftrain_iters > 0:
        experiment_name += f'_st{num_selftrain_iters}'
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True)
    df.to_pickle(str(experiment_dir / 'results.pkl'))

    # Format results for easy copy-paste
    print(f'CLIP {model_name} Avg. Per-Class Acc.')
    transfer_accs = df[df.SourceDomain != df.TargetDomain].AvgPerClassAcc
    print(transfer_accs.to_string(index=False))


def compute_per_class_avg_acc(preds, labels):
    cm = confusion_matrix(labels, preds)
    per_class_correct = np.diag(cm)
    return np.mean(per_class_correct / cm.sum(axis=1))


def zero_shot(model_name):
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(model_name, device)
    print(f'Running experiments for model {model_name}')
    classnames = load_classnames()
    text_inputs = torch.cat(
        [clip.tokenize(f"a photo of a {c}") for c in classnames]
    ).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        for domain in DOMAINS:
            dataset = DomainNet(domain, split='test', root=DOMAINNET_ROOT,
                                transform=preprocess)
            all_preds = []
            all_labels = []
            for images, labels in tqdm(DataLoader(dataset, batch_size=128)):
                # Calculate features
                with torch.no_grad():
                    image_features = model.encode_image(images)
                # Pick the top 5 most similar labels for the image
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100 * image_features @ text_features.T
                sim = logits.softmax(dim=-1)
                preds = sim.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)

            preds = torch.cat(all_preds).numpy()
            labels = torch.cat(all_labels).numpy()
            per_class_avg_acc = 100. * compute_per_class_avg_acc(preds, labels)
            print(f'Per-Class Avg. Acc. for {domain}: {per_class_avg_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLIP DA Experiments')
    parser.add_argument('experiment_type', type=str, choices=EXPERIMENT_TYPES,
                        help='Experiment to run')
    parser.add_argument('model', type=str, choices=MODELS + ['all'],
                        help='CLIP Model')
    parser.add_argument('--C', default=0.316, type=float,
                        help='Inverse regularization for linear probe')
    parser.add_argument('--num_selftrain_iters', default=0, type=int,
                        help='Number of self-training iterations')

    args = parser.parse_args()
    if args.experiment_type == 'linear-probe':
        if args.model == 'all':
            for model_name in MODELS:
                linear_probe(model_name, args.C, args.num_selftrain_iters)
        else:
            linear_probe(args.model, args.C, args.num_selftrain_iters)
    elif args.experiment_type == 'zero-shot':
        if args.model == 'all':
            for model_name in MODELS:
                zero_shot(model_name)
        else:
            zero_shot(args.model)
