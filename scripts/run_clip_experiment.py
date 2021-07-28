import argparse
import clip
import torch
import numpy as np
import pandas as pd
import pathlib
from RandAugment import RandAugment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from unlabeled_extrapolation.datasets.domainnet import DomainNet


DOMAINNET_ROOT = '/scr/biggest/domainnet'
DOMAINNET_CLASSNAMES_FILE = '/u/scr/nlp/domainnet/SENTRY_splits/classnames.txt'
IMAGENET_CLASSNAMES_FILE = 'imagenet_classes.txt'
DOMAINS = ['sketch', 'clipart', 'painting', 'real']
EMBEDDING_DIMS = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'ViT-B/32': 512,
                  'ViT-B/16': 512, 'RN50x16': 768}
EXPERIMENT_TYPES = ['linear-probe', 'zero-shot', 'finetune']
IMAGENET_PROMPTS_FILE = 'imagenet_prompts.txt'
MODELS = ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']


# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-', '').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-', '').replace('.', '').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val


def get_features(dataset, model, source_domain=None, target_domain=None,
                 adapt=False):

    all_features = []
    all_labels = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classnames = load_classnames()
    with torch.no_grad():
        if source_domain:
            source_text_inputs = torch.cat(
                [clip.tokenize(f'a {source_domain} image of a {c}') for c in classnames]
            ).to(device)
            source_text_features = model.encode_text(source_text_inputs)
        if target_domain:
            target_text_inputs = torch.cat(
                [clip.tokenize(f'a {target_domain} image of a {c}') for c in classnames]
            ).to(device)
            target_text_features = model.encode_text(target_text_inputs)
        for images, labels in tqdm(DataLoader(dataset, batch_size=128)):
            features = model.encode_image(images.to(device))
            if adapt:
                features += target_text_features[labels] - source_text_features[labels]

            all_features.append(features)
            all_labels.append(labels)

    features = torch.cat(all_features).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()
    return features, labels


def load_classnames(dataset_name='domainnet'):
    if dataset_name == 'domainnet':
        classnames_filename = DOMAINNET_CLASSNAMES_FILE
    elif dataset_name == 'imagenet':
        classnames_filename = IMAGENET_CLASSNAMES_FILE
    else:
        raise ValueError(f'{dataset_name} not supported')

    with open(classnames_filename, 'r') as classnames_file:
        return [classname.strip() for classname in classnames_file]


def init_optimizer(args, model, predictor):
    optimizer_class = getattr(torch.optim, args.optimizer_name)
    optimizer_args = [{'params': predictor.parameters(), 'lr': args.lr}]
    if not args.freeze:
        optimizer_args.append(
            {'params': model.visual.parameters(), 'lr': args.encoder_lr}
        )
    return optimizer_class(optimizer_args, **args.optimizer_kwargs)


def linear_probe(args):
    # Load the model
    model_name = args.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(model_name, device)
    print(f'Running experiments for model {model_name}')
    num_params = np.sum([int(np.prod(p.shape)) for p in model.parameters()])
    print('Model parameters:', f'{num_params:,}')
    columns = ['SourceDomain', 'TargetDomain', 'AvgPerClassAcc']
    num_selftrain_iters = args.num_selftrain_iters
    columns += [f'AvgPerClassAccST{i}' for i in range(num_selftrain_iters)]
    df = pd.DataFrame(columns=columns)

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

    for src_domain in DOMAINS:
        if not args.translate_features:
            train_features, train_labels = train_features_labels[src_domain]
            classifier = LogisticRegression(random_state=0, C=args.C,
                                            max_iter=10000, warm_start=True)
            classifier.fit(train_features, train_labels)
        for tgt_domain in DOMAINS:
            if args.translate_features:
                train_features, train_labels = train_features_labels[src_domain]
                src_name = 'photo' if src_domain == 'real' else src_domain
                tgt_name = 'photo' if tgt_domain == 'real' else tgt_domain
                src_token = clip.tokenize(src_name).to(device)
                tgt_token = clip.tokenize(tgt_name).to(device)
                with torch.no_grad():
                    src_embed = model.encode_text(src_token)
                    tgt_embed = model.encode_text(tgt_token)
                    translation = torch.flatten(tgt_embed - src_embed)
                    translation = translation.cpu().numpy()
                    train_features = train_features + translation

                classifier = LogisticRegression(random_state=0, C=args.C,
                                                max_iter=10000, warm_start=True)
                classifier.fit(train_features, train_labels)

            accuracies = []
            for selftrain_iter in range(num_selftrain_iters + 1):
                test_features, test_labels = test_features_labels[tgt_domain]
                preds = classifier.predict(test_features)
                per_cls_avg_acc = compute_per_class_avg_acc(preds, test_labels)
                accuracies.append(per_cls_avg_acc)
                print(f'{src_domain} -> {tgt_domain}: {per_cls_avg_acc}')
                if selftrain_iter < num_selftrain_iters:
                    target_features, _ = train_features_labels[tgt_domain]
                    target_pseudolabels = classifier.predict(target_features)
                    classifier.fit(
                        np.concatenate((train_features, target_features)),
                        np.concatenate((train_labels, target_pseudolabels))
                    )

            df.loc[len(df)] = [src_domain, tgt_domain] + accuracies

    # Format results for easy copy-paste
    print(f'CLIP {model_name} Avg. Per-Class Acc.')
    transfer_accs = df[df.SourceDomain != df.TargetDomain].AvgPerClassAcc
    print(transfer_accs.to_string(index=False))

    results_dir = pathlib.Path(__file__).parent.resolve().parent / 'results'
    experiment_name = f'clip_domainnet_C{args.C}_{model_name}'
    if args.translate_features:
        experiment_name += '_translatefeats'
    experiment_name += '_probe'
    experiment_name = experiment_name.replace('/', '')
    if num_selftrain_iters > 0:
        experiment_name += f'_st{num_selftrain_iters}'
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True)
    df.to_pickle(str(experiment_dir / 'results.pkl'))


def compute_per_class_avg_acc(preds, labels):
    cm = confusion_matrix(labels, preds)
    per_class_correct = np.diag(cm)
    return 100. * np.mean(per_class_correct / cm.sum(axis=1))


def generate_prompt_embeddings(model, classnames, device, prompt_type='default'):
    with torch.no_grad():
        if prompt_type == 'default':
            text_inputs = torch.cat(
                [clip.tokenize(f'a photo of a {c}') for c in classnames]
            ).to(device)
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        elif prompt_type == 'imagenet':
            with open(IMAGENET_PROMPTS_FILE, 'r') as imagenet_prompts_file:
                prompts = [prompt.strip() for prompt in imagenet_prompts_file]
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in prompts]
                texts = clip.tokenize(texts).to(device)
                class_embeddings = model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            text_features = torch.stack(zeroshot_weights, dim=1).to(device).T
        return text_features


def zero_shot(model_name):
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(model_name, device)
    print(f'Running experiments for model {model_name}')
    # classnames = load_classnames()
    classnames = load_classnames('imagenet')
    text_features = generate_prompt_embeddings(model, classnames, device, 'imagenet')
    # text_features = generate_prompt_embeddings(model, classnames, device)
    with torch.no_grad():
        for domain in DOMAINS:
            dataset = DomainNet(domain, split='test', root=DOMAINNET_ROOT,
                                transform=preprocess)
            import torchvision
            dataset = torchvision.datasets.ImageFolder('/scr/biggest/imagenet_sketch/sketch',
                                                       transform=preprocess)
            all_preds = []
            all_labels = []
            for images, labels in tqdm(DataLoader(dataset, batch_size=128)):
                # Calculate features
                image_features = model.encode_image(images)
                # Pick the top 5 most similar labels for the image
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100 * image_features @ text_features.T
                # sim = logits.softmax(dim=-1)
                # preds = preds.argmax(dim=-1).cpu()
                preds = logits.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)

            # import ipdb; ipdb.set_trace()
            preds = torch.cat(all_preds).numpy()
            labels = torch.cat(all_labels).numpy()
            per_class_avg_acc = compute_per_class_avg_acc(preds, labels)
            print(f'Per-Class Avg. Acc. for {domain}: {per_class_avg_acc}')
            # print(confusion_matrix(labels, preds))
            print(f'Accuracy for {domain}: {np.mean(preds == labels)}')
            return


def finetune(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = args.model
    print(f'Running experiments for model {model_name}')
    columns = ['SourceDomain', 'TargetDomain', 'AvgPerClassAcc']
    df = pd.DataFrame(columns=columns)
    embed_dim = EMBEDDING_DIMS[model_name]
    loss = torch.nn.CrossEntropyLoss()
    for src_domain in DOMAINS:
        model, preprocess = clip.load(model_name, device, jit=False)
        if args.randaugment_M > 0 and args.randaugment_N > 0:
            augment = RandAugment(args.randaugment_M, args.randaugment_N)
            preprocess.transforms.insert(0, augment)

        model.train()
        model.float()
        train_dataset = DomainNet(src_domain, root=DOMAINNET_ROOT,
                                  transform=preprocess)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True)
        num_classes = train_dataset.get_num_classes()
        classifier = torch.nn.Linear(embed_dim, num_classes).to(device)
        optimizer = init_optimizer(args, model, classifier)
        for epoch in range(args.epochs):
            print(f'Epoch {epoch}')
            total_loss = 0.
            for images, labels in tqdm(train_loader):
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                if args.freeze:
                    with torch.no_grad():
                        embeddings = model.encode_image(images)
                        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                else:
                    embeddings = model.encode_image(images)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                batch_loss = loss(classifier(embeddings), labels)
                total_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
            print(f'Average loss: {total_loss / len(train_loader)}')

        model.eval()
        for tgt_domain in DOMAINS:
            target_dataset = DomainNet(tgt_domain, split='test',
                                       root=DOMAINNET_ROOT,
                                       transform=preprocess)
            target_loader = DataLoader(target_dataset, args.batch_size)
            all_preds = []
            all_labels = []
            for images, labels in tqdm(target_loader):
                images = images.to(device)
                with torch.no_grad():
                    embeddings = model.encode_image(images)
                    embeddings /= embeddings.norm(dim=-1, keepdim=True)
                    logits = classifier(embeddings)
                preds = logits.argmax(dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(labels)
            preds = torch.cat(all_preds).numpy()
            labels = torch.cat(all_labels).numpy()
            per_cls_avg_acc = compute_per_class_avg_acc(preds, labels)
            print(f'{src_domain} -> {tgt_domain}: {per_cls_avg_acc}')
            df.loc[len(df)] = [src_domain, tgt_domain, per_cls_avg_acc]

    # Format results for easy copy-paste
    print(f'CLIP {model_name} Avg. Per-Class Acc.')
    transfer_accs = df[df.SourceDomain != df.TargetDomain].AvgPerClassAcc
    print(transfer_accs.to_string(index=False))

    results_dir = pathlib.Path(__file__).parent.resolve().parent / 'results'
    experiment_name = f'clip_domainnet_{model_name}'
    experiment_name += f'_optimizer{args.optimizer_name}'
    experiment_name += f'_lr{args.lr}_encoderlr{args.encoder_lr}'
    if args.translate_features:
        experiment_name += '_translatefeats'
    experiment_name += '_finetune'
    experiment_name = experiment_name.replace('/', '')
    experiment_dir = results_dir / experiment_name
    experiment_dir.mkdir(parents=True)
    df.to_pickle(str(experiment_dir / 'results.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run CLIP DA Experiments')
    parser.add_argument('experiment_type', type=str, choices=EXPERIMENT_TYPES,
                        help='Experiment to run')
    parser.add_argument('model', type=str,
                        choices=clip.available_models() + ['all'],
                        help='CLIP Model')
    parser.add_argument('--num_selftrain_iters', default=0, type=int,
                        help='Number of self-training iterations')

    linear_probe_group = parser.add_argument_group('Linear Probing Arguments')
    linear_probe_group.add_argument('--C', default=0.316, type=float,
                                    help='Inverse regularization')
    linear_probe_group.add_argument('--translate_features',
                                    action='store_true',
                                    help='Use language to translate domains')

    finetuning_group = parser.add_argument_group('Finetuning Arguments')
    finetuning_group.add_argument('--epochs', default=10, type=int,
                                  help='Number of epochs for finetuning')
    finetuning_group.add_argument('--batch_size', default=128, type=int,
                                  help='Batch size for finetuning')
    finetuning_group.add_argument('--freeze', action='store_true',
                                  help='Freeze encoder during finetuning')

    randaugment_group = parser.add_argument_group('RandAugment Arguments')
    randaugment_group.add_argument('--randaugment_M', default=0, type=int,
                                   help='Value of M for RandAugment')
    randaugment_group.add_argument('--randaugment_N', default=0, type=int,
                                   help='Value of N for RandAugment')
    optimizer_group = parser.add_argument_group('PyTorch Optimizer Arguments')
    optimizer_group.add_argument('--optimizer_name', default='SGD', type=str,
                                 help='Classname of PyTorch optimizer to use')
    optimizer_group.add_argument('--lr', default=1e-3, type=float,
                                 help='Learning rate for finetuning')
    optimizer_group.add_argument('--encoder_lr', default=1e-3, type=float,
                                 help='Learning rate for encoder')
    optimizer_group.add_argument('--optimizer_kwargs', nargs='*',
                                 action=ParseKwargs, default={})
    args = parser.parse_args()
    if args.experiment_type == 'linear-probe':
        if args.model == 'all':
            for model_name in MODELS:
                args.model = model_name
                linear_probe(args)
        else:
            linear_probe(args)
    elif args.experiment_type == 'zero-shot':
        if args.model == 'all':
            for model_name in MODELS:
                zero_shot(model_name)
        else:
            zero_shot(args.model)
    elif args.experiment_type == 'finetune':
        if args.model == 'all':
            for model_name in MODELS:
                args.model = model_name
                finetune(args)
        else:
            finetune(args)
