import argparse
import ast
import clip
import torch
import numpy as np
import pandas as pd
import pathlib
import sklearn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from unlabeled_extrapolation.datasets.domainnet import DomainNet


DOMAINNET_ROOT = '/scr/biggest/domainnet'
DOMAINNET_CLASSNAMES_FILE = '/u/scr/nlp/domainnet/SENTRY_splits/classnames.txt'
IMAGENET_CLASSNAMES_FILE = 'imagenet_classes.txt'
DF_COLUMNS = ['SourceDomain', 'TargetDomain', 'AvgPerClassAcc']
DOMAINS = ['sketch', 'clipart', 'painting', 'real']
EXPERIMENT_TYPES = ['probe', 'zero-shot', 'finetune', 'evaluate']
IMAGENET_PROMPTS_FILE = 'imagenet_prompts.txt'
RESULTS_DIR = pathlib.Path(__file__).parent.resolve().parent / 'results'


translate_functions = dict()


def register_translate_function(function):  # Decorator
    translate_functions[function.__name__] = function
    return function


@register_translate_function
def translate_domain_names(img_encodings, src_domain, tgt_domain, clip_model):
    assert src_domain in DOMAINS and tgt_domain in DOMAINS
    device = img_encodings.device
    src_token = clip.tokenize(src_domain).to(device)
    tgt_token = clip.tokenize(tgt_domain).to(device)
    with torch.no_grad():
        src_embed = clip_model.encode_text(src_token)
        tgt_embed = clip_model.encode_text(tgt_token)
        translation = torch.flatten(tgt_embed - src_embed)
        translated_features = img_encodings + translation
    return translated_features


# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            processed_val = ast.literal_eval(value_str)
            getattr(namespace, self.dest)[key] = processed_val


def stringify_argument_group(args, group_title):
    group_strings = []
    group_args = sorted(args.groups[group_title])
    for arg in group_args:
        val = getattr(args, arg)
        if isinstance(val, dict):
            kwargs = val
            kwargs_strings = []
            for key in sorted(kwargs):
                kwargs_strings.append(f'{key.replace("_", "")}{kwargs[key]}')
            group_strings.append('_'.join(kwargs_strings))
        else:
            group_strings.append(f'{arg.replace("_", "")}{val}')
    return '_'.join(group_strings)


def make_experiment_dir(args):
    experiment_name = f'clip_domainnet_{args.model}'
    experiment_name += f'_source{args.source_domain}'
    experiment_name += f'_target{args.target_domain}'
    if args.translate_features:
        experiment_name += '_translatefeats'
    if args.translate_target:
        experiment_name += '_translatetarget'

    if args.experiment_type == 'finetune':
        experiment_name += f'_{stringify_argument_group(args, "finetune")}'
        experiment_name += f'_{stringify_argument_group(args, "optimizer")}'
    elif args.experiment_type == 'probe':
        experiment_name += f'_{stringify_argument_group(args, "probe")}'

    experiment_name += f'_{args.experiment_type.replace("-", "")}'
    experiment_name = experiment_name.replace('/', '')
    experiment_dir = RESULTS_DIR / experiment_name
    if not args.no_save:
        results_file = experiment_dir / 'results.pkl'
        if results_file.exists() and not args.overwrite:
            error_string = f'Experiment {experiment_name} already exists!'
            error_string += ' Use --overwrite option to replace'
            raise ValueError(error_string)
        else:
            experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def init_sklearn_classifier(classifier_name, classifier_kwargs):
    sklearn_classifiers = dict(sklearn.utils.all_estimators('classifier'))
    return sklearn_classifiers[classifier_name](**classifier_kwargs)


class ClipClassifier(torch.nn.Module):
    def __init__(self, clip_model, preprocess, output_dim,
                 freeze_encoder=False):

        super(ClipClassifier, self).__init__()
        self.dtype = clip_model.dtype
        self.encoder = clip_model.visual.float()
        self.classifier = torch.nn.Linear(self.encoder.output_dim, output_dim)
        self.preprocess = preprocess
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        embeddings = self.encoder(x.type(self.dtype))
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return self.classifier(embeddings)


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


def load_classnames(dataset_name='domainnet'):
    if dataset_name == 'domainnet':
        classnames_filename = DOMAINNET_CLASSNAMES_FILE
    elif dataset_name == 'imagenet':
        classnames_filename = IMAGENET_CLASSNAMES_FILE
    else:
        raise ValueError(f'{dataset_name} not supported')

    with open(classnames_filename, 'r') as classnames_file:
        return [classname.strip() for classname in classnames_file]


def init_optimizer(args, model):
    optimizer_class = getattr(torch.optim, args.optimizer_name)
    optimizer_args = [{'params': model.classifier.parameters(), 'lr': args.lr}]
    if not args.freeze:
        optimizer_args.append(
            {'params': model.encoder.parameters(), 'lr': args.encoder_lr}
        )
    return optimizer_class(optimizer_args, **args.optimizer_kwargs)


def translate_features(src_features, src_domain, tgt_domain, model):
    assert src_domain in DOMAINS and tgt_domain in DOMAINS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    src_name = 'photo' if src_domain == 'real' else src_domain
    tgt_name = 'photo' if tgt_domain == 'real' else tgt_domain
    src_token = clip.tokenize(src_name).to(device)
    tgt_token = clip.tokenize(tgt_name).to(device)
    with torch.no_grad():
        src_embed = model.encode_text(src_token)
        tgt_embed = model.encode_text(tgt_token)
        translation = torch.flatten(tgt_embed - src_embed)
        translation = translation.cpu().numpy()
        translated_features = src_features + translation
    return translated_features


def probe(args):
    experiment_dir = make_experiment_dir(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, preprocess = clip.load(args.model, device)
    train_domain = args.source_domain
    train_dataset = DomainNet(train_domain, root=DOMAINNET_ROOT,
                              transform=preprocess)
    print(f'Generating training embeddings for {train_domain}...')
    train_features, train_labels = get_features(train_dataset, clip_model)
    if args.translate_features:
        train_features = translate_features(train_features, train_domain,
                                            args.target_domain, clip_model)
    probe = init_sklearn_classifier(args.sklearn_classifier_name,
                                    args.sklearn_classifier_kwargs)
    print(f'Training {probe} probe on {train_domain}...')
    probe.fit(train_features, train_labels)

    df = pd.DataFrame(columns=DF_COLUMNS)
    source_domain = args.source_domain
    if not args.skip_source_eval:
        test_dataset = DomainNet(source_domain, split='test',
                                 root=DOMAINNET_ROOT, transform=preprocess)
        print(f'Generating test embeddings for {source_domain}...')
        test_features, test_labels = get_features(test_dataset, clip_model)
        preds = probe.predict(test_features)
        per_cls_avg_acc = compute_per_class_avg_acc(preds, test_labels)
        print(f'{source_domain} test accuracy: {per_cls_avg_acc}')
        df.loc[len(df)] = (source_domain, source_domain, per_cls_avg_acc)

    if args.target_domain == 'all':
        eval_domains = DOMAINS
    else:
        eval_domains = [args.target_domain]

    for eval_domain in eval_domains:
        test_dataset = DomainNet(eval_domain, split='test',
                                 root=DOMAINNET_ROOT, transform=preprocess)
        print(f'Generating test embeddings for {eval_domain}...')
        test_features, test_labels = get_features(test_dataset, clip_model)
        if args.translate_target:
            test_features = translate_features(test_features, eval_domain,
                                               source_domain, clip_model)
        preds = probe.predict(test_features)
        per_cls_avg_acc = compute_per_class_avg_acc(preds, test_labels)
        print(f'{train_domain} -> {eval_domain}: {per_cls_avg_acc}')
        df.loc[len(df)] = (source_domain, eval_domain, per_cls_avg_acc)

    # Format results for easy copy-paste
    print(f'CLIP {args.model} Avg. Per-Class Acc.')
    print(df.AvgPerClassAcc.to_string(index=False))
    if not args.no_save:
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

            preds = torch.cat(all_preds).numpy()
            labels = torch.cat(all_labels).numpy()
            per_class_avg_acc = compute_per_class_avg_acc(preds, labels)
            print(f'Per-Class Avg. Acc. for {domain}: {per_class_avg_acc}')
            # print(confusion_matrix(labels, preds))
            print(f'Accuracy for {domain}: {np.mean(preds == labels)}')
            return


def train(args):
    clip_model, preprocess = clip.load(args.model, 'cpu')
    train_domain = args.source_domain
    train_dataset = DomainNet(train_domain, root=DOMAINNET_ROOT,
                              transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True)

    num_classes = train_dataset.get_num_classes()
    model = ClipClassifier(clip_model, preprocess, num_classes, args.freeze)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    optimizer = init_optimizer(args, model)
    loss = torch.nn.CrossEntropyLoss()
    print(f'Training {args.model} on {train_domain} for {args.epochs} epochs')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        total_loss = 0.
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            batch_loss = loss(model(images), labels)
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        print(f'Average loss: {total_loss / len(train_loader)}')

    return model


def evaluate(model, eval_domain, args):
    model.eval()
    target_dataset = DomainNet(eval_domain, split='test',
                               root=DOMAINNET_ROOT,
                               transform=model.preprocess)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_loader = DataLoader(target_dataset, args.batch_size)
    all_preds = []
    all_labels = []
    print(f'Evaluating on {eval_domain}...')
    for images, labels in tqdm(target_loader):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    per_cls_avg_acc = compute_per_class_avg_acc(preds, labels)
    return per_cls_avg_acc


def finetune(args):
    experiment_dir = make_experiment_dir(args)
    df = pd.DataFrame(columns=DF_COLUMNS)
    source_domain = args.source_domain
    model = train(args)
    if not args.skip_source_eval:
        per_cls_avg_acc = evaluate(model, args.source_domain, args)
        print(f'{source_domain} test accuracy: {per_cls_avg_acc}')
        df.loc[len(df)] = (source_domain, source_domain, per_cls_avg_acc)

    if args.target_domain == 'all':
        eval_domains = DOMAINS
    else:
        eval_domains = [args.target_domain]

    for eval_domain in eval_domains:
        per_cls_avg_acc = evaluate(model, eval_domain, args)
        print(f'{source_domain} -> {eval_domain}: {per_cls_avg_acc}')
        df.loc[len(df)] = (source_domain, eval_domain, per_cls_avg_acc)

    # Format results for easy copy-paste
    print(f'CLIP {args.model} Avg. Per-Class Acc.')
    print(df.AvgPerClassAcc.to_string(index=False))
    if not args.no_save:
        df.to_pickle(str(experiment_dir / 'results.pkl'))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Run CLIP DA Experiments')
    parser.add_argument('source_domain', type=str, choices=DOMAINS,
                        help='Source domain')
    parser.add_argument('target_domain', type=str, choices=DOMAINS + ['all'],
                        help='Target domain')
    parser.add_argument('experiment_type', type=str, choices=EXPERIMENT_TYPES,
                        help='Experiment to run')
    parser.add_argument('model', type=str,
                        choices=clip.available_models() + ['all'],
                        help='CLIP Model')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite previously saved results')
    parser.add_argument('--num_selftrain_iters', default=0, type=int,
                        help='Number of self-training iterations')
    parser.add_argument('--no_save', action='store_true',
                        help='Don\'t save results')

    probe_group = parser.add_argument_group('probe')
    probe_group.add_argument('--sklearn_classifier_name',
                             default='LogisticRegression', type=str,
                             help='Name of sklearn classifier')
    probe_group.add_argument('--sklearn_classifier_kwargs', action=ParseKwargs,
                             nargs='*', default=dict(),
                             help='Keyword arguments for sklearn classifier')

    language_group = parser.add_argument_group('language')
    language_group.add_argument('--translate_features',
                                action='store_true',
                                help='Use language to translate domains')
    language_group.add_argument('--translate_target', action='store_true',
                                help='Translate target domains')
    language_group.add_argument('--translate_function', type=str,
                                help='Function to perform translation')

    finetuning_group = parser.add_argument_group('finetune')
    finetuning_group.add_argument('--epochs', default=10, type=int,
                                  help='Number of epochs for finetuning')
    finetuning_group.add_argument('--batch_size', default=128, type=int,
                                  help='Batch size for finetuning')
    finetuning_group.add_argument('--freeze', action='store_true',
                                  help='Freeze encoder during finetuning')
    finetuning_group.add_argument('--skip_source_eval', action='store_true',
                                  help='Don\'t test on source domain')

    randaugment_group = parser.add_argument_group('randaugment')
    randaugment_group.add_argument('--randaugment_M', default=0, type=int,
                                   help='Value of M for RandAugment')
    randaugment_group.add_argument('--randaugment_N', default=0, type=int,
                                   help='Value of N for RandAugment')
    optimizer_group = parser.add_argument_group('optimizer')
    optimizer_group.add_argument('--optimizer_name', default='SGD', type=str,
                                 help='Classname of PyTorch optimizer to use')
    optimizer_group.add_argument('--lr', default=1e-3, type=float,
                                 help='Learning rate for finetuning')
    optimizer_group.add_argument('--encoder_lr', default=1e-3, type=float,
                                 help='Learning rate for encoder')
    optimizer_group.add_argument('--optimizer_kwargs', nargs='*',
                                 action=ParseKwargs, default={})

    parsed_args = parser.parse_args(args)
    argument_groups = {}
    for group in parser._action_groups:
        if group.title in ('positional arguments', 'optional arguments'):
            continue
        group_args = [arg.dest for arg in group._group_actions]
        argument_groups[group.title] = group_args
    setattr(parsed_args, 'groups', argument_groups)
    return parsed_args


if __name__ == '__in__':
    args = parse_args()
    if args.translate_features and args.target_domain == 'all':
        raise ValueError('If --translate_features is passed then only a single'
                         ' target domain must be used.')

    if args.experiment_type == 'probe':
        if args.model == 'all':
            for model_name in clip.available_models():
                args.model = model_name
                probe(args)
        else:
            probe(args)
    elif args.experiment_type == 'zero-shot':
        if args.model == 'all':
            for model_name in clip.available_models():
                zero_shot(model_name)
        else:
            zero_shot(args.model)
    elif args.experiment_type == 'finetune':
        if args.model == 'all':
            for model_name in clip.available_models():
                args.model = model_name
                finetune(args)
        else:
            finetune(args)
