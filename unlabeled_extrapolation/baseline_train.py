import argparse
import datetime
import json
import logging
import os
import os.path
import pandas as pd
from pathlib import Path
import shutil
import sys
import torch
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
import quinine

from unlabeled_extrapolation.models import resnet
from unlabeled_extrapolation.utils.accumulator import Accumulator
import unlabeled_extrapolation.utils.utils as utils


log_level = logging.INFO


def reset_state(model, training):
    if training:
        model.train()
    else:
        model.eval()


def get_test_stats(config, net, test_loader, criterion, device, max_examples=float('infinity')):
    # Evaluate accuracy and loss on validation.
    # Returns right after we've seen at least max_examples examples (not batches).
    val_loss = Accumulator()
    val_acc = Accumulator()
    training_state = net.training
    net.eval()
    num_examples = 0
    with torch.no_grad():
        for data in test_loader:
            if config['use_cuda']:
                data = utils.to_device(data, device)
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            val_acc.add_values((predicted == labels).tolist())
            loss = criterion(outputs, labels)
            val_loss.add_value(loss.tolist())
            num_examples += len(images)
            if num_examples >= max_examples:
                break
    reset_state(net, training_state)
    return val_loss, val_acc


def update_best_stats(stats, best_stats):
    for k, v in stats.items():
        best_k = 'best_' + k
        if best_k in best_stats:
            cmb = max
            if k.find('loss') != -1 and k.find('acc') == -1:
                cmb = min
            best_stats[best_k] = cmb(best_stats[best_k], v)
        else:
            best_stats[best_k] = v


def get_test_loaders(config):
    test_loaders = {}
    max_test_examples = {}
    logging.info('Found %d testing datasets.', len(config['test_datasets']))
    for test_dataset_config in config['test_datasets']:
        logging.info('test dataset config: ' + str(test_dataset_config))
        # Initialize dataset and data loader.
        # Shuffle is True in case we only test part of the test set.
        test_data = utils.init_dataset(test_dataset_config)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_workers'])
        test_config_name = test_dataset_config['name']
        test_loaders[test_config_name] = test_loader
        # Some test datasets like CINIC are huge so we only test part of the dataset.
        if 'max_test_examples' in test_dataset_config:
            max_test_examples[test_config_name] = test_dataset_config['max_test_examples']
        logging.info('test loader name: ' + test_dataset_config['name'])
        logging.info('test loader: ' + str(test_loader))
        logging.info('test transform: ' + str(test_dataset_config['transforms']))
    return test_loaders, max_test_examples


def build_model(config):
    net = utils.initialize(config['model'])
    # If fine-tune, re-initialize the last layer.
    finetune = 'finetune' in config and config['finetune']
    linear_probe = 'linear_probe' in config and config['linear_probe']
    def count_parameters(model, trainable):
        return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)
    if finetune or linear_probe:
        if linear_probe:
            logging.info('linear probing, freezing bottom layers.')
            if 'use_net_val_mode' not in config:
                config['use_net_val_mode'] = True
                logging.warning('Linear probing, so setting unspecified use_net_val_mode to True')
            net.set_requires_grad(False)
        if 'probe_net' in config:
            probe_net = utils.initialize(config['probe_net'])
            net.add_probe(probe_net)
        else:
            net.new_last_layer(config['num_classes'])
        num_trainable_params = count_parameters(net, True)
        num_params = count_parameters(net, False) + num_trainable_params
        logging.info(f'Fine Tuning {num_trainable_params} of {num_params} parameters.')
    return net


def train(epoch, config, train_loader, net, device, optimizer, criterion, model_loss,
          test_loaders, max_test_examples):
    # Train model.
    training_state = net.training
    logging.info("\nEpoch #{}".format(epoch))
    loss_dict = {
        'train/loss': Accumulator(),
        'train/acc': Accumulator(),
    }
    if 'model_loss' in config:
        loss_dict['train/model_loss'] = Accumulator()
    num_examples = 0
    for i, data in enumerate(train_loader, 0):
        if 'use_net_val_mode' in config and config['use_net_val_mode']:
            net.eval()
        else:
            net.train()
        # get the inputs; data is a list of [inputs, labels]
        if config['use_cuda']:
            data = utils.to_device(data, device)
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, train_preds = torch.max(outputs.data, axis=1)
        loss_dict['train/loss'].add_value(loss.tolist())
        loss_dict['train/acc'].add_values((train_preds == labels).tolist())
        if 'model_loss' in config:
            opt_loss = model_loss(net, inputs, labels)
            opt_loss.backward()
            loss_dict['train/model_loss'].add_value(opt_loss.tolist())
        else:
            loss.backward()
        optimizer.step() 
        num_examples += len(labels)
        outputs, loss, train_preds = None, None, None  # Try to force garbage collection.
        def should_log(log_interval):
            return num_examples // log_interval > (num_examples - len(labels)) // log_interval
        if should_log(config['log_interval']):
            for k in loss_dict:
                logging.info(
                    '[%d, %5d] %s: %.3f' %
                    (epoch + 1, num_examples, k, loss_dict[k].get_mean()))
        # Sometimes we want to log the test loss more often to track things better.
        if 'test_interval' in config and should_log(config['test_interval']):
            stats = get_all_test_stats(
                test_loaders, max_test_examples, config, net, criterion, device,
                loss_name_prefix='inter_test_loss/', acc_name_prefix='inter_test_acc/')
            if config['wandb']:
                wandb.log(stats)
    reset_state(net, training_state)
    train_stats = {'epoch': epoch}
    for key in loss_dict:
        train_stats[key] = loss_dict[key].get_mean()
    return train_stats


def get_all_test_stats(epoch, test_loaders, max_test_examples, config, net, criterion, device,
                   loss_name_prefix, acc_name_prefix):
    stats = {'epoch': epoch}
    for name, test_loader in test_loaders.items():
        max_examples = float('infinity')
        if name in max_test_examples:
            max_examples = max_test_examples[name]
            val_loss, val_acc = get_test_stats(
                config, net, test_loader, criterion, device,
                max_examples=max_examples)
            stats[loss_name_prefix + name] = val_loss.get_mean()
            stats[acc_name_prefix + name] = val_acc.get_mean()
    return stats


def main(config, log_dir, checkpoints_dir):
    # Set up datasets and loaders.
    logging.info("Entering main.")
    train_data = utils.init_dataset(config['train_dataset'])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'])
    # Set up test loaders.
    test_loaders, max_test_examples = get_test_loaders(config)
    # Create model.
    net = build_model(config)
    # Use CUDA if desired. 
    logging.info(f'cuda device count: {torch.cuda.device_count()}') 
    if config['use_cuda']:
        # Often makes things faster, by benchmarking and figuring out how to optimize.
        cudnn.benchmark = True
        device = "cuda"
        net.cuda()
    logging.info('Using cuda? %d', next(net.parameters()).is_cuda)
    # Loss, optimizer, scheduler.
    # Can use a custom loss that takes in a model, inputs, labels, and gets an array of values.
    # For example if you want to regularize weights in some special way.
    # More commonly, we use a criterion, which takes in model_outputs, labels, and outputs a loss.
    # criterion must be specified anyway, since that's the loss we evaluate on test sets.
    criterion = utils.initialize(config['criterion'])
    model_loss = None
    if 'model_loss' in config:
        model_loss = utils.initialize(config['model_loss'])
    optimizer = utils.initialize(
            config['optimizer'], update_args={'params': net.parameters()})
    scheduler = utils.initialize(
            config['scheduler'], update_args={'optimizer': optimizer})
    # Training loop.
    best_stats = {}
    best_accs = {}  # Used to save checkpoints of best models on some datasets.
    train_metrics = []
    test_metrics = []
    prev_ckp_path = None
    for epoch in range(config['epochs']):
        # Save checkpoint once in a while.
        if epoch % config['save_freq'] == 0:
            cur_ckp_filename = 'ckp_' + str(epoch)
            utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, cur_ckp_filename)
            if (prev_ckp_path is not None and not(config['save_all_checkpoints'])):
                os.remove(prev_ckp_path)
            prev_ckp_path = checkpoints_dir / cur_ckp_filename
        # One epoch of model training.
        train_stats = train(
            epoch, config, train_loader, net, device, optimizer, criterion, model_loss,
            test_loaders, max_test_examples)      
        scheduler.step()
        # Get test stats across all test sets.
        test_stats = get_all_test_stats(
            epoch, test_loaders, max_test_examples, config, net, criterion, device,
            loss_name_prefix='test_loss/', acc_name_prefix='test_acc/')
        # Keep track of the best stats.
        update_best_stats(train_stats, best_stats)
        update_best_stats(test_stats, best_stats)
        # Log and save stats.
        train_metrics.append(train_stats)
        test_metrics.append(test_stats)
        train_df = pd.DataFrame(train_metrics)
        test_df = pd.DataFrame(test_metrics)
        train_df.to_csv(log_dir + '/stats_train.tsv', sep='\t')
        test_df.to_csv(log_dir + '/stats_test.tsv', sep='\t')
        if config['wandb']:
            wandb.log(train_stats)
            wandb.log(test_stats)
            wandb.log(best_stats)
        utils.save_json(log_dir + '/current_train.json', train_stats)
        utils.save_json(log_dir + '/current_test.json', test_stats)
        utils.save_json(log_dir + '/best.json', best_stats)
        # Save checkpoint of best model. We save the 'best' for each of a list
        # of specified valid datasets. For example, we might want to save the best
        # model according to in-domain validation metrics, but as an oracle, save
        # the best according to ood validation metrics (or a proxy ood metric).
        if 'early_stop_dataset_names' in config:
            logging.info(f"Early stopping using datasets {config['early_stop_dataset_names']}")
            for name in config['early_stop_dataset_names']:
                if name not in test_loaders:
                    raise ValueError(f"{name} is not the name of a test dataset.")
                metric_name = 'test_acc/' + name
                assert(metric_name in test_stats)
                if metric_name not in best_accs or test_stats[metric_name] > best_accs[metric_name]:
                    best_accs[metric_name] = test_stats[metric_name]
                    checkpoint_name = 'ckp_best_' + name
                    utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, checkpoint_name)
    utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, 'ckp_last')


def make_new_dir(new_dir):
    if os.path.isdir(new_dir):
        raise ValueError('{} already exists.'.format(new_dir))
    os.makedirs(new_dir)


def make_checkpoints_dir(log_dir):
    checkpoints_dir = log_dir + '/checkpoints'
    checkpoints_dir = Path(checkpoints_dir).resolve().expanduser()
    os.makedirs(checkpoints_dir)
    return checkpoints_dir


def copy_folders(log_dir):
    copy_folders = ['code', 'configs', 'scripts', 'lib', 'configs', 'models',
                    'experiments', 'utils', 'examples', 'src', 'datasets']
    for copy_folder in copy_folders:
        if os.path.isdir('./' + copy_folder):
            shutil.copytree('./' + copy_folder, log_dir + '/' + copy_folder)


def time_to_str(ts):
    return pd.Timestamp(ts).strftime('%Y-%m-%dT%H-%M-%S-%f')


def now_to_str():
    return time_to_str(datetime.datetime.now())


def setup_wandb(args, config):
    if not args.no_wandb:
        run_name = now_to_str() if args.run_name is None else args.run_name
        print(args.project_name, run_name, args.group_name, args.entity_name)
        run_obj = wandb.init(
            project=args.project_name, name=run_name,
            group=args.group_name, entity=args.entity_name)
        config['wandb_url'] = run_obj.get_url()
        config['run_name'] = run_name
        config['group_name'] = args.group_name
        config['entity_name'] = args.entity_name
        config['wandb'] = True
        wandb.config.update(config)
    else:
        config['wandb'] = False


def save_command_line_args(log_dir):
    command = ""
    command += sys.executable + " "
    command += " ".join(sys.argv)
    logging.info('Command: ' + command)
    with open(log_dir+'/command.txt', 'w') as f:
        f.write(command)
        f.write('\n')


def update_test_transform_configs(config):
    # Use default test transform for test datasets that don't specify a transform.
    for test_dataset_config in config['test_datasets']:
        if 'transforms' not in test_dataset_config:
            if config['default_test_transforms'] is None:
                raise ValueError('Must either specify default_test_transforms '
                                 'or a transform for each test dataset')
            test_dataset_config['transforms'] = config['default_test_transforms']


def update_net_eval_mode(config):
    # If linear probing, then by default we want to turn off batchnorm while training.
    # In other words we want to use validation mode while training unless otherwise specified.
    linear_probe = 'linear_probe' in config and config['linear_probe']
    if linear_probe:
        if 'use_net_val_mode' not in config:
            config['use_net_val_mode'] = True
            logging.warning('Linear probing, so setting unspecified use_net_val_mode to True')


def setup():
    parser = argparse.ArgumentParser(
        description='Run model')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--log_dir', type=str, metavar='ld',
                        help='Log directory', required=True)
    parser.add_argument('--no_wandb', action='store_true', help='disable W&B')
    parser.add_argument('--copy_all_folders', action='store_true',
                        help='Copy all folders (e.g. code, utils) for reproducibility.')
    parser.add_argument('--project_name', type=str,
                        help='Name of the wandb project', required=True)
    parser.add_argument('--group_name', default=None, help='Name of the wandb group (a group of runs)')
    parser.add_argument('--run_name', default=None, help='Name of the wandb run')
    parser.add_argument('--entity_name', default='p-lambda', help='Name of the team')

    args, unparsed = parser.parse_known_args()
    log_dir = args.log_dir
    # Make log and checkpoint directories.
    make_new_dir(log_dir)
    checkpoints_dir = make_checkpoints_dir(log_dir)
    # If you want to copy folders to get the whole state of code
    # while running. For more reproducibility.
    if args.copy_all_folders:
        copy_folders(args.log_dir)
    # Setup logging.
    utils.setup_logging(log_dir, log_level)
    # Open config, update with command line args.
    config = quinine.Quinfig(args.config)
    utils.update_config(unparsed, config)
    # We make specifying some things more convenient - we don't need to specify
    # a transform for each test, but can specify a default transform.
    update_test_transform_configs(config)
    # If linear probing, by default we turn batch-norm off while training, if unspecified.
    update_net_eval_mode(config)
    # Note: copying config over is not that useful anymore with Quinine, so use json below.
    shutil.copy(args.config, log_dir+'/original_config.yaml')
    # Setup wandb.
    setup_wandb(args, config)
    # Save updated config.
    config_json = log_dir+'/config.json'
    with open(config_json, 'w') as f:
        json.dump(config, f)
    # Save command line arguments.
    save_command_line_args(log_dir)
    return config, log_dir, checkpoints_dir


if __name__ == "__main__":
    config, log_dir, checkpoints_dir = setup()
    main(config, log_dir, checkpoints_dir)
