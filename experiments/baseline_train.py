
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
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml

from models import resnet
from utils.accumulator import Accumulator 
import utils.utils as utils


log_level = logging.DEBUG


def get_test_stats(config, net, test_loader, criterion, device, max_examples=float('infinity')):
    # Evaluate accuracy and loss on validation.
    # Returns right after we've seen at least max_examples examples (not batches).
    val_loss = Accumulator()
    val_acc = Accumulator()
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
    return val_loss, val_acc


def main(config, log_dir, checkpoints_dir):
    # Set up datasets and loaders.
    train_data = utils.init_dataset(config['train_dataset'])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'])
    # Set up test loaders.
    logging.info('Found %d testing datasets.', len(config['test_datasets']))
    test_loaders = {}
    max_test_examples = {}
    for test_dataset_config in config['test_datasets']:
        logging.info('test dataset config: ' + str(test_dataset_config))
        # Use default test transform if test dataset doesn't specify a transform.
        if 'transforms' not in test_dataset_config:
            if config['default_test_transforms'] is None:
                raise ValueError('Must either specify default_test_transforms '
                                 'or a transform for each test dataset')
            test_dataset_config['transforms'] = config['default_test_transforms']
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
    # Create model.
    logging.info(f'cuda device count: {torch.cuda.device_count()}') 
    net = utils.initialize(config['model'])
    # If fine-tune, re-initialize the last layer.
    finetune = 'finetune' in config and config['finetune']
    linear_probe = 'linear_probe' in config and config['linear_probe']
    def count_parameters(model, trainable):
        return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)
    if finetune or linear_probe:
        if linear_probe:
            logging.info('linear probing, freezing bottom layers.')
            net.set_requires_grad(False)
        net.new_last_layer(config['num_classes'])
        num_trainable_params = count_parameters(net, True)
        num_params = count_parameters(net, False) + num_trainable_params
        logging.info(f'Fine Tuning {num_trainable_params} of {num_params} parameters.')
 
    # Use CUDA if desired. 
    if config['use_cuda']:
        device = "cuda"
        net.cuda()
    logging.info('Using cuda? %d', next(net.parameters()).is_cuda)
    # Loss, optimizer, scheduler.
    # Can use a custom loss that takes in a model, inputs, labels, and gets an array of values.
    # Or a criterion, which takes in model_outputs, labels, outputs a loss.
    criterion = utils.initialize(config['criterion'])
    if 'model_loss' in config:
        model_loss = utils.initialize(config['model_loss'])
    optimizer = utils.initialize(
            config['optimizer'], update_args={'params': net.parameters()})
    scheduler = utils.initialize(
            config['scheduler'], update_args={'optimizer': optimizer})
    # Training loop.
    best_acc = 0.0
    prev_ckp_path = None
    for epoch in range(config['epochs']):
        # Save checkpoint once in a while.
        if epoch % config['save_freq'] == 0:
            cur_ckp_filename = 'ckp_' + str(epoch)
            utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, cur_ckp_filename)
            if (prev_ckp_path is not None and not(config['save_all_checkpoints'])):
                os.remove(prev_ckp_path)
            prev_ckp_path = checkpoints_dir / cur_ckp_filename
        # Train model.
        net.train()
        logging.info("\nEpoch #{}".format(epoch))
        loss_dict = {
            'train/loss': Accumulator(),
            'train/acc': Accumulator(),
        }
        if 'model_loss' in config:
            loss_dict['train/model_loss'] = Accumulator()
        num_examples = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if config['use_cuda']:
                data = utils.to_device(data, device)
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if 'model_loss' in config:
                opt_loss = model_loss(net, inputs, labels)
                opt_loss.backward()
                loss_dict['train/model_loss'].add_value(opt_loss.tolist())
            else:
                loss.backward()
            optimizer.step() 
            loss_dict['train/loss'].add_value(loss.tolist())
            _, train_preds = torch.max(outputs.data, 1)
            loss_dict['train/acc'].add_values((train_preds == labels).tolist())
            num_examples += len(labels)
            def should_log(log_interval):
                return num_examples // log_interval > (num_examples - len(labels)) // log_interval
            if should_log(config['log_interval']):
                for k in loss_dict:
                    logging.info(
                        '[%d, %5d] %s: %.3f' %
                        (epoch + 1, num_examples, k, loss_dict[k].get_mean()))
            stats = {}
            if 'test_interval' in config and should_log(config['test_interval']):
                for name, test_loader in test_loaders.items():
                    max_examples = float('infinity')
                    if name in max_test_examples:
                        max_examples = max_test_examples[name]
                        logging.info(f'{name} test set processing max examples {max_examples}')
                    val_loss, val_acc = get_test_stats(
                        config, net, test_loader, criterion, device,
                        max_examples=max_examples)
                    stats['inter_ood_loss/' + name] = val_loss.get_mean()
                    stats['inter_ood_acc/' + name] = val_acc.get_mean()
                wandb.log(stats)
        
        scheduler.step()
        # Get loss for each test set
        stats = {}
        if 'test_interval' not in config:
            for name, test_loader in test_loaders.items():
                max_examples = float('infinity')
                if name in max_test_examples:
                    max_examples = max_test_examples[name]
                val_loss, val_acc = get_test_stats(
                    config, net, test_loader, criterion, device,
                    max_examples=max_examples)
                stats['ood_loss/' + name] = val_loss.get_mean()
                stats['ood_acc/' + name] = val_acc.get_mean()
        for k in loss_dict:
            stats[k] = loss_dict[k].get_mean()
        if config['wandb']:
            wandb.log(stats)
        utils.save_json(log_dir + '/current.json', stats)
        # Save checkpoint of best model.
        if val_acc.get_mean() > best_acc:
            best_acc = val_acc.get_mean()
            utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, 'ckp_best')
        logging.info('Accuracy of the network on the 10000 test images: %.2f %%' %
                     (100.0 * val_acc.get_mean()))
    utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, 'ckp_last')
    utils.save_json(log_dir + '/best.json', {'val_acc': best_acc})


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
    copy_folders = ['code', 'scripts', 'lib', 'configs', 'models',
                    'experiments', 'utils', 'examples', 'src',
                    'datasets']
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


def setup():
    parser = argparse.ArgumentParser(
        description='Run model')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--log_dir', type=str, metavar='ld',
                        help='Log directory', required=True)
    parser.add_argument('--no_wandb', action='store_true', help='disable W&B')
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
    # copy_folders(args.log_dir)
    # Setup logging.
    utils.setup_logging(log_dir, log_level)
    # Open config, update with command line args.
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    utils.update_config(unparsed, config)
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

