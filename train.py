
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

import resnet
from utils.accumulator import Accumulator 
import utils.utils as utils


log_level = logging.DEBUG


def main(config, log_dir, checkpoints_dir):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    # Set up datasets and loaders.
    train_data = utils.init_dataset(config['train_dataset'])
    test_data = utils.init_dataset(config['test_dataset'])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'])
    # Use CUDA if desired.
    logging.info(f'cuda device count: {torch.cuda.device_count()}') 
    net = resnet.resnet18()
    if config['use_cuda']:
        device = "cuda"
        net.cuda()
    logging.info('Using cuda? %d', next(net.parameters()).is_cuda)
    # Loss, optimizer, scheduler.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr_init'], momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['epochs'])
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
        train_loss = Accumulator()
        train_acc = Accumulator()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if config['use_cuda']:
                data = utils.to_device(data, device)
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
            train_loss.add_value(loss.tolist())
            _, train_preds = torch.max(outputs.data, 1)
            train_acc.add_values((train_preds == labels).tolist())
            if i % config['log_interval'] == 0 and i != 0:
                logging.info('[%d, %5d] train_loss: %.3f' %
                        (epoch + 1, i + 1, train_loss.get_mean()))
                logging.info('[%d, %5d] train_acc: %.3f' %
                        (epoch + 1, i + 1, train_acc.get_mean()))
        scheduler.step()
        # Evaluate accuracy and loss on validation.
        val_loss = Accumulator()
        val_acc = Accumulator()
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                if config['use_cuda']:
                    data = utils.to_device(data, device)
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                val_acc.add_values((predicted == labels).tolist())
                loss = criterion(outputs, labels)
                val_loss.add_value(loss.tolist())
        # Save train and val stats to wandb and file.
        stats = {
            'epoch': epoch,
            'train_loss': train_loss.get_mean(),
            'train_acc': train_acc.get_mean(),
            'val_loss': val_loss.get_mean(),
            'val_acc': val_acc.get_mean(),
        }
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
                    'experiments', 'utils', 'examples', 'src']
    for copy_folder in copy_folders:
        if os.path.isdir('./' + copy_folder):
            shutil.copytree('./' + copy_folder, log_dir + '/' + copy_folder)


def time_to_str(ts):
    return pd.Timestamp(ts).strftime('%Y-%m-%dT%H-%M-%S-%f')


def now_to_str():
    return time_to_str(datetime.datetime.now())


def setup_wandb(args, config):
    if not args.no_wandb:
        wandb.init(project='cifar')
#        run_name = now_to_str() if args.run_name is None else args.run_name
#        run_obj = wandb.init(
#            project=args.project_name, name=run_name,
#            group=args.group_name, entity=args.entity_name, reinit=True)
#        config['wandb_url'] = run_obj.get_url()
#        config['run_name'] = run_name
#        config['group_name'] = group_name
#        config['entity_name'] = entity_name
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run model')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--log_dir', type=str, metavar='ld',
                        help='Log directory', required=True)
    parser.add_argument('--no_wandb', action='store_true', help='disable W&B')
    parser.add_argument('--project_name', default=None, help='Name of the wandb project')
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
    # Main loop.
    main(config, log_dir, checkpoints_dir)

