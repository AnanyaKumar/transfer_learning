import argparse
import builtins
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

from torch.utils.data.distributed import DistributedSampler
from unlabeled_extrapolation.datasets.looping import LoopingDataset
from unlabeled_extrapolation.models import resnet
from unlabeled_extrapolation.utils.accumulator import Accumulator
import unlabeled_extrapolation.utils.utils as utils
import unlabeled_extrapolation.utils.multi as mp_utils


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
                logging.info("Breaking after %d examples.", num_examples)
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


def get_test_loaders(config, args):
    test_loaders = {}
    max_test_examples = {}
    
    should_log = (not args.mp) or (args.gpu == 0)
    if should_log:
        logging.info('Found %d testing datasets.', len(config['test_datasets']))
    for test_dataset_config in config['test_datasets']:
        if should_log: 
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
            if should_log:
                logging.info(
                    'Only logging %d examples for %s', test_dataset_config['max_test_examples'],
                    test_dataset_config['name'])
            max_test_examples[test_config_name] = test_dataset_config['max_test_examples']
        else:
            max_test_examples[test_config_name] = float('infinity')
        if should_log:
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
            # If unspecified, we set use_net_val_mode = True for linear-probing.
            # We did this in update_net_eval_mode which we called in main.
            assert('use_net_val_mode' in config)
            # Freeze all the existing weights of the neural network.
            # TODO: enable keeeping the top linear layer.
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


def train(epoch, args, config, train_source_loader, train_target_loader, net, pseudo_net, device, optimizer, criterion, model_loss,
          test_loaders, max_test_examples):
    # Returns a dictionary with epoch, train/loss and train/acc.
    # Train model.
    training_state = net.training
    if args.gpu == 0:
        logging.info("\nEpoch #{}".format(epoch))
    loss_dict = {
        'train/loss': Accumulator(),
        'train/acc': Accumulator(),
    }
    if 'model_loss' in config:
        loss_dict['train/model_loss'] = Accumulator()
    num_examples = 0
#     pseudolabels = []
    print('epoch: ', epoch)
    for i, (source_data, target_data) in enumerate(zip(train_source_loader, train_target_loader)):
        if 'use_net_val_mode' in config and config['use_net_val_mode']:
            net.eval()
        else:
            net.train()
        # get the inputs; data is a list of [inputs, labels]
        if config['use_cuda']:
            source_data = utils.to_device(source_data, device)
            target_data = utils.to_device(target_data, device)
        source_x, source_y = source_data
        target_x, _ = target_data
        
        # get pseudolabels
        target_outputs = pseudo_net(target_x)
        _, target_y = torch.max(target_outputs.data, dim=1)
#         pseudolabels.append(target_y)
        
        inputs = torch.cat([source_x, target_x])
        labels = torch.cat([source_y, target_y])
        
        # shuffle
        shuffled_indices = torch.randperm(len(inputs))
        inputs = inputs[shuffled_indices]
        labels = labels[shuffled_indices]
        
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
            return (args.gpu == 0) and num_examples // log_interval > (num_examples - len(labels)) // log_interval
        if should_log(config['log_interval']):
            for k in loss_dict:
                logging.info(
                    '[%d, %5d] %s: %.3f' %
                    (epoch + 1, num_examples, k, loss_dict[k].get_mean()))
    reset_state(net, training_state)
    train_stats = {}
    for key in loss_dict:
        train_stats[key] = loss_dict[key].get_mean()
#     pseudolabels = torch.cat(pseudolabels)
    
    return train_stats


def get_all_test_stats(stats, test_loaders, max_test_examples, config, net, criterion, device,
                   loss_name_prefix, acc_name_prefix):
    for name, test_loader in test_loaders.items():
        max_examples = float('infinity')
        if name in max_test_examples:
            max_examples = max_test_examples[name]
        val_loss, val_acc = get_test_stats(
            config, net, test_loader, criterion, device,
            max_examples=max_examples)
        # So this will look something like 'test_loss/cinic', 'test_acc/cinic'.
        stats[loss_name_prefix + name] = val_loss.get_mean()
        stats[acc_name_prefix + name] = val_acc.get_mean()
    return stats


def main(gpu, ngpus_per_node, config, args, log_dir, checkpoints_dir):
    args.gpu = gpu
    device = gpu
    if args.gpu == 0:
        logging.info("Entering main.")

    if args.mp:
        if args.gpu != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass
        mp_utils.init_proc_group(args, ngpus_per_node)
    # Set up datasets and loaders.
    train_data_configs = config['train_datasets']
    train_source_data = LoopingDataset(utils.init_dataset(config['train_datasets']['source']))
    train_target_data = LoopingDataset(utils.init_dataset(config['train_datasets']['target']))

    if args.mp:
        train_source_sampler = DistributedSampler(train_source_data)
        train_target_sampler = DistributedSampler(train_target_data)
    else:
        train_source_sampler = train_target_sampler = None
    
    train_source_loader = torch.utils.data.DataLoader(train_source_data, batch_size=config['batch_size']//2, 
                                                      shuffle=(not args.mp), num_workers=config['num_workers'], 
                                                      sampler=train_source_sampler)
    train_target_loader = torch.utils.data.DataLoader(train_target_data, batch_size=config['batch_size']//2,
                                                      shuffle=(not args.mp), num_workers=config['num_workers'], 
                                                      sampler=train_target_sampler)
    # Set up test loaders.
    test_loaders, max_test_examples = get_test_loaders(config, args)
    
    # Build models
    net = build_model(config)
    pseudo_net = build_model(config)
    
    if args.gpu == 0:
        logging.info(f'cuda device count: {torch.cuda.device_count()}') 
        logging.info('Using cuda? %d', config['use_cuda'])
    if config['use_cuda']:
        # Often makes things faster, by benchmarking and figuring out how to optimize.
        #cudnn.benchmark = True
        #device = "cuda"
        net.cuda()
        pseudo_net.cuda()
    
    # Load checkpoint for pseudolabeling and regular network
    checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])
    net = mp_utils.init_data_parallel(args, net, ngpus_per_node)
    pseudo_net.load_state_dict(checkpoint['state_dict'])
    pseudo_net = mp_utils.init_data_parallel(args, pseudo_net, ngpus_per_node)
    
    
    optimizer = utils.initialize(
            config['optimizer'], update_args={'params': net.parameters()})
    #optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = utils.initialize(config['scheduler'], update_args={'optimizer': optimizer})
    #scheduler.load_state_dict(checkpoint['scheduler'])
    
    
    # Set up loss optimizer, and scheduler
    criterion = utils.initialize(config['criterion'])
    model_loss = None
    if 'model_loss' in config:
        model_loss = utils.initialize(config['model_loss'])
    
    
    perc_labels_changed = float('inf')
    iters = 0
#     prev_psuedolabels = None
    best_stats = {}
    best_accs = {}  # Used to save checkpoints of best models on some datasets.
    train_metrics = []
    test_metrics = []
    while iters < config['max_iters']:      
        print('iters: ', iters)
        # Train with labeled source + pseudolabeled target

        for epoch in range(config['epochs']):
            # One epoch of model training.
            train_source_sampler.set_epoch(epoch)
            train_target_sampler.set_epoch(epoch)
            train_stats = train(
                epoch, args, config, train_source_loader, train_target_loader, net, pseudo_net, 
                device, optimizer, criterion, model_loss, test_loaders, max_test_examples)      
            scheduler.step()
            if args.gpu == 0:
                # Get test stats across all test sets.
                test_stats = get_all_test_stats(
                    {'epoch': epoch}, test_loaders, max_test_examples, config, net, criterion, device,
                    loss_name_prefix='test_loss/', acc_name_prefix='test_acc/')
            # Keep track of the best stats.
#             update_best_stats(train_stats, best_stats)
#             update_best_stats(test_stats, best_stats)
            # Log and save stats.
#             train_metrics.append(train_stats)
#             test_metrics.append(test_stats)
#             train_df = pd.DataFrame(train_metrics)
#             test_df = pd.DataFrame(test_metrics)
#             df = train_df.merge(test_df, on='epoch')
#             assert(len(df) == len(train_df) == len(test_df))
#             df.to_csv(log_dir + '/stats.tsv', sep='\t')
#             if config['wandb']:
#                 wandb.log(train_stats)
#                 wandb.log(test_stats)
#                 wandb.log(best_stats)
#             utils.save_json(log_dir + '/current_train.json', train_stats)
#             utils.save_json(log_dir + '/current_test.json', test_stats)
#             utils.save_json(log_dir + '/best.json', best_stats)
            # Save checkpoint of best model. We save the 'best' for each of a list
            # of specified valid datasets. For example, we might want to save the best
            # model according to in-domain validation metrics, but as an oracle, save
            # the best according to ood validation metrics (or a proxy ood metric).
            
        # per iter metrics
        if args.gpu == 0:
            train_stats['iter'] = iters
            test_stats = get_all_test_stats(
                    {'iter': iters}, test_loaders, max_test_examples, config, net, criterion, device,
                    loss_name_prefix='test_loss/', acc_name_prefix='test_acc/')
            train_metrics.append(train_stats)
            test_metrics.append(test_stats)
            train_df = pd.DataFrame(train_metrics)
            test_df = pd.DataFrame(test_metrics)
            df = train_df.merge(test_df, on='iter')
            assert(len(df) == len(train_df) == len(test_df))
            df.to_csv(log_dir + '/stats.tsv', sep='\t')
        
            # save
            state = {
                'iter': iters,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, os.path.join(checkpoints_dir, 'last'))

            # save iteration ckpt
            if iters % config['save_freq'] == 0:
                torch.save(state, os.path.join(checkpoints_dir, f'iter_{iters}'))
        
        # checkpoint = torch.load(os.path.join(checkpoints_dir, 'last'), map_location='cpu')
        pseudo_net.load_state_dict(net.state_dict())
        
        # increment
        iters += 1
#         print('perc_labels_changed: ', perc_labels_changed)


def make_new_dir(new_dir):
    if os.path.isdir(new_dir):
        raise ValueError('{} already exists.'.format(new_dir))
    os.makedirs(new_dir)


def make_checkpoints_dir(log_dir):
    checkpoints_dir = log_dir + '/checkpoints'
    checkpoints_dir = Path(checkpoints_dir).resolve().expanduser()
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
    os.makedirs(checkpoints_dir)
    return checkpoints_dir


def time_to_str(ts):
    return pd.Timestamp(ts).strftime('%Y-%m-%dT%H-%M-%S-%f')


def now_to_str():
    return time_to_str(datetime.datetime.now())

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


def update_root_prefix(config):
    # Go through test datasets, and train dataset. If root_prefix specified, then prepend that
    # to the root.
    def apply_root_prefix(dataset_config, root_prefix):
        if 'root' in dataset_config['args']:
            orig_root = dataset_config['args']['root']
            dataset_config['args']['root'] = root_prefix + '/' + orig_root
    if 'root_prefix' in config:
        root_prefix = config['root_prefix']
        logging.info("Adding root prefix %s to all roots.", root_prefix)
        apply_root_prefix(config['train_dataset'], root_prefix)
        for test_dataset_config in config['test_datasets']:
            apply_root_prefix(test_dataset_config, root_prefix)

def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed + 111)


def setup():
    parser = argparse.ArgumentParser(
        description='Run model')
    
    parser.add_argument('--checkpoint_path', type=str, default='logs/living_transfer_1/checkpoints/ckp_350')
    parser.add_argument('--dist_url_add', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=25)
    parser.add_argument('--mp', action='store_true')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--log_dir', type=str, metavar='ld',
                        help='Log directory', required=True)
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args, unparsed = parser.parse_known_args()
    log_dir = args.log_dir
    # Make log, checkpoint directories.
    make_new_dir(log_dir)
    checkpoints_dir = make_checkpoints_dir(log_dir)

    # Open config, update with command line args.
    config = quinine.Quinfig(args.config)
    utils.update_config(unparsed, config)
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    # If we don't specify a transform for some test datasets, but specify a default transform,
    # then use the default transform for that dataset. For datasets that do specify a transform
    # we use that and not the default transform.
    update_test_transform_configs(config)
   
    # Datasets may be stored in different directories in different clusters and platforms.
    # We allow specifying a root_prefix that gets prepended to any specified dataset roots.
    # So if config['root_prefix'] is defined then we prepend it to dataset['args']['root'] for
    # train and test datasets.
    update_root_prefix(config)
    # Note: copying config over is not that useful anymore with Quinine, so use json below.
    shutil.copy(args.config, log_dir+'/original_config.yaml')
#     # Set seed.
#     config['seed'] = args.seed
#     config['checkpoint_path'] = args.checkpoint_path
    set_random_seed(args.seed)
    # Save updated config.
    config_json = log_dir+'/config.json'
    with open(config_json, 'w') as f:
        json.dump(config, f)
    # Save command line arguments.
    save_command_line_args(log_dir)
    return config, log_dir, checkpoints_dir, args

if __name__ == "__main__":
    config, log_dir, checkpoints_dir, args = setup()
    if args.mp:
        args.multiprocessing_distributed = True
        args.world_size = 1
        args.rank = 0
        args.dist_url = 'tcp://127.0.0.1:' + str(10001 + args.dist_url_add)
        args.dist_backend = 'nccl'
    else:
        args.multiprocessing_distributed = False 
        args.world_size = 0
        args.rank = 0
        args.dist_url = None 
        args.dist_backend = None 
    args.batch_size = config['batch_size']
    args.workers = config['num_workers']
    print(args)
    mp_utils.spawn_processes(main, args, mpargs=(config, args, log_dir, checkpoints_dir))
    #main(config, args, log_dir, checkpoints_dir)

