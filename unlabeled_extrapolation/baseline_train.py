import argparse
import datetime
import json
import logging
import os
import os.path
import pandas as pd
from pathlib import Path
import pickle
import shutil
import socket
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
import numpy as np

from unlabeled_extrapolation.models import resnet
from unlabeled_extrapolation.utils.accumulator import Accumulator
import unlabeled_extrapolation.utils.utils as utils
from timm.data.mixup import Mixup

log_level = logging.INFO


def reset_state(model, training):
    if training:
        model.train()
    else:
        model.eval()


# In some datasets, only some indices are valid.
# So we only take the argmax among the set of valid indices.
def get_argmax_valid(logits, valid_indices):
    compressed_logits = logits[:, valid_indices]
    assert compressed_logits.shape == (logits.shape[0], len(np.unique(valid_indices)))
    compressed_preds = np.argmax(compressed_logits, axis=-1).astype(np.int32)
    valid_indices = np.array(valid_indices)
    preds = valid_indices[compressed_preds]
    assert preds.shape == (logits.shape[0],)
    return preds


def get_test_stats(config, net, test_loader, criterion, device, epoch, loader_name, log_dir,
                   max_examples=float('infinity')):
    # Evaluate accuracy and loss on validation.
    # Returns right after we've seen at least max_examples examples (not batches).
    val_loss = Accumulator()
    val_acc = Accumulator()
    training_state = net.training
    net.eval()
    num_examples = 0
    if 'save_model_preds' in config and config.save_model_preds:
        predicted_list = []
        labels_list = []
    with torch.no_grad():
        for data in test_loader:
            if config['use_cuda']:
                data = utils.to_device(data, device)
            images, labels = data
            outputs = net(images)
            if hasattr(test_loader.dataset, 'valid_indices'):
                # This basically projects onto the set of valid indices.
                # We take the argmax among the set of valid indices.
                logits = outputs.data.detach().cpu().numpy()
                predicted = get_argmax_valid(logits, test_loader.dataset.valid_indices)
            else:
                _, predicted = torch.max(outputs.data, dim=1)
                predicted = predicted.detach().cpu().numpy()
            if 'save_model_preds' in config and config.save_model_preds:
                predicted_list.append(predicted)
                labels_list.append(labels.detach().cpu().numpy())
            correct = (predicted == labels.detach().cpu().numpy())
            val_acc.add_values(correct.tolist())
            loss = criterion(outputs, labels).cpu()
            val_loss.add_values([float(loss.detach().cpu())] * len(images))
            num_examples += len(images)
            if num_examples >= max_examples:
                logging.info("Breaking after %d examples.", num_examples)
                break
    if 'save_model_preds' in config and config.save_model_preds:
        preds = np.concatenate(predicted_list)
        labels = np.concatenate(labels_list)
        pickle_name = log_dir+'/model_preds/'+loader_name+'_'+str(epoch)+'_preds.pkl'
        pickle.dump((preds, labels), open(pickle_name, "wb"))
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


def get_test_loaders(config, shuffle=False):
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
            shuffle=shuffle, num_workers=config['num_workers'])
        test_config_name = test_dataset_config['name']
        test_loaders[test_config_name] = test_loader
        # Some test datasets like CINIC are huge so we only test part of the dataset.
        if 'max_test_examples' in test_dataset_config:
            logging.info(
                'Only logging %d examples for %s', test_dataset_config['max_test_examples'],
                test_dataset_config['name'])
            max_test_examples[test_config_name] = test_dataset_config['max_test_examples']
        else:
            max_test_examples[test_config_name] = float('infinity')
        logging.info('test loader name: ' + test_dataset_config['name'])
        logging.info('test loader: ' + str(test_loader))
        logging.info('test transform: ' + str(test_dataset_config['transforms']))
    return test_loaders, max_test_examples


def get_train_loader(config, shuffle=True):
    train_data = utils.init_dataset(config['train_dataset'])
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config['batch_size'],
        shuffle=shuffle, num_workers=config['num_workers'])
    return train_loader


def count_parameters(model, trainable):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)


def build_model(config):
    net = utils.initialize(config['model'])
    # If fine-tune, re-initialize the last layer.
    finetune = 'finetune' in config and config['finetune']
    linear_probe = 'linear_probe' in config and config['linear_probe']
    freeze_bottom_k = 'freeze_bottom_k' in config
    batch_norm = 'batchnorm_ft' in config and config['batchnorm_ft']
    side_tune = 'side_tune' in config and config['side_tune']
    if finetune or linear_probe or batch_norm or side_tune:
        if freeze_bottom_k:
            # Currently only implemented for some models (including CLIP ViTs).
           net.freeze_bottom_k(config['freeze_bottom_k']) 
        if linear_probe:
            logging.info('linear probing, freezing bottom layers.')
            # If unspecified, we set use_net_val_mode = True for linear-probing.
            # We did this in update_net_eval_mode which we called in main.
            assert('use_net_val_mode' in config)
            # Freeze all the existing weights of the neural network.
            # TODO: enable keeeping the top linear layer.
            net.set_requires_grad(False)
        if batch_norm:
            assert(not linear_probe)
            logging.info("tuning only batch norm layers and lin probe")
            net.set_requires_grad(False)
            for layer in net._model.modules():
                if isinstance(layer, nn.modules.batchnorm.BatchNorm2d): 
                    for param in layer.parameters():
                        param.requires_grad = True 
        if 'probe_net' in config:
            probe_net = utils.initialize(config['probe_net'])
            net.add_probe(probe_net)
        else:
            net.new_last_layer(config['num_classes'])
        if side_tune:
            # This is currently only supported for some networks like ResNet-50,
            # would need to add support for other networks.
            net.enable_side_tuning()
        if ('linear_probe_checkpoint_path' in config and
            config['linear_probe_checkpoint_path'] != ''):
            linprobe_path = config['linear_probe_checkpoint_path']
            coef, intercept, best_c, best_i = pickle.load(open(linprobe_path, "rb"))
            if coef.shape[0] == 1:
                # For binary classification, sklearn returns a 1-d weight
                # vector. So we convert it into a 2D vector with the same
                # logits. To see this conversion, notice that if I have a
                # binary weight vector w, the output is \sigma(w^T x)
                # = e^(w^T x) / (1 + e^(w^T x)). On the other hand,
                # if I have weight vector [w/2, -w/2], and I use softmax
                # I get e^(w^T x / 2) / (e^(w^T x / 2) + e^(-w^T x / 2)
                # and multiplying num / denom by e^(w^T x / 2)
                # we get the same expression as above.
                coef = np.concatenate((-coef/2, coef/2), axis=0)
                intercept = np.array([-intercept[0]/2, intercept[0]/2])
            if 'normalize_lp' in config and config['normalize_lp']:
                logging.info("Normalizing linear probe std-dev")
                saved_stddev = np.std(coef)
                rand_weights = net.get_last_layer().weight.detach().numpy()
                rand_stddev = np.std(rand_weights)
                logging.info(
                    "Weights saved stddev: %f, desired stddev: %f",
                    saved_stddev, rand_stddev)
                logging.info("Intercept saved stddev: %f", np.std(intercept))
                coef = (coef / saved_stddev) * rand_stddev
                intercept = (intercept / saved_stddev) * rand_stddev
                logging.info(
                    "Final stddev weights; %f, intercept: %f",
                    np.std(coef), np.std(intercept))
                # What should it be, based on rep size (maybe get from net last layer)
                # Divide
            net.set_last_layer(coef, intercept)
        num_trainable_params = count_parameters(net, True)
        num_params = count_parameters(net, False) + num_trainable_params
        logging.info(f'Fine Tuning {num_trainable_params} of {num_params} parameters.')
    if 'checkpoint_path' in config and len(config['checkpoint_path']) > 0:
        logging.info(utils.load_ckp(config['checkpoint_path'], net))
        num_trainable_params = count_parameters(net, True)
        num_params = count_parameters(net, False) + num_trainable_params
        logging.info(f'Fine Tuning checkpoint: {num_trainable_params} of {num_params} parameters.')
    return net


def get_l2_dist(weight_dict1, weight_dict2, ignore='.fc.'):
    l2_dist = torch.tensor(0.0).cuda()
    for key in weight_dict1:
        if ignore not in key:
            l2_dist += torch.sum(torch.square(weight_dict1[key] - weight_dict2[key]))
    return l2_dist


def get_param_weights_counts(net, detach):
    weight_dict = {}
    count_dict = {}
    for param in net.named_parameters():
        name = param[0]
        weights = param[1]
        if detach:
            weight_dict[name] = weights.detach().clone()
        else:
            weight_dict[name] = weights
        count_dict[name] = np.prod(np.array(list(param[1].shape)))
    return weight_dict, count_dict


def set_requires_grad(component, val):
    for param in component.parameters():
        param.requires_grad = val


_grad_layer_name = 'train/grad_layer_'
_normalized_grad_layer_name = 'train/normalized_grad_layer_'


def add_layer_grad_stats(net, loss_dict, config, cur_batch_size):
    if getattr(net, 'get_layers', None) is None or 'no_log_grads' in config:
        return
    layers = net.get_layers()
    for i in range(len(layers)):
        if len(layers[i]) == 1:
            # In the older version of model code, we just return a list of layers without names.
            layer_name = 'train/grad_layer_' + str(i)
            normalized_layer_name = 'train/normalized_grad_layer_' + str(i)
            cur_layer = list(layers[i].parameters())
        else:
            # In newer version of model code, we have a (name, layer) tuple.
            layer_name = 'train/grad_' + layers[i][0]
            normalized_layer_name = 'train/normalized_grad_' + layers[i][0]
            cur_layer = list(layers[i][1].parameters())
        if len(cur_layer) == 0 or not cur_layer[0].requires_grad:
            continue
        if layer_name not in loss_dict:
            loss_dict[layer_name] = Accumulator()
        if normalized_layer_name not in loss_dict:
            loss_dict[normalized_layer_name] = Accumulator()
        grads = [p.grad.detach().cpu().numpy() for p in cur_layer]
        grad_norms_squared = [np.linalg.norm(g) ** 2 for g in grads]
        grad_norm = np.sqrt(np.sum(grad_norms_squared)) / cur_batch_size
        loss_dict[layer_name].add_value(grad_norm)
        num_params = np.sum([p.numel() for p in cur_layer])
        if num_params == 0:
            assert np.isclose(grad_norm, 0.0)
            normalized_grad_norm = grad_norm
        else:
            normalized_grad_norm = grad_norm / np.sqrt(num_params)
        loss_dict[normalized_layer_name].add_value(normalized_grad_norm)


def train(epoch, config, train_loader, net, device, optimizer, criterion, model_loss,
          test_loaders, max_test_examples, weight_dict_initial, batch_scheduler=None):
    # Returns a dictionary with epoch, train/loss and train/acc.
    # Train model.
    training_state = net.training
    if 'use_net_val_mode' in config and config['use_net_val_mode']:
        net.eval()
    else:
        net.train()
    logging.info("\nEpoch #{}".format(epoch))
    loss_dict = {
        'train/loss': Accumulator(),
        'train/acc': Accumulator(),
    }
    if 'l2sp_weight' in config:
        loss_dict['train/l2sp_loss'] = Accumulator()
    if 'model_loss' in config:
        loss_dict['train/model_loss'] = Accumulator()
    num_examples = 0
    if 'use_mixup' in config and config['use_mixup']:
        logging.info('Using mixup')
        # Hack for older versions of torch. Use KL-div loss instead of cross-entropy,
        # because mixup produces soft labels.
        if 'mixup_alpha' in config:
            mixup = Mixup(num_classes=config['num_classes'], mixup_alpha=config['mixup_alpha'])
        else:
            mixup = Mixup(num_classes=config['num_classes'])
    # Should we split each batch into multiple parts so we don't run out of GPU memory.
    batch_splits = 1
    if check_exists_not_none(config, 'batch_splits'):
        batch_splits = config['batch_splits']
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        all_inputs, all_labels = data
        if 'use_mixup' in config and config['use_mixup']:
            all_inputs, all_labels = mixup(all_inputs, all_labels)
        if 'l2sp_weight' in config:
            weight_dict, _ = get_param_weights_counts(net, detach=False)
            l2sp_loss = config['l2sp_weight'] * get_l2_dist(weight_dict_initial, weight_dict)
            loss_dict['train/l2sp_loss'].add_value(l2sp_loss.tolist())
            l2sp_loss.backward()
            del l2sp_loss
        # Split up the batch so we can do the backward pass on a GPU, accumulate gradients
        # across the split.
        split_inputs = torch.split(
            all_inputs, split_size_or_sections=len(all_inputs) // batch_splits)
        split_labels = torch.split(
            all_labels, split_size_or_sections=len(all_labels) // batch_splits)
        for j in range(batch_splits):
            inputs, labels = split_inputs[j], split_labels[j]
            if config['use_cuda']:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # Important: nede to adjust the loss. This assumes the reduction operation is
            # mean.
            if config["criterion"]["args"]["reduction"] != 'sum':
                loss = loss * len(inputs) / len(all_inputs)
            _, train_preds = torch.max(outputs.data, axis=1)
            assert len(loss.shape) == 0
            loss_dict['train/loss'].add_values([float(loss.detach().cpu())] * len(inputs))
            if 'use_mixup' in config and config['use_mixup']:
                _, max_labels = torch.max(labels.data, axis=1)
                loss_dict['train/acc'].add_values((train_preds == max_labels).tolist())
            else:
                loss_dict['train/acc'].add_values((train_preds == labels).tolist())
            if 'model_loss' in config:
                opt_loss = model_loss(net, inputs, labels)
                opt_loss.backward()
                assert len(opt_loss.shape) == 0
                loss_dict['train/model_loss'].add_value(float(opt_loss.detach().cpu()))
                del opt_loss
            else:
                loss.backward()
            del inputs, labels, loss
        # Collect the gradients at each layer.
        add_layer_grad_stats(net, loss_dict, config, cur_batch_size=len(all_labels))
        if check_exists_not_none(config, 'max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['max_grad_norm'])
        optimizer.step()
        if batch_scheduler is not None:
            batch_scheduler.step()
        num_examples += len(all_labels)
        outputs, loss, train_preds = None, None, None  # Try to force garbage collection.
        def should_log(log_interval):
            return num_examples // log_interval > (num_examples - len(all_labels)) // log_interval
        if should_log(config['log_interval']):
            for k in loss_dict:
                logging.info(
                    '[%d, %5d] %s: %.3f' %
                    (epoch + 1, num_examples, k, loss_dict[k].get_mean()))
        # Sometimes we want to log the test loss more often to track things better.
        if 'test_interval' in config and should_log(config['test_interval']):
            stats = get_all_test_stats(
                test_loaders, max_test_examples, config, net, criterion, device,
                log_dir=log_dir, loss_name_prefix='inter_test_loss/', acc_name_prefix='inter_test_acc/')
            if config['wandb']:
                wandb.log(stats)
    reset_state(net, training_state)
    train_stats = {'epoch': epoch}
    if batch_scheduler is not None:
        train_stats['train/batch_scheduler_lr'] = batch_scheduler.get_lr()
    for key in loss_dict:
        train_stats[key] = loss_dict[key].get_mean()
    return train_stats


def get_all_test_stats(epoch, test_loaders, max_test_examples, config, net, criterion, device,
                       log_dir, loss_name_prefix, acc_name_prefix):
    stats = {'epoch': epoch}
    for name, test_loader in test_loaders.items():
        logging.info(f'testing {name}')
        max_examples = float('infinity')
        if name in max_test_examples:
            max_examples = max_test_examples[name]
        val_loss, val_acc = get_test_stats(
            config, net, test_loader, criterion, device, epoch, name,
            log_dir=log_dir, max_examples=max_examples)
        # So this will look something like 'test_loss/cinic', 'test_acc/cinic'.
        stats[loss_name_prefix + name] = val_loss.get_mean()
        stats[acc_name_prefix + name] = val_acc.get_mean()
    return stats


def get_params(layers):
    params = []
    for layer in layers:
        for param in layer.parameters():
            params.append(param)
    return params


def check_exists_not_none(d, k):
    return k in d and d[k] != None


def check_exists_value(d, k, v):
    return k in d and d[k] == v


def main(config, log_dir, checkpoints_dir):
    # Set up datasets and loaders.
    logging.info("Entering main.")
    train_loader = get_train_loader(config)  
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
    if 'linear_layer_lr_multiplier' in config:
        base_params = get_params(list(net._model.children())[:-1])
        fc_params = get_params(list(net._model.children())[-1:])
        base_lr = config['optimizer']['args']['lr']
        fc_lr = config['linear_layer_lr_multiplier'] * base_lr
        logging.info('Using lr %f for fc layer, %d params', fc_lr, len(fc_params))
        param_groups = [
            {'params': base_params},
            {'params': fc_params, 'lr': fc_lr}
        ]
        optimizer = utils.initialize(
            config['optimizer'], update_args={'params': param_groups})
    else:
        optimizer = utils.initialize(
            config['optimizer'], update_args={'params': net.parameters()}) 
    # If batch scheduler is specified, then use a batch scheduler that updates the learning
    # rate every step. Otherwise, update the learning rate every epoch.
    batch_scheduler = None
    scheduler = None
    if check_exists_not_none(config, 'batch_scheduler'):
        # TODO: add batch scheduler.
        num_batches = len(train_loader)
        num_training_steps  = num_batches * config['epochs']
        if check_exists_not_none(config, 'warmup_frac'):
            num_warmup_steps = int(config['warmup_frac'] * num_training_steps)
        else:
            num_warmup_steps = int(0.1 * num_training_steps)
            logging.info("Warming up for %d steps", num_warmup_steps)
        batch_scheduler = utils.initialize(
            config['batch_scheduler'], update_args={
                'optimizer': optimizer,
                'num_warmup_steps': num_warmup_steps,
                'num_training_steps': num_training_steps,
        })
    else:
        scheduler = utils.initialize(
            config['scheduler'], update_args={'optimizer': optimizer})
    # Training loop.
    best_stats = {}
    best_accs = {}  # Used to save checkpoints of best models on some datasets.
    train_metrics = []
    test_metrics = []
    prev_ckp_path = None
    # For the first epoch we get stats, but we don't save them or update best checkpoint.
    # This can be useful if we start with a great model that quickly gets corrupted in the first
    # epoch, we don't want the initial statistics to dominate.
    # First argument is epoch.
    test_stats = get_all_test_stats(
        0, test_loaders, max_test_examples, config, net, criterion, device,
        log_dir=log_dir, loss_name_prefix='test_loss/', acc_name_prefix='test_acc/')
    # Log stats.
    logging.info('Initial test stats')
    logging.info(test_stats)
    # If l2sp, then save initial weights so we can regularize towards them.
    weight_dict_initial = None
    if 'l2sp_weight' in config:
        weight_dict_initial, _ = get_param_weights_counts(net, detach=True)

    for epoch in range(config['epochs']):
        # Save checkpoint once in a while.
        if epoch % config['save_freq'] == 0 and (
            'save_no_checkpoints' not in config or not config['save_no_checkpoints']):
            cur_ckp_filename = 'ckp_' + str(epoch)
            # TODO: update to save batch scheduler if specified?
            utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, cur_ckp_filename)
            if (prev_ckp_path is not None and not(config['save_all_checkpoints'])):
                os.remove(prev_ckp_path)
            prev_ckp_path = checkpoints_dir / cur_ckp_filename
        # User might specify an epoch for us to fully fine-tune the model.
        if "full_ft_epoch" in config and config["full_ft_epoch"] is not None:
            if epoch == config["full_ft_epoch"]:
                set_requires_grad(net, True) 
                num_trainable_params = count_parameters(net, True)
                num_params = count_parameters(net, False) + num_trainable_params
                assert(num_trainable_params == num_params)
                logging.info(f'Full FT {num_trainable_params} of {num_params} parameters.')
        # One epoch of model training.
        train_stats = train(
           epoch, config, train_loader, net, device, optimizer, criterion, model_loss,
           test_loaders, max_test_examples, weight_dict_initial, batch_scheduler) 
        # Call scheduler to update learning rate, unless we have a batch scheduler, in which
        # case we will update the learning rate every step.
        if not check_exists_not_none(config, 'batch_scheduler'):
            scheduler.step()
        # Get test stats across all test sets.
        test_stats = get_all_test_stats(
            epoch, test_loaders, max_test_examples, config, net, criterion, device,
            log_dir=log_dir, loss_name_prefix='test_loss/', acc_name_prefix='test_acc/')
        # Keep track of the best stats.
        update_best_stats(train_stats, best_stats)
        update_best_stats(test_stats, best_stats)
        # Log and save stats.
        train_metrics.append(train_stats)
        test_metrics.append(test_stats)
        train_df = pd.DataFrame(train_metrics)
        test_df = pd.DataFrame(test_metrics)
        df = train_df.merge(test_df, on='epoch')
        assert(len(df) == len(train_df) == len(test_df))
        df.to_csv(log_dir + '/stats.tsv', sep='\t')
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
                    if 'save_no_checkpoints' not in config or not config['save_no_checkpoints']:
                        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, checkpoint_name)
    if 'save_no_checkpoints' not in config or not config['save_no_checkpoints']:
        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, 'ckp_last')


def make_new_dir(new_dir, remove_old_ok=True):
    if os.path.isdir(new_dir):
        logging.warning("Removed old run directory.")
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)


def make_checkpoints_dir(log_dir):
    checkpoints_dir = log_dir + '/checkpoints'
    checkpoints_dir = Path(checkpoints_dir).resolve().expanduser()
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
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


def update_train_transform(config):
    if 'no_augmentation' in config and config['no_augmentation']:
        if 'default_test_transforms' not in config:
            raise ValueError('If no_augmentation=True, must specify default_test_transforms.')
        config['train_dataset']['transforms'] = config['default_test_transforms']


def update_test_transform_args_configs(config):
    # Use default test transform for test datasets that don't specify a transform.
    for test_dataset_config in config['test_datasets']:
        if 'transforms' not in test_dataset_config:
            if config['default_test_transforms'] is None:
                raise ValueError('Must either specify default_test_transforms '
                                 'or a transform for each test dataset')
            test_dataset_config['transforms'] = config['default_test_transforms']
        if 'default_test_args' in config and config['default_test_args'] is not None:
            for default_test_arg in config['default_test_args']:
                if default_test_arg not in test_dataset_config['args']:
                    test_dataset_config['args'][default_test_arg] = config['default_test_args'][default_test_arg]


def update_root_prefix(config):
    # Go through test datasets, and train dataset. If root_prefix specified, then prepend that
    # to the root.
    def apply_root_prefix(dataset_config, root_prefix):
        for key in ['root', 'cache_path', 'pickle_file_path']:
            if key in dataset_config['args']:
                orig_path = dataset_config['args'][key]
                logging.info('orig_path %s', orig_path)
                dataset_config['args'][key] = root_prefix + '/' + orig_path

    if 'root_prefix' in config:
        root_prefix = config['root_prefix']
        logging.info("Adding root prefix %s to all roots.", root_prefix)
        apply_root_prefix(config['train_dataset'], root_prefix)
        for test_dataset_config in config['test_datasets']:
            apply_root_prefix(test_dataset_config, root_prefix)


def update_net_eval_mode(config):
    # If linear probing, then by default we want to turn off batchnorm while training.
    # In other words we want to use validation mode while training unless otherwise specified.
    linear_probe = 'linear_probe' in config and config['linear_probe']
    if linear_probe:
        if 'use_net_val_mode' not in config:
            config['use_net_val_mode'] = True
            logging.warning('Linear probing, so setting unspecified use_net_val_mode to True')


def update_dataset_names(config):
    if 'overwrite_dataset_name' in config and config['overwrite_dataset_name'] is not None:
        new_dataset_name = config['overwrite_dataset_name']
        for test_dataset_config in config['test_datasets']:
            test_dataset_config['args']['dataset_name'] = new_dataset_name
        config['train_dataset']['args']['dataset_name'] = new_dataset_name


def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed + 111)


def update_optimizer_args(config):
    if config['optimizer']['classname'] in [
        'torch.optim.Adam', 'torch.optim.AdamW', 'torch.optim.RMSprop']:
        # These optimizers don't have a momentum term.
        # A bit of a hack so we can just pass in torch.optim.Adam as a command line arg,
        # without having to rework the config.
        del config['optimizer']['args']['momentum']


def preprocess_config(config, config_path):
    # If it's not a json config (e.g. if it's yaml) then process it. 
    if not config_path.endswith('.json'):
        # If we don't specify a transform for some test datasets, but specify a default transform,
        # then use the default transform for that dataset. For datasets that do specify a transform
        # we use that and not the default transform.
        update_test_transform_args_configs(config)
        # If linear probing, by default we turn batch-norm off while training, if unspecified.
        # If you want bach-norm even when lin probing then set use_net_val_mode to False in the config.
        update_net_eval_mode(config)
        # Datasets may be stored in different directories in different clusters and platforms.
        # We allow specifying a root_prefix that gets prepended to any specified dataset roots.
        # So if config['root_prefix'] is defined then we prepend it to dataset['args']['root'] for
        # train and test datasets.
        update_root_prefix(config)
        # # Note: copying config over is not that useful anymore with Quinine, so use json below.
        # shutil.copy(args.config, log_dir+'/original_config.yaml')
        # If no_augmentation option in config, then use test_transforms for training.
        update_train_transform(config)
        # Update dataset names, if overwrite_dataset_name is specified. This is useful
        # if we want to run the same experiments on variants of a dataset (e.g., 
        # waterbirds-background (vs. original waterbirds experiment which uses foreground).
        update_dataset_names(config)
        # Update optimizer arguments.
        update_optimizer_args(config)


def setup():
    parser = argparse.ArgumentParser(
        description='Run model')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--log_dir', type=str, metavar='ld',
                        help='Log directory', required=True)
    parser.add_argument('--tmp_par_ckp_dir', type=str,
                        help='Temporary directory to save checkpoints instead of log_dir.')
    parser.add_argument('--no_wandb', action='store_true', help='disable W&B')
    parser.add_argument('--copy_all_folders', action='store_true',
                        help='Copy all folders (e.g. code, utils) for reproducibility.')
    parser.add_argument('--project_name', type=str,
                        help='Name of the wandb project', required=True)
    parser.add_argument('--group_name', default=None, help='Name of the wandb group (a group of runs)')
    parser.add_argument('--run_name', default=None, help='Name of the wandb run')
    parser.add_argument('--entity_name', default='p-lambda', help='Name of the team')
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    args, unparsed = parser.parse_known_args()
    log_dir = args.log_dir
    # Make log and checkpoint directories.
    make_new_dir(log_dir)
    # Sometimes we don't want to overload a distributed file system with checkpoints.
    # So we save checkpoints on a tmp folder on a local machine. Then later we transfer
    # the checkpoints back.
    if args.tmp_par_ckp_dir is not None:
        checkpoints_dir = make_checkpoints_dir(args.tmp_par_ckp_dir)
    else:
        checkpoints_dir = make_checkpoints_dir(log_dir)
    # If you want to copy folders to get the whole state of code
    # while running. For more reproducibility.
    if args.copy_all_folders:
        copy_folders(args.log_dir)
    # Setup logging.
    utils.setup_logging(log_dir, log_level)
    logging.info('Running on machine %s', socket.gethostname())
    # Open config, update with command line args
    if args.config.endswith('.json'):
        # For json files, we just use it directly and don't process it, e.g. by adding
        # root_prefix. Use this for loading saved configurations.
        with open(args.config) as json_file:
            config = json.load(json_file)
    else:
        config = quinine.Quinfig(args.config)
    # Update config with command line arguments.
    utils.update_config(unparsed, config)
    # This makes specifying certain things more convenient, e.g. don't have to specify a
    # transform for every test datset.
    preprocess_config(config, args.config) 
    # If we should save model preds, then save them.
    if 'save_model_preds' in config and config.save_model_preds:
        os.makedirs(log_dir + '/model_preds/')
    # Setup wandb.
    setup_wandb(args, config)
    # Set seed.
    config['seed'] = args.seed
    set_random_seed(args.seed)
    # Save updated config.
    config_json = log_dir+'/config.json'
    with open(config_json, 'w') as f:
        json.dump(config, f)
    # Save command line arguments.
    save_command_line_args(log_dir)
    return config, log_dir, checkpoints_dir, args.tmp_par_ckp_dir


if __name__ == "__main__":
    config, log_dir, checkpoints_dir, tmp_par_ckp_dir = setup()
    main(config, log_dir, checkpoints_dir)
    # Tear down, i.e. copy checkpoints over from local machine to juice.
    if tmp_par_ckp_dir is not None:
        new_checkpoints_dir = log_dir + '/checkpoints'
        logging.info('Copying from %s to %s', checkpoints_dir, new_checkpoints_dir)
        shutil.copytree(checkpoints_dir, new_checkpoints_dir)
    os.system('rm -r pymp*')

