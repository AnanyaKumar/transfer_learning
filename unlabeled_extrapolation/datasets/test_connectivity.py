import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.models as models

import numpy as np
import random
import os

from unlabeled_extrapolation.datasets.connectivity_utils import *

import argparse
parser = argparse.ArgumentParser(description='Test Connectivity of dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset_name', type=str, required=True,
                    help='Which dataset on which to test connectivity.')
parser.add_argument('--source', type=str, required=True,
                    help='Name of source dataset')
parser.add_argument('--target', type=str, default=None,
                    help='Name of target dataset. If not provided, will be set to the '
                    'same as --source')
parser.add_argument('--data_path', type=str, default='/scr/biggest/imagenet',
                    help='Root path of the data')
parser.add_argument('--test_between', type=str, choices=['classes', 'domains'],
                    help='Whether to test intra- or inter- connectivity', required=True)
parser.add_argument('--transform', type=str, choices=['imagenet', 'simclr'], required=True)
parser.add_argument('--num_iters', default=15, type=int,
                    help='If doing class-comparison, the number of random pairs to choose.')
parser.add_argument('--seed', default=20, type=int, help='Seed for choosing pairs of classes.')

# training args
parser.add_argument('-a', '--arch', default='resnet50', help='Architecture')
parser.add_argument('-j', '--workers', default=2, type=int, help='Number of workers')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, dest='lr', help='LR')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay', help='WD')
parser.add_argument('--save-freq', type=int, default=25, help='How often to save')
parser.add_argument('--print-freq', type=int, default=5, help='How often to print')

def main(args):
    num_classes = validate_dataset(args.dataset_name, args.source, args.target)
    transform = get_transforms(args)
    save_dir = os.path.join(
        'connectivity_checkpoints',
        f'{args.source}-{args.target}',
        f'{args.test_between}-{args.transform}'
    )
    if args.test_between == 'classes':
        class_pairs = get_classes_to_compare(num_classes, args.num_iters, args.seed)
        for class_1, class_2 in class_pairs:
            # Do the source first
            source_train_ds, source_test_ds = get_class_datasets(args.dataset_name, args.source, class_1, class_2,
                                                                 transform, args.data_path, True)
            identifier = f'source-classes-{class_1}-{class_2}'
            main_loop(source_train_ds, source_test_ds, save_dir, identifier, args)
            # Now, do the target
            target_train_ds, target_test_ds = get_class_datasets(args.dataset_name, args.target, class_1, class_2,
                                                                 transform, args.data_path, False)
            identifier = f'target-classes-{class_1}-{class_2}'
            main_loop(target_train_ds, target_test_ds, save_dir, identifier, args)
    else: # between domains
        for class_idx in range(num_classes):
            train_ds, test_ds = get_domain_datasets(args.dataset_name, args.source, args.target, args.data_path,
                                                    class_idx, transform)
            identifier = f'class-{class_idx}'
            main_loop(train_ds, test_ds, save_dir, identifier, args)

def main_loop(train_ds, test_ds, save_dir, identifier, args):
    base_file_name = os.path.join(save_dir, identifier)
    if os.path.exists(f'{base_file_name}-final'):
        print(f'Already completed {base_file_name}, skipping...')
        return

    # get model
    model = models.__dict__[args.arch](num_classes=2)
    model = model.cuda()

    # get objective, optimizer, lr scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # get data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_acc = []
    test_acc = []
    for epoch in range(args.epochs):
        # train for one epoch
        train_acc.append(train_epoch(train_loader, model, criterion, optimizer, epoch, args))
        scheduler.step()

        # evaluate on test set
        test_acc.append(validate(test_loader, model, criterion, args))

        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'train_accs': train_acc,
                'test_accs': test_acc
            }, f'{base_file_name}-{epoch}')
            previous_file = f'{base_file_name}-{epoch - args.save_freq}'
            if os.path.exists(previous_file):
                os.remove(previous_file)

    torch.save({
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'train_accs': train_acc,
        'test_accs': test_acc
    }, f'{base_file_name}-final')

def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [loss_meter, acc_meter],
        prefix='Epoch: [{}]'.format(epoch))

    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output, target)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)
    return acc_meter.avg

def validate(test_loader, model, criterion, args):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [loss_meter, acc_meter],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = accuracy(output, target)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc[0], images.size(0))

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        print(' * Acc {acc.avg:.3f}'.format(acc=acc_meter))
    return acc_meter.avg

if __name__ == '__main__':
    args = parser.parse_args()

    if args.target is None:
        args.target = args.source

    main(args)