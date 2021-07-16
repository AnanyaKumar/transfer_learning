import torch
from torch.utils.data import Dataset

from data_utils import *
from train_utils import *

def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [loss_meter, acc_meter],
        prefix='Epoch: [{}]'.format(epoch))

    model.train()
    for i, (images, _) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)

        # get into one block
        images = torch.cat(images)
        features = model(images)

        labels = torch.cat([torch.arange(len(images) / 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = torch.nn.functional.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(len(features), dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)
        assert sim_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        
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

