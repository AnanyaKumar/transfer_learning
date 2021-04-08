import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SwAV
from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVTrainDataTransform, SwAVEvalDataTransform
)
from argparse import ArgumentParser


import math
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import distributed as dist
from torch import nn
from torch.optim.optimizer import Optimizer

from pl_bolts.models.self_supervised.swav.swav_resnet import resnet18, resnet50
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.transforms.dataset_normalizations import (
            cifar10_normalization, imagenet_normalization, stl10_normalization,
                    )



def cli_main():
    from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.swav.transforms import SwAVEvalDataTransform, SwAVTrainDataTransform
    from unlabeled_extrapolation.datasets.domainnet import DomainNetDataModule

    parser = ArgumentParser()

    # model args
    parser = SwAV.add_model_specific_args(parser)
    # see https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/swav/swav_module.py
    parser.add_argument('--checkpoint_dir', type=str, default='.', help="directory to save checkpoints")
    parser.add_argument('--domain', type=str, default='.', help="domain string")
    args = parser.parse_args()

    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)

    if args.dataset == 'stl10':
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False

        normalization = stl10_normalization()
    elif args.dataset == 'cifar10':
        args.batch_size = 2
        args.num_workers = 0

        dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False

        normalization = cifar10_normalization()

        # cifar10 specific params
        args.size_crops = [32, 16]
        args.nmb_crops = [2, 1]
        args.gaussian_blur = False
    elif args.dataset == 'imagenet':
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.size_crops = [224, 96]
        args.nmb_crops = [2, 6]
        args.min_scale_crops = [0.14, 0.05]
        args.max_scale_crops = [1., 0.14]
        args.gaussian_blur = True
        args.jitter_strength = 1.

        args.batch_size = 64
        args.num_nodes = 8
        args.gpus = 8  # per-node
        args.max_epochs = 800

        args.optimizer = 'sgd'
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3

        args.nmb_prototypes = 3000
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]
    elif args.dataset == 'domainnet':
        # copy imagenet params
        args.maxpool1 = True
        args.first_conv = True
        normalization = None # imagenet_normalization()

        args.size_crops = [224, 96]
        args.nmb_crops = [2, 6]
        args.min_scale_crops = [0.14, 0.05]
        args.max_scale_crops = [1., 0.14]
        args.gaussian_blur = True
        args.jitter_strength = 1.

        # args.batch_size = 256
        # args.gpus = 4  # per-node
        args.num_nodes = 1
        # args.max_epochs = 100

        args.optimizer = 'sgd'
        args.lars_wrapper = True
        args.learning_rate = 4.8
        args.final_lr = 0.0048
        args.start_lr = 0.3

        args.nmb_prototypes = 3000
        args.online_ft = True

        dm = DomainNetDataModule(batch_size=args.batch_size, num_workers=args.num_workers, train_domain=args.domain, test_domain=args.domain)
        args.num_samples = dm.num_unlabeled_samples
        args.input_height = dm.size()[-1]

    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SwAVTrainDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )

    dm.val_transforms = SwAVEvalDataTransform(
        normalize=normalization,
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength
    )

    # swav model init
    model = SwAV(**args.__dict__)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
        )

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint, online_evaluator] if args.online_ft else [model_checkpoint]

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
