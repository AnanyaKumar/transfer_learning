#!/bin/bash


python ../unlabeled_extrapolation/pretrain_swav.py \
    --gpus 1 \
    --batch_size 32 \
    --dataset domainnet
