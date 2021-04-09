#!/bin/bash

set -x

SCR=$1
CHKPT_SCR=$2
LOCALDIR=$3
batch_size=$4
max_epochs=$5
domain=$6
gpus=$7
cpus=$8


mkdir -p $LOCALDIR

DOMAINNET_ZIP=/u/scr/nlp/domainnet/domainnet.zip
# COPY domainnet
if [ ! -d "${LOCALDIR}/domainnet" ]; then
  echo "copying file"
  cp "${DOMAINNET_ZIP}" "${LOCALDIR}"
  mkdir -p ${LOCALDIR}/domainnet
  unzip ${LOCALDIR}/domainnet.zip -d ${LOCALDIR}/domainnet
fi


python ${SCR}/unlabeled_extrapolation/pretrain_swav.py  \
    --gpus $gpus \
    --num_workers $cpus \
    --batch_size ${batch_size} \
    --max_epochs ${max_epochs} \
    --dataset domainnet \
    --fast_dev_run 0 \
    --checkpoint_dir ${CHKPT_SCR}/swav_domainnet_${domain} \
    --domain ${domain} \
    --data_dir $LOCALDIR/domainnet \
    --warmup_epochs 0 \
    --learning_rate 0.6 \
    --final_lr 0.0006 \
    --queue_length 3840 \
    --warmup_epochs 0 \
    --arch resnet50
    # some params copied from https://github.com/facebookresearch/swav/blob/master/scripts/swav_200ep_bs256_pretrain.sh

