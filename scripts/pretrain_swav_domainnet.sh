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
conda_env=$9


mkdir -p $LOCALDIR

DOMAINNET_ZIP=/u/scr/nlp/domainnet/domainnet.zip
# COPY domainnet
if [ ! -f "${LOCALDIR}/domainnet" ]; then
  echo "copying file"
  cp "${DOMAINNET_ZIP}" "${LOCALDIR}"
  unzip ${LOCALDIR}/${DOMAINNET_ZIP}
fi


source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate ${conda_env}

python ${SCR}/unlabeled_extrapolation/pretrain_swav.py  \
    --gpus $gpus \
    --num_workers $cpus \
    --batch_size ${batch_size} \
    --max_epochs ${max_epochs} \
    --dataset domainnet \
    --fast_dev_run 0 \
    --checkpoint_dir ${CHKPT_SCR}/swav_domainnet_${domain} \
    --domain ${domain} \
    --data_dir $LOCALDIR/domainnet
