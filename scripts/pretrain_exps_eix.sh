#!/bin/bash

SCR=/u/scr/eix/unlabeled_extrapolation
CHKPT_SCR=/u/scr/eix/unlabeled_extrapolation/checkpoints
LOCALDIR=/scr/scr-with-most-space/eix
mkdir -p $SCR/logs
mkdir -p $CHKPT_SCR
partition=jag-hi
exclude=jagupard[4-8]
gpus=4
mem=64G
cpus=16

batch_size=128
max_epochs=200
domain=all
conda_env=eix-ue

sbatch --exclude $exclude --partition $partition --gres=gpu:${gpus} --mem $mem -c $cpus --output $SCR/logs/swav_domainnet_all /u/scr/eix/run_sbatch.sh \
"bash ${SCR}/scripts/pretrain_swav_domainnet.sh ${SCR} ${CHKPT_SCR} ${LOCALDIR} ${batch_size} ${max_epochs} ${domain} ${gpus} ${cpus} ${conda_env}"
