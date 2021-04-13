#!/bin/bash

SCR=/u/scr/eix/unlabeled_extrapolation
CHKPT_SCR=/u/scr/eix/unlabeled_extrapolation/checkpoints
LOCALDIR=/scr/scr-with-most-space/eix
mkdir -p $SCR/logs
mkdir -p $CHKPT_SCR
partition=jag-hi
exclude=jagupard[4-8]
gpus=4
mem=96G
cpus=16

batch_size=64
max_epochs=200
domain=all
conda_env=eix-ue

sbatch --exclude $exclude --partition $partition --gres=gpu:${gpus} --mem $mem -c ${cpus} --output $SCR/logs/swav_domainnet_all /u/scr/eix/run_sbatch.sh \
"source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh && conda activate eix-ue && bash ${SCR}/scripts/pretrain_swav_domainnet.sh ${SCR} ${CHKPT_SCR} ${LOCALDIR} ${batch_size} ${max_epochs} ${domain} ${gpus} ${cpus}"

# bash ${SCR}/scripts/pretrain_swav_domainnet_local.sh ${SCR} ${CHKPT_SCR} ${LOCALDIR} ${batch_size} ${max_epochs} ${domain} 1 4

