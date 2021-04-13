#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=swav_200ep_bs256_pretrain
#SBATCH --mem=96G

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

# COPY to local
LOCALDIR=/scr/scr-with-most-space/eix
mkdir -p $LOCALDIR
DOMAINNET_ZIP=/u/scr/nlp/domainnet/domainnet.zip
# COPY domainnet
if [ ! -d "${LOCALDIR}/domainnet" ]; then
  echo "copying file"
  cp "${DOMAINNET_ZIP}" "${LOCALDIR}"
  mkdir -p ${LOCALDIR}/domainnet
  unzip ${LOCALDIR}/domainnet.zip -d ${LOCALDIR}/domainnet
fi
DATASET_PATH=${LOCALDIR}/domainnet
# DATASET_PATH=/u/scr/nlp/domainnet

EXPERIMENT_PATH="/u/scr/eix/unlabeled_extrapolation/checkpoints/domainnet_swav_200ep_bs256_pretrain"
mkdir -p $EXPERIMENT_PATH

source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate eix-ue
srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u main_swav.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 3840 \
--epoch_queue_starts 15 \
--epochs 200 \
--batch_size 64 \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--dist_url $dist_url \
--arch resnet50 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH \
--domains all \
--dataset_name domainnet
