#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --exclude=jagupard[4-8],jagupard[10-15]

set -x

breeds_name=$1
use_source=$2
use_target=$3
epochs=$4
nmb_prototypes=$5
epoch_queue_starts=$6
conda_env=${7:-`whoami`-ue}
port=${8:-":13321"}

echo "Running Breeds $1 exp with Source=$use_source and Target=$use_target"
echo "For $4 epochs, using $5 prototypes, starting queue at $epoch_queue_starts"
echo "Using conda environment $conda_env"

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
# dist_url+=:40000
dist_url+=$port

# COPY to local
LOCAL_IMAGENET_PATH=/scr/scr-with-most-space/imagenet
GLOBAL_IMAGENET_PATH=/u/scr/nlp/eix/imagenet
# COPY imagenet
if [ ! -d "$LOCAL_IMAGENET_PATH" ]; then
  mkdir -p $LOCAL_IMAGENET_PATH
  echo "Copying ImageNet files to $LOCAL_DATASET_PATH"
  cp $GLOBAL_IMAGENET_PATH/*.tar.gz $LOCAL_IMAGENET_PATH
  for f in $LOCAL_IMAGENET_PATH/*.tar.gz;
  do
    tar xzf $f -C $LOCAL_IMAGENET_PATH;
  done
fi

DATASET_PATH=$LOCAL_IMAGENET_PATH
echo "Using ImageNet data from $DATASET_PATH"
EXPERIMENT_NAME="breeds_${breeds_name}_source_${use_source}_target_${use_target}_${epochs}epochs_${nmb_prototypes}protos_${epoch_queue_starts}qstart"
echo "Experiment name: $EXPERIMENT_NAME"
EXPERIMENT_PATH="checkpoints/$EXPERIMENT_NAME"
mkdir -p $EXPERIMENT_PATH
echo "Saving checkpoints to $EXPERIMENT_PATH"

source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate $conda_env
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
--nmb_prototypes $nmb_prototypes \
--queue_length 3840 \
--epoch_queue_starts $epoch_queue_starts \
--epochs $epochs \
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
--dataset_name breeds \
--epsilon 0.03 \
--dataset_kwargs breeds_name=$breeds_name source=$use_source target=$use_target

# epsilon added for living17 target only
