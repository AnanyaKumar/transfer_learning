#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -x

breeds_name=$1
use_source=$2
use_target=$3
conda_env=${4:-`whoami`-ue}

echo "Running Breeds $1 exp with Source=$use_source and Target=$use_target"
echo "Using conda environment $conda_env"

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
    tar xzf $f;
  done
fi
DATASET_PATH=$LOCAL_IMAGENET_PATH

# DATASET_PATH=$GLOBAL_IMAGENET_PATH

echo "Using ImageNet data from $DATASET_PATH"
EXPERIMENT_NAME="breeds_${breeds_name}_source_${use_source}_target_${use_target}"
echo "Experiment name: $EXPERIMENT_NAME"
EXPERIMENT_PATH="checkpoints/$EXPERIMENT_NAME"
echo "Using checkpoints from $EXPERIMENT_PATH"
EXPERIMENT_PATH_LINEAR=$EXPERIMENT_PATH/eval_linear
mkdir -p EXPERIMENT_PATH_LINEAR
echo "Results saved in $EXPERIMENT_PATH_LINEAR"


source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate $conda_env
srun --output=${EXPERIMENT_PATH_LINEAR}/%j.out --error=${EXPERIMENT_PATH_LINEAR}/%j.err --label python -u eval_linear.py \
--data_path $DATASET_PATH \
--epochs 100 \
--batch_size 32 \
--arch resnet50 \
--dump_path $EXPERIMENT_PATH_LINEAR \
--dataset_name breeds \
--dataset_kwargs breeds_name=$breeds_name
