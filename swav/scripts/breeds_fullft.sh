#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -x

breeds_name=$1
use_source=$2
use_target=$3
conda_env=${4:-`whoami`-ue}
# port=${5:-":13321"}

echo "Running Breeds $1 exp with Source=$use_source and Target=$use_target"
echo "Using conda environment $conda_env"

# master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
# dist_url="tcp://"
# dist_url+=$master_node
# # dist_url+=:40000
# dist_url+=$port

# COPY to local
LOCAL_IMAGENET_PATH=/scr/scr-with-most-space/imagenet
GLOBAL_IMAGENET_PATH=/u/scr/nlp/eix/imagenet
# COPY imagenet
if [ ! -d "$LOCAL_IMAGENET_PATH" ]; then
  mkdir -p $LOCAL_IMAGENET_PATH
  echo "Copying ImageNet files to $LOCAL_DATASET_PATH"
  cp $GLOBAL_IMAGENET_PATH/*.tar.gz $LOCAL_IMAGENET_PATH
fi
if [ ! -d "${LOCAL_IMAGENET_PATH}/train" ]; then
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
EXPERIMENT_PATH_FULL="$EXPERIMENT_PATH/eval_fullft"
mkdir -p $EXPERIMENT_PATH_FULL
echo "Results saved in $EXPERIMENT_PATH_FULL"


source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate $conda_env
PYTHON_CMD=python
srun --output=${EXPERIMENT_PATH_FULL}/%j.out --error=${EXPERIMENT_PATH_FULL}/%j.err --label $PYTHON_CMD -u eval_semisup.py \
--data_path $DATASET_PATH \
--pretrained $EXPERIMENT_PATH/checkpoints/ckp-199.pth \
--epochs 100 \
--batch_size 32 \
--arch resnet50 \
--dump_path $EXPERIMENT_PATH_FULL \
--dataset_name breeds \
--workers 1 \
--dataset_kwargs breeds_name=$breeds_name

# for interactive
# $PYTHON_CMD -m torch.distributed.launch --nproc_per_node=1 eval_semisup.py \
# --is_not_slurm_job True \

