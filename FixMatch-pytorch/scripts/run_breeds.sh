#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -x

breeds_name=$1
conda_env=${2:-`whoami`-ue}

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

# PYTHON_CMD=../swav/.env/bin/python
source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate $conda_env
PYTHON_CMD=python
$PYTHON_CMD -m torch.distributed.launch --nproc_per_node 4 ./train.py \
    --dataset breeds \
    --arch resnet50 \
    --batch-size 8 \
    --lr 0.03 \
    --wdecay 0.001 \
    --expand-labels \
    --total-steps 32768 \
    --seed 5 \
    --amp \
    --opt_level O1 \
    --out results/$breeds_name \
    --dataset_kwargs root=${LOCAL_IMAGENET_PATH} breeds_name=$breeds_name

    # --lr 0.01 \
    # --mu 3 \
    # --lambda-u 10 \
    # --threshold 0.95 \

