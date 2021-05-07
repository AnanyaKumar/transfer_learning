#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=jagupard[4-8,28-29]

set -x

checkpoint=$1
source_domain=$2
target_domain=$3
arch=${4:-resnet18}
conda_env=${5:-`whoami`-ue}
port=${6:-":40000"}

checkpoint_parent=$(dirname $checkpoint)

if [ $(basename $checkpoint_parent) != checkpoints ]; then
    echo "The checkpoint must come from the 'checkpoints' directory!"
    exit
fi

pretrain_experiment_path=$(dirname $checkpoint_parent)

echo "Running linear probing for experiment $(basename $pretrain_experiment_path)"
echo "Using conda environment $conda_env"

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=$port

# COPY to local
LOCAL_DOMAINNET_PATH=/scr/scr-with-most-space/domainnet
GLOBAL_DOMAINNET_PATH=/u/scr/nlp/domainnet/domainnet.zip
# COPY domainnet
if [ ! -d "$LOCAL_DOMAINNET_PATH" ]; then
  mkdir -p $LOCAL_DOMAINNET_PATH
  echo "Copying DomainNet files to $LOCAL_DOMAINNET_PATH"
  cp $GLOBAL_DOMAINNET_PATH $LOCAL_DOMAINNET_PATH
  unzip -q ${LOCAL_DOMAINNET_PATH}/domainnet.zip -d $LOCAL_DOMAINNET_PATH
fi

DATASET_PATH=$LOCAL_DOMAINNET_PATH
echo "Using DomainNet data from $DATASET_PATH"
pretrain_experiment_name=$(basename $pretrain_experiment_path)
dump_path=/scr/scr-with-most-space/$(whoami)/swav_experiments/$pretrain_experiment_name/eval_linear
mkdir -p $dump_path
echo "Saving linear probe checkpoints to $dump_path"
linear_experiment_path="$pretrain_experiment_path/eval_linear"
mkdir -p $linear_experiment_path
echo "Final checkpoints and logs will be copied to $linear_experiment_path"

if [ ! -f $checkpoint.oldformat ]; then
    source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
    conda activate $conda_env
    # convert checkpoint
    python convert_checkpoints.py --pretrained $checkpoint --arch resnet50
    conda deactivate
fi

PYTHON_CMD=.env/bin/python
srun --output=${dump_path}/%j.out --error=${dump_path}/%j.err --label $PYTHON_CMD -u eval_linear.py \
--data_path $DATASET_PATH \
--pretrained $checkpoint.oldformat \
--epochs 100 \
--batch_size 64 \
--arch $arch \
--dump_path $dump_path \
--dataset_name domainnet \
--domains "$source_domain,$target_domain" \
--dist_url $dist_url \
--dataset_kwargs version=sentry

echo "Copying from $dump_path to $linear_experiment_path"
cp -r $dump_path/* $linear_experiment_path
