#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -x

domains=${1:-'all'}
epochs=${2:-200}
batch_size=${3:-64}
queue_start=${4:-15}
arch=${5:-resnet50}
epsilon=${6:-'0.05'}
nmb_prototypes=${7:-3000}
conda_env=${8:-`whoami`-ue}
port=${9:-'40000'}

printf "Running DomainNet with domains $domains for $epochs epochs "
printf " with batch size $batch_size, introducing queue at epoch $queue_start "
printf " epsilon $epsilon, arch $arch and nmb_prototypes $nmb_prototypes\n"
echo "Using conda environment $conda_env"

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=":$port"

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

DATASET_PATH=${LOCAL_DOMAINNET_PATH}
echo "Using DomainNet data from $DATASET_PATH"
domain_list=`echo $domains | tr , -`
experiment_name="domainnet_${domain_list}_queue${queue_start}_epochs${epochs}"
experiment_name="${experiment_name}_batchsize${batch_size}"
experiment_name="${experiment_name}_epsilon${epsilon}_arch$arch"
experiment_name="${experiment_name}_prototypes${nmb_prototypes}"
echo "Experiment name: $experiment_name"
dump_path="/scr/scr-with-most-space/$(whoami)/swav_experiments/$experiment_name"
mkdir -p $dump_path
echo "Will dump checkpoints in $dump_path"
experiment_path="checkpoints/$experiment_name"
mkdir -p $experiment_path
echo "Final checkpoints and logs will be copied to $experiment_path"

# Use linear scaling for learning rate
DEFAULT_LR=4.8
DEFAULT_BATCH_SIZE=4096
effective_batch_size=$((batch_size * SLURM_NTASKS_PER_NODE))
if [ $effective_batch_size = 256 ]; then
    base_lr=0.6
else
    base_lr=$(python3 -c "print($DEFAULT_LR / ($DEFAULT_BATCH_SIZE / $effective_batch_size))")
fi

final_lr=$(python3 -c "print($base_lr / 1000)")

echo "Using base_lr=$base_lr and final_lr=$final_lr"

source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda activate $conda_env
srun --output=${dump_path}/%j.out --error=${dump_path}/%j.err --label python -u main_swav.py \
--data_path $DATASET_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon $epsilon \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes $nmb_prototypes \
--queue_length 3840 \
--epoch_queue_starts $queue_start \
--epochs $epochs \
--batch_size $batch_size \
--base_lr $base_lr \
--final_lr $final_lr \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 4 \
--dist_url $dist_url \
--arch $arch \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $dump_path \
--domains $domains \
--dataset_name domainnet

echo "Copying from $dump_path to $experiment_path"
cp -r $dump_path/* $experiment_path
