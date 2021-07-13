#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

show_help() {
    usage_string="Usage: domainnet.sh [--epochs EPOCHS]"
    usage_string+="[-s|--source SOURCE_DOMAIN]"
    usage_string+="[-t|--target TARGET_DOMAIN]"
    usage_string+="[--source_amount SOURCE_AMOUNT]"
    usage_string+="[--target_amount TARGET_AMOUNT]"
    usage_string+="[--related_amount RELATED_AMOUNT]"
    usage_string+="[-b|--batch_size BATCH_SIZE]"
    usage_string+="[-q|--queue_start QUEUE_START]"
    usage_string+="[--queue_length QUEUE_LENGTH]"
    usage_string+="[-a|--arch ARCHITECTURE]"
    usage_string+="[--epsilon EPSILON]"
    usage_string+="[--nmb_prototypes NUM_PROTOTYPES]"
    usage_string+="[--conda_env CONDA_ENV]"
    usage_string+="[-p|--port PORT]"

    usage_string+="\n\n"
    usage_string+="-s|--source Source domain"
    usage_string+="-t|--target Target domain"
    usage_string+="\t--source_amount Amount of source data to use (as a fraction of min(source, target) size) (default: 0).\n"
    usage_string+="\t--target_amount Amount of target data to use (as a fraction of min(source, target) size) (default: 0).\n"
    usage_string+="\t--related_amount Amount of related data to use (as a fraction of min(source, target) size) (default: 0).\n"
    usage_string+="\t-b|--batch_size Batch Size (default: 64)\n"
    usage_string+="\t-q|--queue_start Epoch to introduce queue (default: 15)\n"
    usage_string+="\t--queue_length Length of queue (default: 3840)\n"
    usage_string+="\t-a|--arch ResNet architecture (default: resnet50)\n"
    usage_string+="\t--epsilon Epsilon (default: 0.05)\n"
    usage_string+="\t--nmb_prototypes Number of prototypes (default: 3000)\n"
    usage_string+="\t--conda_env Conda environment (default: \$(whoami)-ue)\n"
    usage_string+="\t-p|--port TCP port for distributed training (default: 40000)\n"
    printf "$usage_string"
}

source=""
target=""
source_amount=0
target_amount=0
related_amount=0
epochs=200
batch_size=64
queue_start=15
queue_length=3840
arch=resnet50
epsilon=0.05
nmb_prototypes=3000
conda_env=$(whoami)-ue
port=40000

while true; do
    case $1 in
    -h|--help) # Print help
        show_help
        exit
        ;;
    -s|--source) # Source domain
        if [ "$2" ]; then
        source=$2
        shift
        else
        echo '--source must be non-empty!'; exit 1
        fi
        ;;
    -t|--target) # Target domain
        if [ "$2" ]; then
        target=$2
        shift
        else
        echo '--target must be non-empty!'; exit 1
        fi
        ;;
    --source_amount) # How much source data to use for pre-training
        if [ "$2" ]; then
        source_amount=$2
        shift
        else
        echo '--source_amount must be non-empty!'; exit 1
        fi
        ;;
    --target_amount) # How much target data to use for pre-training
        if [ "$2" ]; then
        target_amount=$2
        shift
        else
        echo '--target_amount must be non-empty!'; exit 1
        fi
        ;;
    --related_amount) # How much related data to use for pre-training
        if [ "$2" ]; then
        related_amount=$2
        shift
        else
        echo '--related_amount must be non-empty!'; exit 1
        fi
        ;;
    --epochs)
        if [ "$2" ]; then
        epochs=$2
        shift
        else
        echo '--epochs must be non-empty!'; exit 1
        fi
        ;;
    -b|--batch_size) # Batch size for a single GPU
        if [ "$2" ]; then
        batch_size=$2
        shift
        else
        echo '-b|--batch_size must be non-empty!'; exit 1
        fi
        ;;
    -q|--queue_start) # Epoch to introduce queue
        if [ "$2" ]; then
        queue_start=$2
        shift
        else
        echo '-q|--queue_start must be non-empty!'; exit 1
        fi
        ;;
    --queue_length) # Length of queue
        if [ "$2" ]; then
        queue_length=$2
        shift
        else
        echo '--queue_length must be non-empty!'; exit 1
        fi
        ;;
    -a|--arch) # ResNet architecture
        if [ "$2" ]; then
        arch=$2
        shift
        else
        echo '-a|--arch must be non-empty!'; exit 1
        fi
        ;;
    --epsilon) # Epsilon for entropy minimization
        if [ "$2" ]; then
        epsilon=$2
        shift
        else
        echo '--epsilon must be non-empty!'; exit 1
        fi
        ;;
    --nmb_prototypes) # Number of prototypes
        if [ "$2" ]; then
        nmb_prototypes=$2
        shift
        else
        echo '--nmb_prototypes must be non-empty!'; exit 1
        fi
        ;;
    --conda_env) # Conda environment
        if [ "$2" ]; then
        conda_env=$2
        shift
        else
        echo '--conda_env must be non-empty!'; exit 1
        fi
        ;;
    -p|--port) # TCP Port for distributed training
        if [ "$2" ]; then
        port=$2
        shift
        else
        echo '-p|--port must be non-empty!'; exit 1
        fi
        ;;
    *)
        if [ -z "$1" ]; then
           break
        else
        echo "Unsupported argument: $1"; exit 1
        fi
        ;;
    esac
    shift
done

set -x

if [[ $source_amount = 0 && $target_amount = 0 && $related_amount = 0 ]]; then
    echo 'At least one of --source_amount, --target_amount, and --related_amount must be positive!'
    exit 1
fi
if [[ $source_amount != 0 && $related_amount != 0 ]]; then
    echo 'Cannot set both --source_amount and --related_amount.'
    exit 1
fi

printf "Running DomainNet (SENTRY) with source $source and target $target for $epochs epochs "
printf " with source amount $source_amount, target amount $target_amount, and related amount $related_amount"
printf " with batch size $batch_size, introducing queue at epoch $queue_start "
printf " epsilon $epsilon, arch $arch and nmb_prototypes $nmb_prototypes\n"
echo "Using conda environment $conda_env"

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=":$port"

# COPY to local
LOCAL_DOMAINNET_PATH=/scr/biggest/domainnet
GLOBAL_DOMAINNET_PATH=/u/scr/nlp/domainnet/domainnet.zip
../../scripts/copy_dataset.sh domainnet

DATASET_PATH=${LOCAL_DOMAINNET_PATH}
echo "Using DomainNet data from $DATASET_PATH"
experiment_name="domainnet_source${source}_target${target}"
experiment_name+="_sourceamount${source_amount}_targetamount${target_amount}_relatedamount${related_amount}"
experiment_name+="_queue${queue_start}_epochs${epochs}"
experiment_name+="_batchsize${batch_size}"
experiment_name+="_epsilon${epsilon}_arch$arch"
experiment_name+="_prototypes${nmb_prototypes}"
echo "Experiment name: $experiment_name"
dump_path="/scr/biggest/$(whoami)/swav_experiments/$experiment_name"
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
--queue_length $queue_length \
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
--dataset_name domainnet \
--dataset_kwargs source_domain=$source target_domain=$target \
    source_amount=$source_amount target_amount=$target_amount related_amount=$related_amount

echo "Copying from $dump_path to $experiment_path"
cp -r $dump_path/* $experiment_path
