#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

show_help() {
    usage_string="Usage: domainnet.sh [-d|--domains DOMAINS] [--epochs EPOCHS]"
    usage_string="$usage_string [-b|--batch_size BATCH_SIZE]"
    usage_string="$usage_string [-q|--queue_start QUEUE_START]"
    usage_string="$usage_string [-a|--arch ARCHITECTURE]"
    usage_string="$usage_string [--epsilon EPSILON]"
    usage_string="$usage_string [--nmb_prototypes NUM_PROTOTYPES]"
    usage_string="$usage_string [--conda_env CONDA_ENV]"
    usage_string="$usage_string [-p|--port PORT]"
    usage_string+="[--use_sentry USE_SENTRY]"

    usage_string="$usage_string\n\n"
    usage_string="$usage_string\t-d|--domains Domains (default:all)\n"
    usage_string="$usage_string\t-b|--batch_size Batch Size (default: 64)\n"
    usage_string="$usage_string\t-q|--queue_start Epoch to introduce queue (default: 15)\n"
    usage_string="$usage_string\t-a|--arch ResNet architecture (default: resnet50)\n"
    usage_string="$usage_string\t--epsilon Epsilon (default: 0.05)\n"
    usage_string="$usage_string\t--nmb_prototypes Number of prototypes (default: 3000)\n"
    usage_string="$usage_string\t--conda_env Conda environment (default: \$(whoami)-ue)\n"
    usage_string="$usage_string\t-p|--port TCP port for distributed training (default: 40000)\n"
    usage_string+="\t--use_sentry Whether or not to use SENTRY splits (default: false).\n"
    printf "$usage_string"
}

domains=all
use_sentry=false
epochs=200
batch_size=64
queue_start=15
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
	-d|--domains) # Comma-separated list of domains
	    if [ "$2" ]; then
		domains=$2
		shift
	    else
		echo '-d|--domains must be non-empty!'; exit 1
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
	--use_sentry)
        if [ "$2" ]; then
        use_sentry=$2
        shift
        else
        echo '--use_sentry must be true or false!'; exit 1
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

printf "Running DomainNet (SENTRY splits: $use_sentry) with domains $domains for $epochs epochs "
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
echo "Using DomainNet data from $DATASET_PATH, using SENTRY splits $use_sentry"
domain_list=`echo $domains | tr , -`
experiment_name="domainnet_sentry${use_sentry}_${domain_list}_queue${queue_start}_epochs${epochs}"
experiment_name="${experiment_name}_batchsize${batch_size}"
experiment_name="${experiment_name}_epsilon${epsilon}_arch$arch"
experiment_name="${experiment_name}_prototypes${nmb_prototypes}"
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
--dataset_name domainnet \
--dataset_kwargs domains=$domains use_sentry=$use_sentry

echo "Copying from $dump_path to $experiment_path"
cp -r $dump_path/* $experiment_path
