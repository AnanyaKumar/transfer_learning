#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --exclude=jagupard[4-8],jagupard[10-15]

show_help() {
    usage_string="Usage: breeds.sh breeds_name"
    usage_string+=" [-s|--use_source] [-t|-- use_target]"
    usage_string+=" [--epochs EPOCHS] [--nmb_prototypes NUM_PROTOTYPES]"
    usage_string+=" [-q|--queue_start QUEUE_START] [-b|--batch_size BATCH_SIZE]"
    usage_string+=" [--epsilon EPSILON] [-a|--arch ARCHITECTURE]"
    usage_string+=" [--queue_length QUEUE_LENGTH]"
    usage_string+=" [--conda_env CONDA_ENV] [-p|--port PORT]"

    usage_string+="\n\n"
    usage_string+="\t-s|--use_source Use source domain data for pretraining.\n"
    usage_string+="\t-t|--use_target Use target domain data for pretraining.\n"
    usage_string+="\t--standardize_ds_size Standardize dataset size.\n"
    usage_string+="\t--epochs Number of epochs to pretrain (default: 400)\n"
    usage_string+="\t--nmb_prototypes Number of prototypes (default: 3000)\n"
    usage_string+="\t-q|--queue_start Epoch to introduce queue (default: 15)\n"
    usage_string+="\t-b|--batch_size Batch Size (default: 64)\n"
    usage_string+="\t-a|--arch ResNet architecture (default: resnet50)\n"
    usage_string+="\t--queue_length Length of queue (default: 3840)\n"
    usage_string+="\t--epsilon Epsilon (default: 0.05)\n"
    usage_string+="\t--conda_env Conda environment (default: \$(whoami)-ue)\n"
    usage_string+="\t-p|--port TCP port for distributed training (default: 40000)\n"
    printf "$usage_string"
}



if [[ $# -lt 1 ]]; then
    show_help
    exit 1
fi

if [[ $1 = -h || $1 = --help ]]; then
    show_help
    exit
fi

breeds_name=$1
if [[ $breeds_name != living17 && $breeds_name != entity30 ]]; then
    echo "Unsupported BREEDS name: $breeds_name"
    exit 1
fi
shift
use_source=False
use_target=False
standardize_ds_size=False
epochs=400
nmb_prototypes=3000
epoch_queue_starts=15
batch_size=64
arch=resnet50
epsilon=0.05
queue_length=3840
conda_env=$(whoami)-ue
port=40000

while true; do
    case $1 in
	-h|--help) # Print help
	    show_help
	    exit
	    ;;
	-s|--use_source) # Use source domain for pretraining
	    use_source=True
	    ;;
	-t|--use_target) # Use target domain for pretraining
	    use_target=True
	    ;;
	--standardize_ds_size)
		standardize_ds_size=True
		;;
	--epochs)
	    if [ "$2" ]; then
		epochs=$2
		shift
	    else
		echo '--epochs must be non-empty!'; exit 1
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
	-q|--queue_start) # Epoch to introduce queue
	    if [ "$2" ]; then
		epoch_queue_starts=$2
		shift
	    else
		echo '-q|--queue_start must be non-empty!'; exit 1
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
	--queue_length) # Length of queue
	    if [ "$2" ]; then
		queue_length=$2
		shift
	    else
		echo '--queue_length must be non-empty!'; exit 1
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

if [[ $use_source = False && $use_target = False ]]; then
    echo 'At least one of -s|--use_source or -t|--use_target must be given!'
    exit 1
fi

# Use linear scaling for learning rate
DEFAULT_LR=4.8
DEFAULT_BATCH_SIZE=4096
effective_batch_size=$((batch_size * SLURM_NTASKS_PER_NODE))
if [ $effective_batch_size = 256 ]; then
    base_lr=0.6
else
    base_lr=$(python3 -c "print($DEFAULT_LR / ($DEFAULT_BATCH_SIZE / $effective_batch_size))")
    if [[ $? -ne 0 ]]; then
	echo 'Error computing batch size, exiting...'
	exit $?
    fi
fi

final_lr=$(python3 -c "print($base_lr / 1000)")

echo "Using base_lr=$base_lr and final_lr=$final_lr"

exp_info="Running Breeds $breeds_name exp"
exp_info+=" with Source=$use_source and Target=$use_target for $epochs epochs"
exp_info+=", using $nmb_prototypes prototypes, architecture $arch"
exp_info+=", starting queue at $epoch_queue_starts, batch size $batch_size"
exp_info+=", epsilon $epsilon, queue length $queue_length"
exp_info+=", base_lr $base_lr, final_lr $final_lr"
exp_info+=". Using conda environment $conda_env"
echo $exp_info

master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:9:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:$port

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
experiment_name="breeds_${breeds_name}_source${use_source}_target${use_target}"
experiment_name+="_${epochs}epochs_${nmb_prototypes}protos"
experiment_name+="_${epoch_queue_starts}qstart_${queue_length}qlength"
experiment_name+="_batchsize${batch_size}"
experiment_name+="_epsilon${epsilon}_arch$arch"
experiment_name+="_standardsize${standardize_ds_size}"
echo "Experiment name: $experiment_name"
dump_path="/scr/scr-with-most-space/$(whoami)/swav_experiments/$experiment_name"
mkdir -p $dump_path
echo "Will dump checkpoints in $dump_path"
experiment_path="checkpoints/$experiment_name"

echo "Final checkpoints and logs will be copied to $experiment_path"

source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda deactivate
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
--epoch_queue_starts $epoch_queue_starts \
--epochs $epochs \
--batch_size $batch_size \
--base_lr $base_lr \
--final_lr $final_lr \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--dist_url $dist_url \
--arch $arch \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $dump_path \
--dataset_name breeds \
--standardize_ds_size $standardize_ds_size \
--epsilon $epsilon \
--dataset_kwargs breeds_name=$breeds_name source=$use_source target=$use_target

echo "Copying from $dump_path to $experiment_path"
mkdir -p $experiment_path
cp -r $dump_path/* $experiment_path
