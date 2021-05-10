#!/bin/bash
#SBATCH --partition jag-standard
#SBATCH --exclude jagupard[4-8,28-29]
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

show_help() {
    usage_string="Usage: breeds_linear_probe.sh"
    usage_string+=" breeds_name checkpoint"
    usage_string+=" [-e|--epochs EPOCHS]"
    usage_string+=" [-b|--batch_size BATCH_SIZE] [-a|--arch ARCHITECTURE]"
    usage_string+=" [--lr|--learning_rate LEARNING_RATE]"
    usage_string+=" [--overwrite] [--conda_env CONDA_ENV] [-p|--port PORT]"

    usage_string+="\n\nOptions:\n"
    usage_string+="\t-e|--epochs Number of epochs to train (default: 100)\n"
    usage_string+="\t-b|--batch_size Batch size (default: 64)\n"
    usage_string+="\t-a|--arch ResNet architecture (default: resnet18)\n"
    usage_string+="\t--lr|--learning_rate Learning rate (default: 0.3 for batch size 64, linearly scaled)\n"
    usage_string+="\t--overwrite Overwrite existing experiment\n"
    usage_string+="\t--conda_env Conda environment (default: \$(whoami)-ue)\n"
    usage_string+="\t-p|--port TCP port for distributed training (default: 40000)\n"
    printf "$usage_string"
}

if [[ "$1" = "-h" || "$1" = "--help" ]]; then
    show_help
    exit
fi

if [ $# -lt 2 ]; then
    show_help
    exit 1
fi

breeds_name=$1
if [[ $breeds_name != living17 && $breeds_name != entity30 ]]; then
    echo "Unsupported BREEDS name: $breeds_name"
    exit 1
fi
checkpoint=$2
checkpoint_base=$(basename $checkpoint)
checkpoint_extension=${checkpoint_base#*.}
if [[ "$checkpoint_extension" != pth.tar && "$checkpoint_extension" != pth ]]; then
    echo "Invalid checkpoint: $checkpoint"
    exit 1
fi

shift 2

epochs=100
batch_size=64
arch=resnet50
overwrite=False
conda_env=$(whoami)-ue
port=40000

while true; do
    case $1 in
	-h|--help) # Print help
	    show_help
	    exit
	    ;;
	-e|--epochs) # Epochs
	    if [ "$2" ]; then
		epochs=$2
		shift
	    else
		echo '-e|--epochs must be non-empty!'; exit 1
	    fi
	    ;;
	-b|--batch_size) # Batch size
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
	--lr|--learning_rate) # Learning rate
	    if [ "$2" ]; then
		lr=$2
		shift
	    else
		echo '--lr|--learning_rate must be non-empty!'; exit 1
	    fi
	    ;;
	--overwrite) # Overwrite existing run
	    overwrite=True
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

checkpoint_realpath=$(realpath $checkpoint)
checkpoint_parent=$(dirname $checkpoint_realpath)
if [ "$(basename $checkpoint_parent)" = checkpoints ]; then
    pretrain_experiment_path=$(dirname $checkpoint_parent)
elif [ "$(basename $(dirname $(dirname $checkpoint_parent)))" = swav ]; then
    pretrain_experiment_path=$checkpoint_parent
else
    echo "Invalid checkpoint path: $checkpoint_realpath!"
    exit 1
fi

if [ -z "$lr" ]; then
    # Use linear scaling for learning rate
    DEFAULT_LR=0.3
    DEFAULT_BATCH_SIZE=64
    effective_batch_size=$((batch_size * SLURM_NTASKS_PER_NODE))
    lr=$(python3 -c "print($DEFAULT_LR / ($DEFAULT_BATCH_SIZE / $effective_batch_size))")
fi

experiment_name="linearprobe_epochs${epochs}_lr$lr"
experiment_name+="_batchsize${batch_size}"

experiment_path_linear="$pretrain_experiment_path/$experiment_name"
if [[ -d "$experiment_path_linear" && "$overwrite" != True ]]; then
    echo "$experiment_path_linear already exists! Pass --overwrite to overwrite."
    exit 1
fi
mkdir -p $experiment_path_linear
echo "Final checkpoints and logs will be copied to $experiment_path"

dump_relative=$(basename $pretrain_experiment_path)/$experiment_name
dump_path="/scr/scr-with-most-space/$(whoami)/swav_experiments/$dump_relative"
mkdir -p $dump_path
echo "Will dump checkpoints in $dump_path"

echo "Running linear probing for experiment $(basename $pretrain_experiment_path)"
echo "Using conda environment $conda_env"

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
fi
if [ ! -d "${LOCAL_IMAGENET_PATH}/train" ]; then
  for f in $LOCAL_IMAGENET_PATH/*.tar.gz;
  do
    tar xzf $f;
  done
fi
DATASET_PATH=$LOCAL_IMAGENET_PATH
echo "Using ImageNet data from $DATASET_PATH"


if [ ! -f "$checkpoint.oldformat" ]; then
    source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
    conda activate $conda_env
    # convert checkpoint
    python convert_checkpoints.py --pretrained $checkpoint --arch $arch
    conda deactivate
fi

PYTHON_CMD=.env/bin/python
srun --output=${dump_path}/%j.out --error=${dump_path}/%j.err --label $PYTHON_CMD -u eval_linear.py \
--data_path $DATASET_PATH \
--pretrained $checkpoint.oldformat \
--epochs $epochs \
--lr $lr \
--batch_size $batch_size \
--arch $arch \
--dump_path $dump_path \
--dataset_name breeds \
--dist_url $dist_url \
--dataset_kwargs breeds_name=$breeds_name \
--workers $SLURM_NTASKS_PER_NODE

echo "Copying from $dump_path to $experiment_path_linear"
cp -r $dump_path/* $experiment_path_linear
