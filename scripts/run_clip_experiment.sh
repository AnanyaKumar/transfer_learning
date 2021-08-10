#!/usr/bin/env bash
#SBATCH --mail-user=rmjones@stanford.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=jag-standard
#SBATCH --exclude=jagupard[28-29]
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --requeue

MODELS="RN50 RN101 RN50x4 ViT-B/32 all"

print_usage () {
    usage_string="run_clip_experiment.sh model"
    usage_string+=" [-h|--help]"
    usage_string+=" [--C inv_reg_strength]"
    usage_string+="\n\nOptions:\n"
    usage_string+="-h, --help               Show this help message\n"
    usage_string+="--C                      Inverse regularization for linear probe\n"
    usage_string+="--num_selftrain_iters    Iterations of self-training\n"
    printf "$usage_string"
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit
fi

model=$1
valid_model="false"
for model_name in $MODELS; do
    if [[ "$model" == "$model_name" ]]; then
       valid_model="true"
       break
    fi
done
if [[ "$valid_model" != "true" ]]; then
    echo "Unsupported model name: $model"
    echo "Valid models: $MODELS"
    exit 1
fi
shift

C=0.316
num_selftrain_iters=0

while true; do
    if [[ -z "$1" ]]; then
	break
    fi
    case "$1" in
	-h|--help)
	    print_usage
	    exit
	    ;;
	--C)
	    if [[ -z "$2" ]]; then
		echo "--C must be non-empty!"
		exit 1
	    fi
	    C=$2
	    shift
	    ;;
	--num_selftrain_iters)
	    if [[ -z "$2" ]]; then
		echo "--num_selftrain_iters must be non-empty!"
		exit 1
	    fi
	    num_selftrain_iters=$2
	    shift
	    ;;
	*)
	    echo "Unsupported argument $1"
	    print_usage
	    exit 1
	    ;;
    esac
    shift
done

set -ex
source ../.env/bin/activate
python run_clip_experiment.py $model \
       --C $C --num_selftrain_iters $num_selftrain_iters
