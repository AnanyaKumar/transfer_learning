#!/usr/bin/env bash

print_usage () {
    USAGE_STRING="launch_clip_experiments_sbatch.sh [-t|--translate_features]"
    USAGE_STRING="$USAGE_STRING [--no_dependency]"
    USAGE_STRING="$USAGE_STRING experiment_type model"
    USAGE_STRING="$USAGE_STRING [experiment_args]"
    echo $USAGE_STRING
}

DOMAINS=(real clipart painting sketch)
NUM_DOMAINS=${#DOMAINS[@]}
SBATCH_SCRIPT="run_clip_experiment_array.sh"


while true; do
    case "$1" in
	-t|--translate_features)
	    translate_features=true
	    ;;
	--no_dependency)
	    no_dependency=true
	    ;;
	-*)
	    echo "Unsupported argument $1"
	    exit 1
	    ;;
	*)
	    break
	    ;;
    esac
    shift
done

if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

experiment_type=$1
model=$2
experiment_args=$3

echo $experiment_type
echo $model

python_args="$experiment_type $model"
if [ $translate_features ]; then
    python_args+=" --translate_features"
fi
if [ "$experiment_args" ]; then
    python_args+=" $experiment_args"
fi

if [ $translate_features ]; then
    num_tasks=$((NUM_DOMAINS ** 2))
else
    num_tasks=$NUM_DOMAINS
fi

sbatch_cmd="sbatch --array=0-$((num_tasks - 1))"
sbatch_cmd="$sbatch_cmd --parsable $SBATCH_SCRIPT '$python_args'"
job_number=$(eval $sbatch_cmd)

email_cmd="source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh"
email_cmd="$email_cmd; conda deactivate; conda activate $(whoami)-ue"

aggregate_cmd="python aggregate_clip_results.py $experiment_type $model"
if [ $translate_features ]; then
    aggregate_cmd+=" ${DOMAINS[@]} --translate_features"
else
    aggregate_cmd+=" all"
fi

if [ "$experiment_args" ]; then
    aggregate_cmd+=" --experiment_kwargs ${experiment_args//-}"
fi

email_cmd+="; $aggregate_cmd"
./send_email_sbatch.sh --slurm_dependency $job_number \
		       "CLIP Results: $python_args" command "$email_cmd"
