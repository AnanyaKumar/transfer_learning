#!/usr/bin/env bash

print_usage () {
    USAGE_STRING="launch_clip_experiments_sbatch.sh [-t|--translate_features]"
    USAGE_STRING="$USAGE_STRING [--slurm_dependency JOBID]"
    USAGE_STRING="$USAGE_STRING experiment_type model"
    USAGE_STRING="$USAGE_STRING [experiment_args]"
    echo $USAGE_STRING
}

launch_job () {
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
    if [ "$slurm_dependency" ]; then
	sbatch_cmd+=" --dependency=afterany:$slurm_dependency"
    fi
    if [ "$big_jags" ]; then
	sbatch_cmd+=" --exclude jagupard[10-25]"
    else
	sbatch_cmd+=" --exclude jagupard[26-29]"
    fi
    sbatch_cmd="$sbatch_cmd --parsable $SBATCH_SCRIPT '$python_args'"
    echo $sbatch_cmd
    job_number=$(eval $sbatch_cmd)
    slurm_dependency=$job_number
    echo "Submitted batch job $job_number"
    email_cmd="source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh"
    email_cmd="$email_cmd; conda deactivate; conda activate $(whoami)-ue"

    aggregate_cmd="python aggregate_clip_results.py $experiment_type $model"
    if [ $translate_features ]; then
	aggregate_cmd+=" ${DOMAINS[@]} --translate_features"
    else
	aggregate_cmd+=" all"
    fi

    if [ "$experiment_args" ]; then
	experiment_kwargs=${experiment_args//-}
	aggregate_cmd+=" --experiment_kwargs ${experiment_kwargs//-}"
    fi

    email_cmd+="; $aggregate_cmd"
    ./send_email_sbatch.sh --slurm_dependency $job_number \
			   "CLIP Results: $python_args" command "$email_cmd"
}

DOMAINS=(real clipart painting sketch)
MODELS=(RN50 RN101 RN50x4 RN50x16 ViT-B/32 ViT-B/16)
NUM_DOMAINS=${#DOMAINS[@]}
SBATCH_SCRIPT="run_clip_experiment_array.sh"


while true; do
    case "$1" in
	-t|--translate_features)
	    translate_features=true
	    ;;
	--slurm_dependency)
	    if [ "$2" ]; then
		slurm_dependency="$2"
		shift
	    else
		echo '--slurm_dependency must be non-empty!'
		exit 1
	    fi
	    ;;
	--big_jags)
	    big_jags=true
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

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
    print_usage
    exit 1
fi

experiment_type=$1
model=$2
experiment_args=$3

if [ "$model" = all ]; then
    for model in ${MODELS[@]}; do
	launch_job
    done
else
    launch_job
fi

