#!/usr/bin/env bash

print_usage () {
    USAGE_STRING="Usage: launch_clip_experiments_sbatch.sh"
    USAGE_STRING+=" [--test_all] [--slurm_partition PARTITION] [--big_jags]"
    USAGE_STRING+=" [--slurm_dependency JOBID] [--email_per_job]"
    USAGE_STRING+=" experiment_type model [experiment_args...]"
    echo $USAGE_STRING
}

launch_job () {
    if [ $test_all ]; then
	num_tasks=$NUM_DOMAINS
    else
	num_tasks=$((NUM_DOMAINS ** 2))
    fi

    sbatch_cmd="sbatch --array=0-$((num_tasks - 1))"
    if [ "$slurm_dependency" ]; then
	sbatch_cmd+=" --dependency=afterany:$slurm_dependency"
    fi
    sbatch_cmd+=" --partition $slurm_partition"
    if [ "$slurm_partition" = jag* ]; then
        if [ "$big_jags" ]; then
	    sbatch_cmd+=" --exclude jagupard[10-25]"
	else
	    sbatch_cmd+=" --exclude jagupard[26-29]"
	fi
    elif [ "$slurm_partition" = john ]; then
	sbatch_cmd+=" --gres=gpu:0"
    fi
    python_args="$experiment_type $model $experiment_args"
    sbatch_cmd="$sbatch_cmd --parsable $SBATCH_SCRIPT $python_args"
    echo $sbatch_cmd
    job_number=$(eval "$sbatch_cmd")
    echo "Submitted batch job $job_number"
    if [ $email_per_job ]; then
	launch_email $experiment_type $model "$experiment_args" $job_number
    fi
}

launch_email () {
    if [ $# -ne 4 ]; then
	launch_email_usage="Usage: launch_email experiment_type model"
	launch_email_usage+=" experiment_args slurm_dependency"
	echo $launch_email_usage
	exit 1
    fi

    experiment_type=$1
    model=$2
    experiment_args=$3
    slurm_dependency_id=$4
    email_cmd="source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh"
    email_cmd="$email_cmd; conda deactivate; conda activate $(whoami)-ue"

    aggregate_cmd="python run_clip_experiment.py results"
    aggregate_cmd+=" $experiment_type $model $experiment_args"
    if [ "$model" = all ]; then
	aggregate_cmd+=" --sep=,"
	subject="CLIP Results Summary: $experiment_type $experiment_args"
    else
	subject="CLIP Results: $experiment_type $model $experiment_args"
    fi
    email_cmd+="; $aggregate_cmd"
    ./send_email_sbatch.sh --slurm_dependency $slurm_dependency_id \
			   "$subject" command "$email_cmd"
}

set -e
DOMAINS=(real clipart painting sketch)
MODELS=(RN50 RN101 RN50x4 RN50x16 ViT-B/32 ViT-B/16)
NUM_DOMAINS=${#DOMAINS[@]}
SBATCH_SCRIPT="run_clip_experiment_array.sh"

slurm_partition='jag-standard'
while true; do
    case "$1" in
	-h|--help)
	    print_usage
	    exit
	    ;;
	--test_all)
	    test_all=true
	    ;;
	--slurm_partition)
	    if [ "$2" ]; then
		slurm_partition="$2"
		shift
	    else
		echo '--slurm_partition must be non-empty!'
		exit 1
	    fi
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
	--email_per_job)
	    email_per_job=true
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
shift 2
experiment_args="$@"

if [ ! $test_all ]; then
    experiment_args+=" --skip_source_eval"
fi

slurm_dependency_list=""
if [ "$model" = all ]; then
    for model in ${MODELS[@]}; do
	launch_job
	slurm_dependency_list+="$job_number:"
    done
    slurm_dependency_list=${slurm_dependency_list%?} # Remove trailing colon
    model=all
else
    launch_job
    slurm_dependency_list+="$job_number"
fi

launch_email $experiment_type $model "$experiment_args" $slurm_dependency_list
