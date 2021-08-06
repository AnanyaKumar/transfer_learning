#!/usr/bin/env bash
#SBATCH --partition=jag-standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --output=/scr/biggest/rmjones/slurm_logs/slurm-%A_%a.out
#SBATCH --mail-user=rmjones@cs.stanford.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue

# Print execute commands in the log.
set -x

DOMAINS=(real clipart painting sketch)
NUM_DOMAINS=${#DOMAINS[@]}
if [[ $SLURM_ARRAY_TASK_COUNT = $NUM_DOMAINS ]]; then
    source_domain_index=$((SLURM_ARRAY_TASK_ID % NUM_DOMAINS))
    source_domain=${DOMAINS[@]:source_domain_index:1}
    target_domain=all
elif [[ $SLURM_ARRAY_TASK_COUNT = $((NUM_DOMAINS ** 2)) ]]; then
    source_domain_index=$((SLURM_ARRAY_TASK_ID / NUM_DOMAINS))
    target_domain_index=$((SLURM_ARRAY_TASK_ID % NUM_DOMAINS))
    source_domain=${DOMAINS[@]:source_domain_index:1}
    target_domain=${DOMAINS[@]:target_domain_index:1}
else
    echo "Invalid number of tasks launched: $SLURM_ARRAY_TASK_COUNT"
    exit 1
fi

echo "Source Domain: $source_domain"
echo "Target Domain: $target_domain"

conda_env=`whoami`-ue

# source scripts/copy_imagenet_local.sh
source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $conda_env
cd $PWD
cmd_string="python run_clip_experiment.py $source_domain $target_domain"
cmd_string+=" $1"

eval $cmd_string
