#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=nlp
#SBATCH --exclude=jagupard[10-25]
#SBATCH --account=nlp

# Print execute commands in the log.
set -x
# conda_env=unlabeled_extrapolation
conda_env=`whoami`-ue-new

# source scripts/copy_imagenet_local.sh
source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda deactivate
# source ~/.bashrc
conda activate $conda_env
cd $PWD

eval $1

