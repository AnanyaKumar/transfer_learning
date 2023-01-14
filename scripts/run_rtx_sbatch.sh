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
# source ~/.bashrc
source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $conda_env
# alias python=python3.8
python --version
cd $PWD

eval $1

