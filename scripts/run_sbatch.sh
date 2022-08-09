#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=jag-standard
#SBATCH --mem=16G
#SBATCH --exclude=jagupard[14,15,18]
# #SBATCH --exclude=jagupard[14,17,18,20]

# Print execute commands in the log.
set -x
conda_env=`whoami`-ue-new

# source scripts/copy_imagenet_local.sh
source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate $conda_env
cd $PWD

eval $1

