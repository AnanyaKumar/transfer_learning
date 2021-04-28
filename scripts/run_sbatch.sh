#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --exclude=jagupard[4-8],jagupard[10-15]

# Print execute commands in the log.
set -x

conda activate `whoami`-ue
cd $PWD

eval $1

