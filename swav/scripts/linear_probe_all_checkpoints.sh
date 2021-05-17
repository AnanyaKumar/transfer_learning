#!/bin/bash

show_help() {
    usage_string="Usage: $(basename $0) script checkpoint_dir starting_port script_args..."
    echo "$usage_string"
}

if [ "$#" -lt 3 ]; then
    show_help
    exit 1
fi

script=$1
if ! command -v "$1" &> /dev/null
then
    echo "$1 is not a valid script"
    exit 1
fi

checkpoint_dir=$2
if [ ! -d "$checkpoint_dir" ]; then
    echo "$checkpoint_dir not a valid checkpoint directory!"
    exit 1
fi

starting_port=$3

shift 3
SLURM_COMMAND="sbatch $script"
for file in $checkpoint_dir/*; do
    file_base=$(basename $file)
    file_ext=${file_base#*.}
    if [[ "$file_ext" != pth && "$file_ext" != pth.tar ]]; then
	continue
    fi
    slurm_command="sbatch $script $file $@ -p $starting_port"
    eval "$slurm_command"
    starting_port=$((starting_port+1))
done
