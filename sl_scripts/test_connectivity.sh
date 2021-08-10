#!/bin/bash

# Non-representation first
domainnet_path="/scr/biggest/domainnet"
prefix="python unlabeled_extrapolation/datasets/test_connectivity.py"

do_sbatch() {
    sbatch -p jag-standard -x "jagupard[11-14,16,17,20,24]" sl_scripts/run_sbatch_kendrick_gpu.sh "$1"
}

test_betweens=(
    "classes"
    "domains"
)

breeds_tasks=(
    "living17"
    "entity30"
)

source="real"
targets=(
    "sketch"
    "painting"
)

for test_between in ${test_betweens[@]}; do
    for breeds_task in ${breeds_tasks[@]}; do
        cmd="$prefix --dataset_name breeds --source $breeds_task --test_between $test_between --transform simclr"
        # do_sbatch "$cmd"
    done
    for target in ${targets[@]}; do
        cmd="$prefix --dataset_name domainnet --source $source --target $target --data_path $domainnet_path --test_between $test_between --transform simclr"
        do_sbatch "$cmd"
    done
done

# Using representations
readfile() {
    checkpoints=()
    while IFS= read -r line
    do
        if [[ ! "$line" =~ ^#.*  ]]; then
            checkpoints+=($line)
        fi
    done < sl_scripts/$1_ckpts.txt
}

for breeds_task in ${breeds_tasks[@]}; do
    readfile $breeds_task
    for test_between in ${test_betweens[@]}; do
        for checkpoint in ${checkpoints[@]}; do
            cmd="$prefix --dataset_name breeds --source $breeds_task --test_between $test_between --transform simclr --swav_dir $checkpoint --swav_ckpt ckp-399.pth"
            # do_sbatch "$cmd"
        done
    done
done

for target in ${targets[@]}; do
    readfile "${source}${target}"
    for test_between in ${test_betweens[@]}; do
        for checkpoint in ${checkpoints[@]}; do
            cmd="$prefix --dataset_name domainnet --source $source --target $target --data_path $domainnet_path --test_between $test_between --transform simclr --swav_dir $checkpoint --swav_ckpt ckp-399.pth"
            do_sbatch "$cmd"
        done
    done
done
