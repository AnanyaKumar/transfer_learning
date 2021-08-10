#!/bin/bash

if [ "--prevent_excess_jobs" == $1 ]
then
    echo "Preventing queueing excess feature extraction jobs! (Make sure you know what this does)"
    prevent_excess_jobs=1
else
    prevent_excess_jobs=0
fi

avoid_jags="14,15,16,19,20,25-29"

# this runs first
extract_features() {
    if [ $# -eq 2 ]
    then
        swav_ckp_num=$2
    else
        swav_ckp_num=399
    fi
    cmd="python swav/finetuning/extract_features_new_fmt.py --dataset domainnet"
    cmd+=" --run_dir $run_dir --domainnet_version $version"
    cmd+=" --domains $1 --ckpt_epoch $swav_ckp_num"
    if [ "$prevent_excess_jobs" == 1 ]
    then
        # check if it's done running. If it is, then do not queue up a new job
        file_name="$run_dir/finetuning/features_and_labels_${version}_${swav_ckp_num}_new_fmt.pickle"
        if test -f "$file_name"; then
            run_1_slurm_id=-1
            return
        fi
    fi
    run_1_slurm_id=$(sbatch --parsable -p jag-standard -x "jagupard[$avoid_jags]" sl_scripts/run_sbatch_kendrick_gpu.sh "$cmd")
    echo "Feature extraction: $run_1_slurm_id"
}

# this doesn't run until after the feature extraction is complete
linear_probe() {
    if [ $# -eq 3 ]
    then
        swav_ckp_num=$3
    else
        swav_ckp_num=399
    fi
    # label_quantities="0.01 0.05 0.25 0.50 1.0"
    label_quantities="1.0"
    args="--run_dir $run_dir --train_data_fracs $label_quantities"
    args+=" --file_name features_and_labels_${version}_${swav_ckp_num}_new_fmt"
    args+=" --id_domain $1 --ood_domains $2"
    cmd_1="python swav/finetuning/log_reg_sk.py $args"
    cmd_2="python swav/finetuning/summarize_results.py $args"
    if [ "$run_1_slurm_id" == -1 ]
    then
        dependency=""
    else
        dependency="--dependency=afterok:$run_1_slurm_id"
    fi
    run_2_slurm_id=$(sbatch --parsable $dependency sl_scripts/run_sbatch_kendrick.sh "$cmd_1" "$cmd_2")
    echo "Linear probing: $run_2_slurm_id"
}

# run_dir and version arguments are shared across feature extraction and linear probing

#### SENTRY VERSIONS ####
version="sentry"

# SENTRY VERSION, SOURCE-ONLY REAL
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentrytrue_real_queue500_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes400"
extract_features "real,sketch"
linear_probe "real" "sketch"

# SENTRY VERSION, SOURCE-ONLY SKETCH
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sourcereal_targetsketch_sourceamount0_targetamount1.0_relatedamount0_queue15_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes400"
extract_features "sketch,real,painting"
linear_probe "sketch" "real,painting"

# SENTRY VERSION, SOURCE-ONLY PAINTING
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sourcereal_targetpainting_sourceamount0_targetamount1.0_relatedamount0_queue15_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes400"
extract_features "painting,sketch"
linear_probe "painting" "sketch"

# SENTRY VERSION, REAL + SKETCH
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentrytrue_real-sketch_queue500_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes400"
extract_features "real,sketch"
linear_probe "real" "sketch"
linear_probe "sketch" "real"

# SENTRY VERSION, SKETCH + PAINTING
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentrytrue_sketch-painting_queue500_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes400"
extract_features "sketch,painting"
linear_probe "sketch" "painting"
linear_probe "painting" "sketch"

# SENTRY VERSION, ALL DOMAINNET
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentrytrue_all_queue500_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes400"
extract_features "real,sketch,painting"
linear_probe "real" "sketch"
linear_probe "sketch" "real,painting"
linear_probe "painting" "sketch"

# FULL VERSION, ALL DOMAINNET, NO QUEUE (SENTRY LINPROB)
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentryfalse_all_queue500_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes3000"
extract_features "real,sketch,painting"
linear_probe "real" "sketch"
linear_probe "sketch" "real,painting"
linear_probe "painting" "sketch"

# FULL VERSION, ALL DOMAINNET, QUEUE (SENTRY LINPROB)
run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentryfalse_all_queue60_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes3000"
extract_features "real,sketch,painting" 350
linear_probe "real" "sketch" 350
linear_probe "sketch" "real,painting" 350
linear_probe "painting" "sketch" 350

#### FULL VERSIONS ####
version="full"

# # FULL VERSION, SOURCE-ONLY REAL
# run_dir=""
# extract_features "sketch,real"
# linear_probe "sketch" "real"

# # FULL VERSION, SOURCE-ONLY SKETCH
# run_dir=""
# extract_features "sketch,real,quickdraw"
# linear_probe "sketch" "real,quickdraw"

# # FULL VERSION, SOURCE-ONLY QUICKDRAW
# run_dir=""
# extract_features "quickdraw,sketch"
# linear_probe "quickdraw" "sketch"

# # FULL VERSION, REAL + SKETCH
# run_dir=""
# extract_features "sketch,real"
# linear_probe "real" "sketch"
# linear_probe "sketch" "real"

# # FULL VERSION, SKETCH + QUICKDRAW
# run_dir=""
# extract_features "sketch,quickdraw"
# linear_probe "sketch" "quickdraw"
# linear_probe "quickdraw" "sketch"

# FULL VERSION, ALL DOMAINNET, NO QUEUE
# run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentryfalse_all_queue500_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes3000"
# extract_features "real,sketch,quickdraw"
# linear_probe "real" "sketch"
# linear_probe "sketch" "real,quickdraw"
# linear_probe "quickdraw" "sketch"

# FULL VERSION, ALL DOMAINNET, QUEUE
# run_dir="/juice/scr/kshen6/unlabeled_extrapolation/swav/checkpoints/domainnet_sentryfalse_all_queue60_epochs400_batchsize128_epsilon0.03_archresnet50_prototypes3000"
# extract_features "real,sketch,quickdraw" 350
# linear_probe "real" "sketch" 350
# linear_probe "sketch" "real,quickdraw" 350
# linear_probe "quickdraw" "sketch" 350
