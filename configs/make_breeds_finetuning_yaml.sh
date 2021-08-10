#!/bin/bash
#SBATCH --output=/juice/scr/kshen6/unlabeled_extrapolation/sl_outs/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --partition=jag-standard
#SBATCH --exclude=jagupard[14,15,19,20,25-29]

source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate kshen6-uee
echo $HOSTNAME

breeds_name=""
run_dir=""
ckp_num=399
epochs=50

while true; do
    case $1 in
    --breeds_name)
        if [ "$2" ]; then
        breeds_name=$2
        shift
        else
        echo '--breeds_name must be non-empty!'; exit 1
        fi
        ;;
    --run_dir)
        if [ "$2" ]; then
        run_dir=$2
        shift
        else
        echo '--run_dir must be non-empty!'; exit 1
        fi
        ;;
    --ckp_num)
        if [ "$2" ]; then
        ckp_num=$2
        shift
        else
        echo '--ckp_num must be non-empty!'; exit 1
        fi
        ;;
    --epochs)
        if [ "$2" ]; then
        epochs=$2
        shift
        else
        echo '--epochs must be non-empty!'; exit 1
        fi
        ;;
    *)
        if [ -z "$1" ]; then
           break
        else
        echo "Unsupported argument: $1"; exit 1
        fi
        ;;
    esac
    shift
done

set -x

if [[ $breeds_name != "entity30" && $breeds_name != "living17" ]]; then
    echo "--breeds_name must be one of entity30 or living17. Exiting..."
    exit 1
fi

if [[ $breeds_name == "entity30" ]]; then
    num_classes=30
else
    num_classes=17
fi

ckp_path="$run_dir/checkpoints/ckp-${ckp_num}.pth"

# get UUID, make a new file
uuid=$(uuidgen)
base_dir="/juice/scr/kshen6/unlabeled_extrapolation/configs"
base_file="${base_dir}/breeds_finetune_template.yaml"
new_config="${base_dir}/temp_configs/config_${uuid}.yaml"
cp $base_file $new_config

# replace everything that should be replaced
sed -i "s/BREEDS_NAME_REPLACE/${breeds_name}/" $new_config
sed -i "s/NUM_CLASSES_REPLACE/${num_classes}/" $new_config
sed -i "s|CHECKPOINT_PATH_REPLACE|${ckp_path}|" $new_config # use | to avoid collisions with / in path
sed -i "s/EPOCHS_REPLACE/${epochs}/" $new_config

# make sure that REPLACE is not found in new config. Otherwise fail.
left_to_replace=$(grep -c "REPLACE" "$new_config")
if (( $left_to_replace > 0 )); then
    echo "Improperly edited the template, still fields to replace"
    exit 0
fi

python unlabeled_extrapolation/baseline_train.py \
    --config $new_config \
    --log_dir $run_dir/finetuning/$uuid \
    --project_name breeds_erm \
    --group_name swav_finetuning \
    --run_name $breeds_name
