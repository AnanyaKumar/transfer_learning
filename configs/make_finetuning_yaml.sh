#!/bin/bash
#SBATCH --output=/juice/scr/kshen6/unlabeled_extrapolation/sl_outs/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --partition=jag-standard
#SBATCH --exclude=jagupard[14,15,19,20,25-29]
# #SBATCH --exclude=jagupard[11-14,16,17,19,20,24-29]

source /u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate kshen6-uee
echo $HOSTNAME

id_domain=""
ood_domains=()
version=""
run_dir=""
ckp_num=399
epochs=50

while true; do
    case $1 in
    --id_domain)
        if [ "$2" ]; then
        id_domain=$2
        shift
        else
        echo '--id_domain must be non-empty!'; exit 1
        fi
        ;;
    --ood_domain)
        if [ "$2" ]; then
        ood_domains+=($2)
        shift
        else
        echo '--ood_domain must be non-empty!'; exit 1
        fi
        ;;
    --version)
        if [ "$2" ]; then
        version=$2
        shift
        else
        echo '--version must be non-empty!'; exit 1
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

if [[ $version != "sentry" && $version != "full" ]]; then
    echo "--version must be one of sentry or full. Exiting..."
    exit 1
fi

if [[ $version == "sentry" ]]; then
    num_classes=40
else
    num_classes=345
fi

ckp_path="$run_dir/checkpoints/ckp-${ckp_num}.pth"

# get UUID, make a new file
uuid=$(uuidgen)
base_dir="/juice/scr/kshen6/unlabeled_extrapolation/configs"
base_file="${base_dir}/domainnet_finetune_template.yaml"
new_config="${base_dir}/temp_configs/config_${uuid}.yaml"
cp $base_file $new_config

# replace everything that should be replaced
sed -i "s/ID_DOMAIN_REPLACE/${id_domain}/" $new_config
sed -i "s/DOMAINNET_VERSION_REPLACE/${version}/" $new_config
sed -i "s/NUM_CLASSES_REPLACE/${num_classes}/" $new_config
sed -i "s|CHECKPOINT_PATH_REPLACE|${ckp_path}|" $new_config # use | to avoid collisions with / in path
sed -i "s/EPOCHS_REPLACE/${epochs}/" $new_config

ood_template="${base_dir}/ood_template.txt"
# parse the ood domains, edit the config
for (( i=0; i<${#ood_domains[@]}; i++)); do
    python configs/template_add_ood.py --file_name $new_config \
        --ood_domain ${ood_domains[i]} \
        --ood_template $ood_template
done

run_name="$id_domain"
for ood_domain in ${ood_domains[@]}; do
    run_name+="_${ood_domain}"
done
run_name+="_${version}"

replace_marker="# ADD OOD HERE"
sed -i "s/${replace_marker}//" $new_config

# make sure that REPLACE is not found in new config. Otherwise fail.
left_to_replace=$(grep -c "REPLACE" "$new_config")
if (( $left_to_replace > 0 )); then
    echo "Improperly edited the template, still fields to replace"
    exit 0
fi

python unlabeled_extrapolation/baseline_train.py \
    --config $new_config \
    --log_dir $run_dir/finetuning/$uuid \
    --project_name dn_erm \
    --group_name single_source \
    --run_name $run_name
