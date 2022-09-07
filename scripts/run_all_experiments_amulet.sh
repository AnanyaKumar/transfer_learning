#!/bin/bash

declare -a models=(timm_vit_b16_in21k clip_vit_b16 dino_vit_b16 bit_resnet_50_in21k convnext_vit_b clip_vit_l14 bit_resnet_101_in21k)
declare -a datasets=(waterbirds living17_nonorm domainnet)
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    python scripts/run_adaptation_experiments.py --experiment=fine_tuning_experiments --datasets $dataset --model_name=$model --partition=jag-standard --sbatch_script_name=run_rtx_sbatch.sh --amulet_option=run
    python scripts/run_adaptation_experiments.py --experiment=fine_tuning_experiments --datasets $dataset --model_name=$model --partition=jag-standard --amulet_option=run --sbatch_script_name=run_rtx_sbatch.sh --freeze_bottom_k=2
    python scripts/run_adaptation_experiments.py --experiment=fine_tuning_experiments --datasets $dataset --model_name=$model --partition=jag-standard --sbatch_script_name=run_rtx_sbatch.sh --optimizer='torch.optim.AdamW' --amulet_option=run
    python scripts/run_adaptation_experiments.py --experiment=fine_tuning_experiments --datasets $dataset --model_name=$model --partition=jag-standard --sbatch_script_name=run_rtx_sbatch.sh --optimizer='torch.optim.AdamW' --freeze_bottom_k=2 --amulet_option=run
  done
done

