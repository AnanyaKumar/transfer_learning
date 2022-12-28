
cl upload cifar10_dataset
cl upload stl10_dataset
cl upload domainnet
cl upload simclr_weights
cl upload configs
cl upload unlabeled_extrapolation
cl upload scripts

# CIFAR-STL experiments.
python scripts/run_adaptation_experiments.py --datasets cifar_stl --experiment=linprobe_experiments --codalab --only_one_run
python scripts/run_adaptation_experiments.py --datasets cifar_stl --experiment=fine_tuning_experiments --codalab --no_replications
python scripts/run_adaptation_experiments.py --datasets cifar_stl --experiment=lp_then_ft_valmode_experiments --codalab --no_replications

# DomainNet experiments.
python scripts/run_adaptation_experiments.py --datasets domainnet --model_name=clip_resnet50 --experiment=linprobe_experiments --codalab --only_one_run
python scripts/run_adaptation_experiments.py --datasets domainnet --model_name=clip_resnet50 --experiment=fine_tuning_experiments --codalab --no_replications
python scripts/run_adaptation_experiments.py --datasets domainnet --model_name=clip_resnet50 --experiment=lp_then_ft_valmode_experiments --codalab --no_replications
python scripts/run_adaptation_experiments.py --datasets domainnet --model_name=clip_resnet50 --experiment=ft_val_mode_experiment --codalab --no_replications

# FMoW experiments.
python scripts/run_adaptation_experiments.py --datasets fmow --model_name=mocotp_fmow_resnet50 --experiment=linprobe_experiments --codalab --only_one_run
python scripts/run_adaptation_experiments.py --datasets fmow --model_name=mocotp_fmow_resnet50 --experiment=fine_tuning_experiments --codalab --no_replications
python scripts/run_adaptation_experiments.py --datasets fmow --model_name=mocotp_fmow_resnet50 --experiment=lp_then_ft_valmode_experiments --codalab --no_replications

# Living-17, Entity-30, CIFAR-STL.
declare -a datasets=(living17 entity30)
for dataset in "${datasets[@]}"; do
    python scripts/run_adaptation_experiments.py --datasets $dataset --experiment=linprobe_experiments --codalab --only_one_run
    python scripts/run_adaptation_experiments.py --datasets $dataset --experiment=fine_tuning_experiments --codalab --no_replications
    python scripts/run_adaptation_experiments.py --datasets $dataset --experiment=lp_then_ft_valmode_experiments --codalab --no_replications

done


