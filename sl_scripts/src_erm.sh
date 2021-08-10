sbatch configs/make_transfer_yaml.sh --id_domain real --ood_domain sketch --version sentry
sbatch configs/make_transfer_yaml.sh --id_domain sketch --ood_domain real --ood_domain painting --version sentry
sbatch configs/make_transfer_yaml.sh --id_domain painting --ood_domain sketch --version sentry

sbatch configs/make_transfer_yaml.sh --id_domain real --ood_domain sketch --version full
sbatch configs/make_transfer_yaml.sh --id_domain sketch --ood_domain real --ood_domain quickdraw --version full
sbatch configs/make_transfer_yaml.sh --id_domain quickdraw --ood_domain sketch --version full
