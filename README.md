
# A Lightweight Framework for Training Models

Adapted from Michael Xie's framework for "In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness", 
https://arxiv.org/abs/2012.04550, Sang Michael Xie*, Ananya Kumar*, Robbie Jones*, Fereshte Khani, Tengyu Ma, Percy Liang. Thanks to contributions by Robbie Jones.

Benefits of framework:
- Config files to specify dataset, model, augmentations, and other arguments, for better readability and reproducibility
- Nice logging
- Saves checkpoints
- Sets up and outputs to weights and biases
- New: Saves the command line argument, and config, to run the experiment
- New: Optionally, can save all the code used to run the experiment so you know exactly what you ran

Setup:
- First install a virtualenv and activate it (instructions coming)
- Create a weights and biases account
- Run "wandb login" and "wandb on"

To fine-tune an ImageNet model on CIFAR, run:
python experiments/baseline_train.py --config=configs/imnet_finetune_cifar_resnet_baseline.yaml --log_dir=logs/inet_ft_run_0 --project_name=cifar group_name=inet_ft --run_name=run_0

