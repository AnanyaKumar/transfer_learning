## Steps to run an experiment

The first time you run this project, in the current directory, which contains README, create a virtualenv:
```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```
In subsequent runs you only need to activate the environment:
```
source .env/bin/activate
```

Also login to weights and biases. Usually you don't have to re-run this and it remembers the login, even across sessions:
```
wandb login
wandb on
```

## Configs

The main script is in `experiments/baseline_train.py`. The script requires a YAML config
file - an example is `configs/breeds_living_moco_probe.yaml`.
To dynamically change values of the config file with command line arguments,
simply add new arguments of the form `--key=val` where the key can be any
string of multiple keys separated by periods. This is to allow for changing
nested components of the config file. For example `--model.args.depth=3` changes
the config dictionary in this way: `config['model']['args']['depth'] = 3`.
It is important that the key and value are separated by an equals sign.

We use Quinine so you can inherit other configs.
This makes things modularized if you have configs for a few datasets, and configs for a few models.
Try not to make these nested by more than one or two levels
(A inherits B, B inherits C, C inherits D, etc)

If you base your config off of `configs/breeds_living_moco_probe.yaml` here are some important
args to keep in mind:
- finetune: If fine-tuning, this should be True.
- linear_probe: If only training the last linear layer (freeze lower layers), set True, for full fine-tuning set False
- use_net_val_mode: True if you want to keep the network in "val" mode while training. This should usually be True for linear probing, to turn off batchnorm.
- num_classes: Specifies the number of classes in the new task you're fine-tuning on.

Notice that we inherit datasets_breeds_living.yaml, where we have:
- A train dataset, which we initialize using args, including a transform
- We specify a list of transforms, which are applied top to bottom
- We have a list of test_datasets, and we evaluate performance on all of these.
- Each test dataest has a name, which is used both for visualization, and to decide which ones to early stop on
- If a test_transform is not specified, it uses the default_test_transform for the test dataset
- You can also add "transforms" as a separate field for a test dataset, and we'll use that instead of default_test_transform
- We specify early_stop_dataset_names. We don't literally stop the experiment. But we save the best model checkpoint according to the accuracy on these datasets. If multiple datasets are specified, we save the best checkpoint according to each of these. E.g. you can have one where you early stop on in-domain validation, and an oracle model where you early stop on the OOD validation.

We also inherit resnet50_transfer.yaml which specifies the scheduler, learning rate, loss criterion, and a few other things that should be self-explanatory. Some things of note:
- log_interval is after how many training examples we log the training loss. This is cheap since we keep track of a rolling average in the epoch and output that.
- save_freq is how often we should save checkpoints besides the best checkpoint. For these experiments, the config sets this to a high number so we don't save checkpoints besides the initial, final, and best (saving disk space)
- model.args.pretrained is True, indicating that we initialize from a supervised Resnet50 model (pre-trained on ImageNet labels)

## Example run

```
export PYTHONPATH="."
. env/bin/python experiments/baseline_train.py --config=configs/breeds_living_moco_probe.yaml \\
--log_dir=logs/breeds_moco_linprobe_0.003 --project_name=breeds --group_name=breeds_moco_linprobe \\
--run_name=breeds_moco_linprobe_0.003 --optimizer.args.lr=0.003
```

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

