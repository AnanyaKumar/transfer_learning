Code for our paper:

```
Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. 
Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang. ICLR 2022.
```


This paper will be published as an Oral at ICLR 2022.
This repository is still being improved and will be updated without backwards compatibility for now. 

## Setup and installation

The first time you run this project, in the current directory, which contains README, create a virtualenv:
```
python3 -m venv .env
source .env/bin/activate
pip install -f https://download.pytorch.org/whl/torch_stable.html -e .
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

## Running paper experiments (Slurm)

The main experiments are in scripts/run_adaptation_experiments.py. For example, if you're on the Stanford NLP cluster, then you can run fine-tuning experiments on Living-17 as follows:

```
python scripts/run_adaptation_experiments.py --experiment=fine_tuning_experiments --datasets living17 --model_name=resnet50 --partition=jag-standard
```

For other slurm clusters, you will need to modify the sbatch file "run_sbatch.sh" and change the partition accordingly.

To run LP-FT experiments:

```
python scripts/run_adaptation_experiments.py --experiment=lp_then_ft_valmode_experiments --datasets living17 --model_name=resnet50 --partition=jag-standard
```

To run linear probing experiments:

```
python scripts/run_adaptation_experiments.py --experiment=linprobe_experiments --datasets living17 --model_name=resnet50 --partition=jag-standard
```

This code may look complicated, but it's written to simplify launching and managing a large number of jobs on a cluster.
For example, scripts/run_adaptation_experiments.py essentially contains the information to run every experiment in our paper, and exactly what information to track, on a Slurm cluster.

After running the experiments, we have scripts that can produce a tsv file (e.g., which you can copy onto Excel or Google Sheets) with a detailed summary of all the runs (e.g., ID accuracy, OOD accuracy, when early stopping on a variety of metrics). For example, for Living17, run the following:

```
python  scripts/summarize_all_results.py --results_dir_glob=logs/*living17* --val_metrics test_acc/source_val_living test_acc/target_val_living LAST --output_metrics epoch train/acc test_acc/source_val_living test_acc/target_val_living --output_file=tmp.tsv 
```

The summary of all living17 runs will now be contained in tmp.tsv.

## Configs

The config files that specify hyperparameters are located in configs. Note that run_adaptation_experiments runs a sweep over these configs and therefore modifies the hyperparameters on configs. So consult run_adaptation_experiments.py for the hyperparameters used in the sweep.

## Main code

The main code for fine-tuning and LP-FT are in unlabeled_extrapolation/baseline_train.py.

For linear probing, you can first extract model features using unlabeled_extrapolation/extract_features.py, and then train a logistic regression classifier using log_reg_sk.py.

## Example run (without slurm)

You can run the training code directly without slurm.

```
export PYTHONPATH="."
. env/bin/python experiments/baseline_train.py --config=configs/adaptation/living17.yaml \\
--log_dir=logs/living17_lr_0.003 --project_name=living17 --group_name=living17 \\
--run_nameliving17_lr_0.003 --optimizer.args.lr=0.003
```

## Alternative setup (under construction): 
Use the `ananya-ue` conda env. Make sure you have this following code in your `.bashrc`:
```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/u/nlp/anaconda/main/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/u/nlp/anaconda/main/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/u/nlp/anaconda/main/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

## Guide to our codebase

The main script is in `experiments/baseline_train.py`. The script requires a YAML config
file - an example is `configs/adaptation/living17.yaml`.
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

If you base your config off of `configs/adaptation/living17.yaml` here are some important
args to keep in mind:
- finetune: If fine-tuning, this should be True.
- linear_probe: If only training the last linear layer (freeze lower layers), set True, for full fine-tuning set False
- use_net_val_mode: True if you want to keep the network in "val" mode while training. This should usually be True for linear probing, to turn off batchnorm.
- num_classes: Specifies the number of classes in the new task you're fine-tuning on.

Notice that we inherit datasets_living17.yaml, where we have:
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

# A Lightweight Framework for Training Models

Adapted from the framework for "In-N-Out: Pre-Training and Self-Training using Auxiliary Information for Out-of-Distribution Robustness", 
https://arxiv.org/abs/2012.04550, Sang Michael Xie*, Ananya Kumar*, Robbie Jones*, Fereshte Khani, Tengyu Ma, Percy Liang.

Benefits of framework:
- Config files to specify dataset, model, augmentations, and other arguments, for better readability and reproducibility
- Nice logging
- Saves checkpoints
- Sets up and outputs to weights and biases
- New: Saves the command line argument, and config, to run the experiment
- New: Optionally, can save all the code used to run the experiment so you know exactly what you ran
- New: can early stop based on a variety of metrics
- New: track a variety of metrics, and produce a nice TSV with all the results

Setup:
- First install a virtualenv and activate it (instructions coming)
- Create a weights and biases account
- Run "wandb login" and "wandb on"

