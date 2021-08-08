import argparse
from collections import namedtuple
from copy import deepcopy
import os
import re
import shlex
import subprocess

PROJECT_NAME = 'finetuning'
WORKSHEET_NAME = 'nlp::ananyak-fine-tuning'
DOCKER_IMAGE = 'ananya/unlabeled-extrapolation'


def run_sbatch(cmd, job_name, args, exclude=None,
               nodes=1, gres='gpu:1', cpus_per_task=2, mem='16G', deps=[]):
    output_path = args.output_dir + '/' + job_name
    sbatch_script_path = args.scripts_dir + '/' + args.sbatch_script_name 
    slurm_cmd = f'sbatch --partition={args.partition} --job-name={job_name} --output={output_path} ' +\
                f'--mail-type=END,FAIL --mail-user={args.mail_user} --nodes={nodes} ' +\
                f'--gres={gres} --cpus-per-task={cpus_per_task} --mem={mem} '
    deps = filter(lambda s: str(s) != '-1', deps)
    deps = [str(s) for s in deps]
    if len(deps) > 0:
        slurm_cmd += ' --dependency=afterok:' + ':'.join(deps) + ' '
    slurm_cmd += f' {sbatch_script_path} '
    slurm_cmd += f'"{cmd}"'
    print(slurm_cmd + '\n')
    output = subprocess.check_output(shlex.split(slurm_cmd)).decode('utf8')
    job_names = list(re.findall(r'\d+', output))
    assert(len(job_names) == 1)
    return job_names[0]


def run_codalab(cmd, job_name, args, gpus=1, mem='16G', cpus=1, nlp=True, deps=''):
    prefix = (f'cl run -n {job_name} -w {WORKSHEET_NAME} --request-docker-image={DOCKER_IMAGE} '
              f'--request-gpus={gpus} --request-memory={mem} --request-cpus={cpus} ')
    if nlp:
        nlp_opt = '--request-queue tag=nlp '
    else:
        nlp_opt = ''
    cl_deps_str = ':' + ' :'.join(deps)
    bundles = ':unlabeled_extrapolation :scripts :configs ' + cl_deps_str + ' '
    makedirs = '"export PYTHONPATH="."; export PYTHONPATH="' + args.code_dir + '"; '
    codalab_cmd = prefix + nlp_opt + bundles + makedirs + cmd + '"'
    print(codalab_cmd)
    subprocess.run(shlex.split(codalab_cmd))
    return job_name


def run_job(cmd, job_name, args, deps=[]):
    if args.codalab:
        return run_codalab(cmd, job_name, args, deps=deps)
    else:
        return run_sbatch(cmd, job_name, args, deps=deps)


def format_value(v):
    if type(v) == list:
        if type(v[0]) == list:
            raise ValueError('We only support 1D lists.')
        return ' ' + ' '.join([str(e) for e in v])
    return '=' + str(v)


def get_python_cmd(code_path, python_path='python', kwargs=None, args=None):
    if kwargs is not None:
        opts = ''.join([f"--{k}{format_value(v)} " for k, v in kwargs.items()])
        # opts += ''.join([f"--{k} " for k, v in kwargs.items() if isinstance(v, bool) and v and '.' not in k])
    else:
        opts = ''
    python_cmd = python_path + ' ' + code_path + ' '
    python_cmd += opts
    if args.codalab:
        python_cmd += ' --nowandb '
    return python_cmd


def group_run_to_log_path(group_name, run_name, args):
  return args.log_dir + '/' + group_name + '/' + run_name


def get_baseline_experiment_cmd(config_path, run_name, group_name, project_name, root_prefix,
                                kwargs, args, run_saved=False):
    # If run_saved, then we ignore root_prefix since we get this from the config.
    kwargs = deepcopy(kwargs)
    # Sometimes we might want to run from a saved json config file, in a custom location.
    # Saved files have full dataset paths, e.g. /scr/biggest/..., so no need to add root_prefix.
    kwargs['config'] = config_path
    if not(run_saved):
        if not(args.codalab):
            kwargs['root_prefix'] = root_prefix
        else:
            kwargs['root_prefix'] = args.codalab_data_dir
    # On slurm, we need to save locally to avoid overloading the distributed file system.
    if not(args.codalab):
        kwargs['log_dir'] = group_run_to_log_path(group_name, run_name, args)
        kwargs['tmp_par_ckp_dir'] = args.tmp_dir + '/' + group_name + '_' + run_name
    else:
        kwargs['log_dir'] = args.log_dir
    kwargs['project_name'] = project_name
    kwargs['group_name'] = group_name
    kwargs['run_name'] = run_name
    code_path = args.code_dir + '/' + 'baseline_train.py'
    return (get_python_cmd(code_path=code_path, python_path=args.python_path, kwargs=kwargs,
                           args=args),
            kwargs['log_dir'])


def config_run(args, kwargs, config_path, run_name, group_name, project_name,
               dataset_copy_cmd=None, root_prefix=None, run_saved=False, deps=[], rerun=False):
    # If rerun is False, don't rerun if the log firectory exists and contains stats.tsv.
    cmd, log_dir = get_baseline_experiment_cmd(
        config_path=config_path, run_name=run_name, group_name=group_name,
        project_name=project_name, root_prefix=root_prefix, kwargs=kwargs, args=args, run_saved=run_saved)
    if dataset_copy_cmd is not None and not args.codalab:
        cmd = dataset_copy_cmd + ' && ' + cmd
    job_name = group_name + '_' + run_name
    if os.path.isfile(log_dir + '/stats.tsv') and not(rerun):
        # TODO: should we rerun only if stats.tsv is entirely filled out?
        # Maybe design this a bit better.
        return -1
    else:
        return run_job(cmd, job_name, args, deps=deps)


############################################
## Functions to get directory/job names.
############################################

def hyperparams_to_str(hyperparams, item_sep='_', key_value_sep='-'):
    """Convert hyperparameters into string."""
    sorted_hyperparams = sorted(hyperparams.items())
    return item_sep.join([str(k) + key_value_sep + str(v) for k, v in sorted_hyperparams])
 
def get_group_name(adapt_name, dataset_name, model_name):
    return adapt_name+'_'+dataset_name+'_'+model_name
 
def get_job_name(adapt_name, dataset_name, model_name, hyperparams):
    """Get the name for a run."""
    hyperparams_str = hyperparams_to_str(hyperparams)
    return get_group_name(adapt_name, dataset_name, model_name)+'_'+hyperparams_str

def group_run_to_log_path(group_name, run_name, args):
    return args.log_dir + '/' + group_name + '/' + run_name
 
def get_run_dir_path(adapt_name, dataset_name, model_name, hyperparams, args):
    """Get path to directory for a specific run (method + dataset + hyperparameters)."""
    hyperparams_str = hyperparams_to_str(hyperparams)
    if args.codalab:
        return args.log_dir+'/'+get_job_name(adapt_name, dataset_name, hyperparams)
    else:
        group_dir_path = get_group_dir_path(adapt_name, dataset_name, model_name)
    return group_dir_path + '/ '+ hyperparams_str + '/'
 
def get_group_dir_path(adapt_name, dataset_name, model_name, args):
    """Get path to directory for all runs for an adaptation method + model on a dataset."""
    if args.codalab:
        return args.log_dir
    else:
        group_name = get_group_name(adapt_name, dataset_name, model_name)
        return args.log_dir + '/'+ group_name + '/'
 
def get_config_path(args, config_rel_path):
    return args.config_dir + '/' + config_rel_path


############################################
## Adaptation sweeps and replication.
############################################

def add_dataset_model_deps(deps, args, dataset, model):
    if args.codalab:
        deps = deps + dataset.bundles + model.bundles
    return deps


def get_best_config_path(adapt_name, dataset, model, args):
    if args.codalab:
        summarize_job_name = get_summarize_job_name(adapt_name, dataset.name, model.name)
        return summarize_job_name + '/best_config.json'
    else:
        group_dir = get_group_dir_path(adapt_name, dataset.name, model.name, args)
        return group_dir + '/best_config.json'


def add_model_to_kwargs(kwargs, args, model):
    kwargs['model.classname'] = model.classname
    kwargs['model.args.pretrained'] = model.pretrained
    kwargs['model.args.pretrain_style'] = model.pretrain_style
    if model.checkpoint_rel_path is not None:
        kwargs['model.args.checkpoint_path'] = (
            args.pretrained_checkpoints_dir + '/' + model.checkpoint_rel_path)


def run_adapt_sweep(adapt_name, dataset, model, hyperparams, args, deps=[], rerun=False):
    run_name = hyperparams_to_str(hyperparams)
    group_name = get_group_name(adapt_name, dataset.name, model.name)
    project_name = PROJECT_NAME
    kwargs = deepcopy(hyperparams)
    add_model_to_kwargs(kwargs, args, model)
    config_path = get_config_path(args, dataset.config_rel_path)
    dataset_copy_cmd = None
    if dataset.slurm_data_cmd is not None:
        dataset_copy_cmd = dataset.slurm_data_cmd.format(scripts_dir=args.scripts_dir)
    deps = add_dataset_model_deps(deps, args, dataset, model)
    return config_run(args, kwargs=kwargs, config_path=config_path,
        run_name=run_name, group_name=group_name, project_name=project_name,
        dataset_copy_cmd=dataset_copy_cmd, root_prefix=dataset.slurm_data_dir,
        deps=deps, rerun=rerun)


def run_adapt_replication(adapt_name, dataset, model, seed, args, deps=[],
                          replication_hyperparams=None, rerun=False):
    run_name = 'replication_'+str(seed)
    group_name = get_group_name(adapt_name, dataset.name, model.name)
    project_name = PROJECT_NAME
    best_config_path = get_best_config_path(adapt_name, dataset, model, args)
    if replication_hyperparams is not None:
        kwargs = deepcopy(replication_hyperparams)
    else:
        kwargs = {}
    kwargs['seed'] = seed
    dataset_copy_cmd = None
    if dataset.slurm_data_cmd is not None:
        dataset_copy_cmd = dataset.slurm_data_cmd.format(scripts_dir=args.scripts_dir)
    deps = add_dataset_model_deps(deps, args, dataset, model)
    return config_run(args, kwargs=kwargs, config_path=best_config_path ,
        run_name=run_name, group_name=group_name, project_name=project_name,
        dataset_copy_cmd=dataset_copy_cmd, root_prefix=dataset.slurm_data_dir,
        deps=deps, run_saved=True, rerun=rerun)


def adaptation_sweep(adapt_name, dataset, model, hyperparams_list, args, deps=[], rerun=False):
    sweep_ids = []
    for hyperparams in hyperparams_list:
        job_id = run_adapt_sweep(adapt_name, dataset, model,
            hyperparams=hyperparams, args=args, deps=deps, rerun=rerun)
        # Job id of -1 means we didn't run the job because it's already run.
        if job_id != -1:
            sweep_ids.append(job_id)
    return sweep_ids 


def adaptation_replication(adapt_name, dataset, model, num_replications, args, deps=[],
                           replication_hyperparams_list=None, rerun=False):
    # If replication_hyperparams_list is not None, then it should be the same size as
    # num_replications. And we will use the i-th entry of the list as additional hyperparameters
    # for the i-th run. Useful if the replication needs to vary checkpoints to other models.
    replication_ids = []
    assert (replication_hyperparams_list is None or
            len(replication_hyperparams_list) == num_replications)
    for i in range(num_replications):
        replication_hyperparams = None
        if replication_hyperparams_list is not None:
            replication_hyperparams = replication_hyperparams_list[i]
        job_id = run_adapt_replication(adapt_name, dataset, model, seed=args.seed+i+1,
            args=args, deps=deps, replication_hyperparams=replication_hyperparams, rerun=rerun)
        if job_id != -1:
            replication_ids.append(job_id)
    return replication_ids


def adaptation_experiment(adapt_name, dataset, model, hyperparams_list, num_replications, args,
                          deps=[], replication_hyperparams_list=None, rerun=False):
    """Sweep over hyperparams_list, find best, and replicate on a dataset for a model"""
    sweep_ids = adaptation_sweep(adapt_name, dataset, model, hyperparams_list,
        args=args, deps=deps, rerun=rerun)
    summarize_id = summarize_adaptation(adapt_name, dataset, model, args,
        deps=sweep_ids)
    replication_ids = adaptation_replication(adapt_name, dataset, model, num_replications,
        args=args, replication_hyperparams_list=replication_hyperparams_list,
        deps=[summarize_id], rerun=rerun)
    summarize_rep_id = summarize_adaptation(adapt_name, dataset, model, args, deps=replication_ids,
        replication=True)
    all_ids = sweep_ids + [summarize_id] + replication_ids + [summarize_rep_id]
    return [summarize_rep_id], all_ids


def linprobe_run(args, job_name, model, seed, config_path, features_save_path, results_save_path,
                 weights_save_path, val_metric, num_reg_values=50, deps=[], rerun=False):
    extract_code_path = args.code_dir + '/extract_features.py'
    kwargs = {}
    add_model_to_kwargs(kwargs, args, model)
    kwargs['config'] = config_path
    kwargs['save_path'] = features_save_path
    extract_cmd = get_python_cmd(code_path=extract_code_path, python_path=args.python_path,
                                 kwargs=kwargs, args=args)
    log_reg_code_path = args.code_dir + '/log_reg_sk.py'
    kwargs = {
        'load_path': features_save_path, 'results_save_path': results_save_path,
        'weights_save_path': weights_save_path, 'num_reg_values': num_reg_values,
        'val_metric': val_metric, 'seed': seed,
    }
    log_reg_cmd = get_python_cmd(code_path=log_reg_code_path, python_path=args.python_path,
                                  kwargs=kwargs, args=args)
    if os.path.isfile(features_save_path) and not(rerun):
        cmd = log_reg_cmd
    else:
        cmd = extract_cmd + ' && ' + log_reg_cmd 
    if os.path.isfile(results_save_path) and not(rerun):
        return -1
    return run_job(cmd, job_name, args, deps=deps)


def run_linprobe_replication(adapt_name, dataset, model, seed, args, deps=[], rerun=False):
    group_dir_path = get_group_dir_path(adapt_name, dataset.name, model.name, args)
    config_path = get_config_path(args, dataset.config_rel_path)
    features_save_path = group_dir_path + '/features_' + str(seed)
    # summarize_results looks for files which start with stats and end with .tsv
    results_save_path = group_dir_path + '/stats_' + str(seed) + '.tsv'
    weights_save_path = group_dir_path + '/weights_' + str(seed) + '.pkl'
    val_metric = dataset.val_metric
    job_name = get_group_name(adapt_name, dataset.name, model.name) + '_' + str(seed)
    deps = add_dataset_model_deps(deps, args, dataset, model)
    return linprobe_run(
        args, job_name, model, seed, config_path, features_save_path, results_save_path,
        weights_save_path, val_metric, deps=deps, rerun=rerun)


def linprobe_experiment(adapt_name, dataset, model, num_replications, args, deps=[], rerun=False):
    replication_ids = []
    for i in range(num_replications):
        job_id = run_linprobe_replication(adapt_name, dataset, model, seed=i, args=args,
                                          deps=deps, rerun=rerun)
        replication_ids.append(job_id)
    summarize_id = summarize_linprobe(adapt_name, dataset, model, args, deps=replication_ids)
    all_ids = replication_ids + [summarize_id]
    return [summarize_id], all_ids


############################################
## Functions to summarize results.
############################################

def get_summarize_job_name(adapt_name, dataset_name, model_name, replication=False):
    job_name = 'summarize_'+get_group_name(adapt_name, dataset_name, model_name)
    if replication:
        job_name = job_name + '_final'
    else:
        job_name = job_name + '_sweep'
    return job_name


def get_summarize_cmd(dir_path, val_metric, secondary_val_metrics, output_metrics, args,
    summarize_script_name='summarize_results.py'):
    """Python cmd to summarize all the results in dir_path according to specified metrics."""
    kwargs = {}
    kwargs['results_dir'] = dir_path
    kwargs['val_metric'] = val_metric
    if secondary_val_metrics is not None:
        kwargs['secondary_val_metrics'] = secondary_val_metrics
    if output_metrics is not None:
        kwargs['output_metrics'] = output_metrics
    code_path = args.scripts_dir + '/' + summarize_script_name
    return get_python_cmd(code_path, args.python_path, kwargs=kwargs, args=args)
 

def summarize_run(dir_path, job_name, val_metric, secondary_val_metrics, output_metrics, args, deps):
    """Run a job to summarize all the results in dir_path according to specified metrics."""
    cmd = get_summarize_cmd(dir_path, val_metric, secondary_val_metrics, output_metrics, args)
    return run_job(cmd, job_name, args, deps)


def summarize_adaptation(adapt_name, dataset, model, args, deps, replication=False):
    group_dir_path = get_group_dir_path(adapt_name, dataset.name, model.name, args)
    job_name = get_summarize_job_name(adapt_name, dataset.name, model.name, replication)
    return summarize_run(group_dir_path, job_name, dataset.val_metric, dataset.secondary_val_metrics,
        dataset.output_metrics, args, deps)


def summarize_linprobe_run(dir_path, job_name, val_metric, secondary_val_metrics, output_metrics, args, deps):
    cmd = get_summarize_cmd(dir_path, val_metric, secondary_val_metrics, output_metrics, args,
                            summarize_script_name='summarize_linprobe_results.py')
    return run_job(cmd, job_name, args, deps)


def summarize_linprobe(adapt_name, dataset, model, args, deps):
    group_dir_path = get_group_dir_path(adapt_name, dataset.name, model.name, args)
    job_name = get_summarize_job_name(adapt_name, dataset.name, model.name, replication=True)
    secondary_val_metrics = dataset.linprobe_secondary_val_metrics
    if secondary_val_metrics is None:
        secondary_val_metrics = dataset.secondary_val_metrics
    return summarize_linprobe_run(group_dir_path, job_name, dataset.val_metric,
        secondary_val_metrics, dataset.linprobe_output_metrics, args, deps)

############################################
## Datasets.
############################################


# If linprobe_secondary_val_metrics is None, use secondary_val_metrics.
Dataset = namedtuple(
    'Dataset',
    ['name', 'val_metric', 'secondary_val_metrics', 'output_metrics',
     'linprobe_secondary_val_metrics', 'linprobe_output_metrics',
     'config_rel_path', 'bundles', 'slurm_data_cmd', 'slurm_data_dir',
     'eval_config_rel_path'])
 
living17 = Dataset(
    name='living17',
    val_metric='test_acc/source_val_living',
    secondary_val_metrics=['test_acc/target_val_living', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    config_rel_path='adaptation/living17.yaml',
    bundles=['imagenet'],
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh imagenet',
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/living17_eval.yaml')
 
entity30 = Dataset(
    name='entity30',
    val_metric='test_acc/source_val_entity',
    secondary_val_metrics=['test_acc/target_val_entity', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/source_val_entity',
        'test_acc/target_val_entity'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/source_val_entity',
        'test_acc/target_val_entity'],
    config_rel_path='adaptation/entity30.yaml',
    bundles=['imagenet'],
    slurm_data_dir='/scr/biggest/',
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh imagenet',
    eval_config_rel_path='adaptation/entity30_eval.yaml')

cifar_stl = Dataset(
    name='cifar_stl',
    val_metric='test_acc/cifar10-test',
    secondary_val_metrics=['test_acc/stl-test', 'test_acc/imnet-n-cifar', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/cifar10-test',
        'test_acc/stl-test', 'test_acc/imnet-n-cifar'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/cifar10-test',
        'test_acc/stl-test', 'test_acc/imnet-n-cifar'],
    config_rel_path='adaptation/cifar_stl.yaml',
    bundles=['cifar_stl'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/ananya/',
    eval_config_rel_path='adaptation/cifar_stl_eval.yaml')


############################################
## Models.
############################################

Model = namedtuple(
    'Model',
    ['name', 'classname', 'pretrained', 'pretrain_style', 'checkpoint_rel_path', 'bundles'])

moco_resnet50 = Model(
    name='resnet50',
    classname='models.imnet_resnet.ResNet50',
    pretrained=True,
    pretrain_style='mocov2',
    checkpoint_rel_path='moco_v2_800ep_pretrain.pth.tar',
    bundles=['simclr_weights'])

swav_resnet50 = Model(
    name='swav_resnet50',
    classname='models.imnet_resnet.ResNet50',
    pretrained=True,
    pretrain_style='swav',
    checkpoint_rel_path='swav_800ep_pretrain.pth.tar',
    bundles=['simclr_weights'])

sup_resnet50 = Model(
    name='sup_resnet50',
    classname='models.imnet_resnet.ResNet50',
    pretrained=True,
    pretrain_style='supervised',
    checkpoint_rel_path=None,
    bundles=[])

############################################
## Functions to specify hyperparameter sweeps.
############################################

def union_dicts(d1, d2):
    return dict(d1, **d2)
 
def zip_dict_lists(dlist1, dlist2):
    dlist = []
    assert(len(dlist1) == len(dlist2))
    for i in range(len(dlist1)):
        dlist.append(union_dicts(dlist1[i], dlist2[i]))
    return dlist
 
def product_dict_lists(dlist1, dlist2):
    dlist = []
    for i in range(len(dlist1)):
        for j in range(len(dlist2)):
            dlist.append(union_dicts(dlist1[i], dlist2[j]))
    return dlist
 
def range_hyper(name, values):
    dlist = []
    for value in values:
        dlist.append({name: value})
    return dlist
 
# Append more_hyperparams to every hyperparams in hyperparams_list
def append_to_each(hyperparams_list, more_hyperparams):
    return [union_dicts(d, more_hyperparams) for d in hyperparams_list]


############################################
## Main experiments.
############################################

def fine_tuning_experiments(args):
    # TODO: enable all datasets.
    # datasets = [cifar_stl, living17, entity30]
    datasets = [cifar_stl]
    model = moco_resnet50
    # Fine-tuning (can optionally add 3e-2, 1e-1 if you want)
    # hyperparams_list = range_hyper('optimizer.args.lr', [3e-5, 1e-4])
    hyperparams_list = range_hyper('optimizer.args.lr', [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    hyperparams_list = append_to_each(hyperparams_list, {'seed': args.seed})
    for dataset in datasets:
        _, all_ids = adaptation_experiment(
            adapt_name='full_ft', dataset=dataset, model=model, hyperparams_list=hyperparams_list,
            num_replications=5, args=args)
        print('Job IDs: ' + ' '.join([str(id) for id in all_ids]))


def linprobe_experiments(args):
    datasets = [cifar_stl, living17, entity30]
    model = moco_resnet50
    for dataset in datasets:
        _, all_ids = linprobe_experiment(
            adapt_name='linprobe', dataset=dataset, model=model, num_replications=5, args=args)
        print('Job IDs: ' + ' '.join([str(id) for id in all_ids]))

def lp_then_ft_experiments(args):
    # datasets = [cifar_stl, living17, entity30]
    adapt_name = 'lp_then_ft'
    num_replications = 5
    datasets = [cifar_stl]
    model = moco_resnet50
    # Fine-tuning (can optionally add 3e-2, 1e-1 if you want)
    hyperparams_list = range_hyper('optimizer.args.lr', [3e-5, 1e-4])
    # hyperparams_list = range_hyper('optimizer.args.lr', [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    hyperparams_list = append_to_each(hyperparams_list, {'seed': args.seed})
    for dataset in datasets:
        cur_hyperparams_list = deepcopy(hyperparams_list)
        group_path = get_group_dir_path(adapt_name, dataset.name, model.name, args)
        cur_hyperparams_list = append_to_each(
            cur_hyperparams_list,
            {'linear_probe_checkpoint_path': group_path + '/weights_0.pkl'})
        replication_hyperparams_list = []
        for i in range(num_replications):
            replication_hyperparams_list.append({
                'linear_probe_checkpoint_path': group_path + '/weights_' + str(i) + '.pkl'})
        _, all_ids = adaptation_experiment(
            adapt_name=adapt_name, dataset=dataset, model=model,
            hyperparams_list=cur_hyperparams_list, num_replications=num_replications,
            replication_hyperparams_list=replication_hyperparams_list, args=args)
        print('Job IDs: ' + ' '.join([str(id) for id in all_ids]))


############################################
## Functions to spray dataset on jags.
############################################

def spray_dataset_jags(copy_cmd):
    for i in range(10, 30):
        cmd = f'sbatch -p jag-lo --cpus-per-task=1 --gres=gpu:0 --mem=4G --nodelist=jagupard{i} ' +\
              f'scripts/run_sbatch.sh "{copy_cmd}"'
        subprocess.run(cmd, shell=True)


def spray_fmow_jags(args):
    spray_dataset_jags(
        f'source {args.scripts_dir}/copy_local.sh /u/scr/nlp/wilds/data/fmow_v1.1.tar.gz wilds/data')


def spray_imagenet_jags(args):
    spray_dataset_jags(
        f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def spray_domainnet_jags(args):
    for i in range(10, 30):
        cmd = 'sbatch -p jag-lo --cpus-per-task=1 --gres=gpu:0 --mem=4G'
        cmd += f' --nodelist=jagupard{i} -J copy_domainnet_jag{i} -o %x.out'
        cmd += ' copy_dataset.sh domainnet'
        subprocess.run(shlex.split(cmd))


def main(args):
    experiment_to_fns = {
        'spray_fmow_jags': spray_fmow_jags,
        'spray_imagenet_jags': spray_imagenet_jags,
        'spray_domainnet_jags': spray_domainnet_jags,
        'fine_tuning_experiments': fine_tuning_experiments,
        'linprobe_experiments': linprobe_experiments,
        'lp_then_ft_experiments': lp_then_ft_experiments,
    }
    if args.experiment in experiment_to_fns:
        experiment_to_fns[args.experiment](args)
    else:
        raise ValueError(f'Experiment {args.experiment} does not exist.')


def fill_platform_specific_default_args(args):
    if args.codalab:
        args.log_dir = args.log_dir if args.log_dir else '.'
        args.pretrained_checkpoints_dir = (args.pretrained_checkpoints_dir if
                                           args.pretrained_checkpoints_dir else
                                           'simclr_weights/')
    else:
        args.log_dir = args.log_dir if args.log_dir else 'logs/'
        args.pretrained_checkpoints_dir = (args.pretrained_checkpoints_dir if
                                           args.pretrained_checkpoints_dir else
                                           '/u/scr/ananya/simclr_weights/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run celeba experiments.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment to run.')
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help='Base seed, we typically add to this seed for replication runs.')
    parser.add_argument('--codalab', action='store_true', help='run on CodaLab not slurm')
    parser.add_argument('--partition', type=str, required=False, default='jag-standard',
                        help='(Slurm only) What priority to use.')
    parser.add_argument('--mail_user', type=str, required=False,
                        help='(Slurm only) Email if slurm job fails.', default=None)
    # Locations of folders and files.
    parser.add_argument('--codalab_data_dir', type=str, required=False, default='.',
                        help='(Codalab only) Dir (root_prefix in config) where the data is stored')
    parser.add_argument('--scripts_dir', type=str, required=False, default='scripts/',
                        help='Path to dir where scripts are stored.')
    parser.add_argument('--output_dir', type=str, required=False, default='slurm_outputs/',
                        help='(Slurm only) Path to dir to store stdout for experiment.')
    parser.add_argument('--log_dir', type=str, required=False, default=None,
                        help='Path to dir where we save logs and run checkpoints.')
    parser.add_argument('--pretrained_checkpoints_dir', type=str, required=False,
                        default=None, help='Path to dir where we keep pretrained checkpoints.')
    parser.add_argument('--config_dir', type=str, required=False, default='configs/',
                        help='Directory where config files are stored.')
    parser.add_argument('--code_dir', type=str, required=False, default='unlabeled_extrapolation/',
                        help='Path to directory where code files are located.')
    parser.add_argument('--python_path', type=str, required=False, default='python',
                        help='Path or alias to Python interpreter')
    parser.add_argument('--tmp_dir', type=str, required=False, default='/scr/biggest/ue/',
                        help='(Slurm only) Directory where tmp files are stored.')
    parser.add_argument('--sbatch_script_name', type=str, required=False, default='run_sbatch.sh',
                        help='(Slurm only) sbatch script')
    args, unparsed = parser.parse_known_args()
    fill_platform_specific_default_args(args)
    main(args)

