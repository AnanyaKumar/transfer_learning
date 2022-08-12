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


def run_sbatch(cmd, job_name, args, exclude=None, deps=[]):
    output_path = args.output_dir + '/' + job_name
    sbatch_script_path = args.scripts_dir + '/' + args.sbatch_script_name
    slurm_cmd = f'sbatch --partition={args.partition} --job-name={job_name} --output={output_path} ' +\
                f'--mail-type=END,FAIL --mail-user={args.mail_user} '
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
    elif args.print_command:
        print(cmd + '\n')
    else:
        return run_sbatch(cmd, job_name, args, deps=deps)


def format_key_value(k, v):
    if type(v) == list:
        if type(v[0]) == list:
            raise ValueError('We only support 1D lists.')
        return f'--{k} ' + ' '.join([str(e) for e in v])
    # I wanted to do this, but this messes up with update_config in utils.py, and hard to fix that.
    # if type(v) == bool:
    #     if v:
    #         return f'--{k}'
    #     return ''
    return f'--{k}=' + str(v)


def get_python_cmd(code_path, python_path='python', kwargs=None, args=None, overwrite_options=None):
    if kwargs is not None:
        # Make sure to keep the space at the end.
        opts = ''.join([f"{format_key_value(k, v)} " for k, v in kwargs.items()])
        # opts += ''.join([f"--{k} " for k, v in kwargs.items() if isinstance(v, bool) and v and '.' not in k])
    else:
        opts = ''
    python_cmd = python_path + ' ' + code_path + ' '
    python_cmd += opts
    if args.codalab:
        python_cmd += ' --nowandb '
    if overwrite_options is not None:
        python_cmd += ' ' + overwrite_options + ' '
    return python_cmd


def group_run_to_log_path(group_name, run_name, args):
  return args.log_dir + '/' + group_name + '/' + run_name


def get_baseline_experiment_cmd(config_path, run_name, group_name, project_name, root_prefix,
                                kwargs, args, run_saved=False, overwrite_options=None):
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
                           args=args, overwrite_options=overwrite_options),
            kwargs['log_dir'])


def config_run(args, kwargs, config_path, run_name, group_name, project_name,
               dataset_copy_cmd=None, root_prefix=None, run_saved=False, deps=[], rerun=False,
               overwrite_options=None):
    # If rerun is False, don't rerun if the log firectory exists and contains stats.tsv.
    cmd, log_dir = get_baseline_experiment_cmd(
        config_path=config_path, run_name=run_name, group_name=group_name,
        project_name=project_name, root_prefix=root_prefix, kwargs=kwargs, args=args,
        run_saved=run_saved, overwrite_options=overwrite_options)
    if dataset_copy_cmd is not None and not args.codalab:
        cmd = dataset_copy_cmd + ' && ' + cmd
    job_name = group_name + '_' + run_name
    if os.path.isfile(log_dir + '/stats.tsv') and not(rerun):
        # TODO: should we rerun if stats.tsv is not completely filled out?
        # Maybe design this a bit better.
        return -1
    else:
        return run_job(cmd, job_name, args, deps=deps)


############################################
## Functions to get directory/job names.
############################################

def hyperparams_to_str(hyperparams, item_sep='_', key_value_sep='-', ignore_name_hypers={}):
    """Convert hyperparameters into string."""
    sorted_hyperparams = sorted(hyperparams.items())
    return item_sep.join([str(k) + key_value_sep + str(v) for k, v in sorted_hyperparams
                          if k not in ignore_name_hypers])

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
        summarize_job_name = get_summarize_job_name(adapt_name, dataset.name, args.model_name)
        return summarize_job_name + '/best_config.json'
    else:
        group_dir = get_group_dir_path(adapt_name, dataset.name, args.model_name, args)
        return group_dir + '/best_config.json'


def add_model_to_kwargs(kwargs, args, model):
    assert 'classname' in model.kwargs
    for k, v in model.kwargs.items():
        if k == 'checkpoint_rel_path' and v is not None:
            kwargs['model.args.checkpoint_path'] = args.pretrained_checkpoints_dir + '/' + v
        else:
            kwargs['model.' + k] = v


def run_adapt_sweep(adapt_name, dataset, model, hyperparams, args, deps=[], rerun=False,
                    ignore_name_hypers={}, run_name_suffix=''):
    run_name = hyperparams_to_str(hyperparams, ignore_name_hypers=ignore_name_hypers)
    run_name += run_name_suffix
    group_name = get_group_name(adapt_name, dataset.name, args.model_name)
    project_name = PROJECT_NAME
    kwargs = deepcopy(hyperparams)
    add_model_to_kwargs(kwargs, args, model)
    config_path = get_config_path(args, dataset.config_rel_path)
    dataset_copy_cmd = None
    if dataset.slurm_data_cmd is not None:
        dataset_copy_cmd = dataset.slurm_data_cmd.format(scripts_dir=args.scripts_dir)
    deps = add_dataset_model_deps(deps, args, dataset, model)
    overwrite_options = dataset.overwrite_options
    return config_run(args, kwargs=kwargs, config_path=config_path,
        run_name=run_name, group_name=group_name, project_name=project_name,
        dataset_copy_cmd=dataset_copy_cmd, root_prefix=dataset.slurm_data_dir,
        deps=deps, rerun=rerun, overwrite_options=overwrite_options)


def run_adapt_replication(adapt_name, dataset, model, seed, args, deps=[],
                          replication_hyperparams=None, rerun=False):
    run_name = 'replication_'+str(seed)
    group_name = get_group_name(adapt_name, dataset.name, args.model_name)
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
    overwrite_options = dataset.overwrite_options
    return config_run(args, kwargs=kwargs, config_path=best_config_path ,
        run_name=run_name, group_name=group_name, project_name=project_name,
        dataset_copy_cmd=dataset_copy_cmd, root_prefix=dataset.slurm_data_dir,
        deps=deps, run_saved=True, rerun=rerun, overwrite_options=overwrite_options)


def adaptation_sweep(adapt_name, dataset, model, hyperparams_list, args, deps=[], rerun=False,
                     ignore_name_hypers={}):
    sweep_ids = []
    for hyperparams in hyperparams_list:
        job_id = run_adapt_sweep(adapt_name, dataset, model,
            hyperparams=hyperparams, args=args, deps=deps, rerun=rerun,
            ignore_name_hypers=ignore_name_hypers)
        # Job id of -1 means we didn't run the job because it's already run.
        if job_id != -1:
            sweep_ids.append(job_id)
    return sweep_ids


def replicated_sweep(adapt_name, dataset, model, hyperparams_list, num_replications,
                     args, deps=[], replication_hyperparams_list=[], rerun=False,
                     ignore_name_hypers={}):
    # Run multiple replications for each sweep run.
    sweep_ids = []
    for i in range(num_replications):
        for hyperparams in hyperparams_list:
            if len(replication_hyperparams_list) > 0:
                kwargs = union_dicts(hyperparams, replication_hyperparams_list[i])
            else:
                kwargs = deepcopy(hyperparams)
            kwargs['seed'] = args.seed + i
            job_id = run_adapt_sweep(adapt_name, dataset, model,
                hyperparams=kwargs, args=args, deps=deps, rerun=rerun,
                ignore_name_hypers=ignore_name_hypers, run_name_suffix='_run'+str(i))
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
                          deps=[], replication_hyperparams_list=None, rerun=False, ignore_name_hypers={}):
    """Sweep over hyperparams_list, find best, and replicate on a dataset for a model"""
    # ignore_name_hypers is the set of hyperparameters to ignore in the run name.
    sweep_ids = adaptation_sweep(adapt_name, dataset, model, hyperparams_list,
        args=args, deps=deps, rerun=rerun, ignore_name_hypers=ignore_name_hypers)
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
                 weights_save_path, val_metric, num_reg_values=50, deps=[], rerun=False, aug=True,
                 root_prefix='', train_mode=False, use_new_bn_stats=False,
                 overwrite_options=None):
    extract_code_path = args.code_dir + '/extract_features.py'
    kwargs = {}
    add_model_to_kwargs(kwargs, args, model)
    kwargs['config'] = config_path
    kwargs['save_path'] = features_save_path
    kwargs['root_prefix'] = root_prefix
    if train_mode:
        kwargs['train_mode'] = True
    if use_new_bn_stats:
        kwargs['use_new_bn_stats'] = True
    # If no augmentation, then use test transform for train.
    kwargs['use_test_transforms_for_train'] = not(aug)
    extract_cmd = get_python_cmd(code_path=extract_code_path, python_path=args.python_path,
                                 kwargs=kwargs, args=args, overwrite_options=overwrite_options)
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


def run_linprobe_replication(adapt_name, dataset, model, seed, args, deps=[], rerun=False,
                             aug=True, train_mode=False, use_new_bn_stats=False):
    group_dir_path = get_group_dir_path(adapt_name, dataset.name, args.model_name, args)
    config_path = get_config_path(args, dataset.config_rel_path)
    features_save_path = group_dir_path + '/features_' + str(seed)
    # summarize_results looks for files which start with stats and end with .tsv
    results_save_path = group_dir_path + '/stats_' + str(seed) + '.tsv'
    weights_save_path = group_dir_path + '/weights_' + str(seed) + '.pkl'
    val_metric = dataset.val_metric
    job_name = get_group_name(adapt_name, dataset.name, args.model_name) + '_' + str(seed)
    overwrite_options = dataset.overwrite_options
    deps = add_dataset_model_deps(deps, args, dataset, model)
    if not(args.codalab):
        root_prefix = dataset.slurm_data_dir
    else:
        root_prefix = args.codalab_data_dir
    return linprobe_run(
        args, job_name, model, seed, config_path, features_save_path, results_save_path,
        weights_save_path, val_metric, deps=deps, rerun=rerun, aug=aug, root_prefix=root_prefix,
        train_mode=train_mode, use_new_bn_stats=use_new_bn_stats,
        overwrite_options=overwrite_options)


def linprobe_experiment(adapt_name, dataset, model, num_replications, args, deps=[], rerun=False,
                        aug=True, train_mode=False, use_new_bn_stats=False):
    replication_ids = []
    for i in range(num_replications):
        job_id = run_linprobe_replication(adapt_name, dataset, model, seed=i, args=args,
                                          deps=deps, rerun=rerun, aug=aug, train_mode=train_mode,
                                          use_new_bn_stats=use_new_bn_stats)
        replication_ids.append(job_id)
    summarize_id = summarize_linprobe(adapt_name, dataset, model, args, deps=replication_ids, max_num=num_replications)
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
    summarize_script_name='summarize_results.py', max_num=None):
    """Python cmd to summarize all the results in dir_path according to specified metrics."""
    kwargs = {}
    kwargs['results_dir'] = dir_path
    kwargs['val_metric'] = val_metric
    if secondary_val_metrics is not None:
        kwargs['secondary_val_metrics'] = secondary_val_metrics
    if output_metrics is not None:
        kwargs['output_metrics'] = output_metrics
    if max_num is not None:
        kwargs['max_num'] = max_num
    code_path = args.scripts_dir + '/' + summarize_script_name
    return get_python_cmd(code_path, args.python_path, kwargs=kwargs, args=args)


def summarize_run(dir_path, job_name, val_metric, secondary_val_metrics, output_metrics, args, deps):
    """Run a job to summarize all the results in dir_path according to specified metrics."""
    cmd = get_summarize_cmd(dir_path, val_metric, secondary_val_metrics, output_metrics, args)
    return run_job(cmd, job_name, args, deps)


def summarize_adaptation(adapt_name, dataset, model, args, deps, replication=False):
    group_dir_path = get_group_dir_path(adapt_name, dataset.name, args.model_name, args)
    job_name = get_summarize_job_name(adapt_name, dataset.name, args.model_name, replication)
    return summarize_run(group_dir_path, job_name, dataset.val_metric, dataset.secondary_val_metrics,
        dataset.output_metrics, args, deps)


def summarize_linprobe_run(dir_path, job_name, val_metric, secondary_val_metrics, output_metrics, args, deps, max_num=None):
    cmd = get_summarize_cmd(dir_path, val_metric, secondary_val_metrics, output_metrics, args,
                            summarize_script_name='summarize_linprobe_results.py', max_num=max_num)
    return run_job(cmd, job_name, args, deps)


def summarize_linprobe(adapt_name, dataset, model, args, deps, max_num=None):
    group_dir_path = get_group_dir_path(adapt_name, dataset.name, args.model_name, args)
    job_name = get_summarize_job_name(adapt_name, dataset.name, args.model_name, replication=True)
    secondary_val_metrics = dataset.linprobe_secondary_val_metrics
    if secondary_val_metrics is None:
        secondary_val_metrics = dataset.secondary_val_metrics
    return summarize_linprobe_run(group_dir_path, job_name, dataset.val_metric,
        secondary_val_metrics, dataset.linprobe_output_metrics, args, deps,
        max_num=max_num)

############################################
## Datasets.
############################################


# If linprobe_secondary_val_metrics is None, use secondary_val_metrics.
# TODO: slurm_data_dir should populate root_prefix in configs, which it
# does not do at the moment.
fields = ['name', 'val_metric', 'secondary_val_metrics', 'output_metrics',
    'linprobe_secondary_val_metrics', 'linprobe_output_metrics',
    'config_rel_path', 'bundles', 'slurm_data_cmd', 'slurm_data_dir',
    'eval_config_rel_path', 'overwrite_options']

Dataset = namedtuple(
    'Dataset', fields, defaults=[None] * len(fields)
)

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

living17_mixup = Dataset(
    name='living17_mixup',
    val_metric='test_acc/source_val_living',
    secondary_val_metrics=['test_acc/target_val_living', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    config_rel_path='adaptation/living17_mixup.yaml',
    bundles=['imagenet'],
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh imagenet',
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/living17_mixup_eval.yaml')

celeba = Dataset(
    name='celeba',
    val_metric='test_acc/val',
    secondary_val_metrics=['test_acc/test', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/val',
        'test_acc/test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/val',
        'test_acc/test'],
    config_rel_path='adaptation/celeba.yaml',
    bundles=['celeba_pickle'],
    slurm_data_cmd=None,
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/celeba_eval.yaml')

waterbirds = Dataset(
    name='waterbirds',
    val_metric='test_acc/val',
    secondary_val_metrics=['LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    config_rel_path='adaptation/waterbirds.yaml',
    bundles=['waterbirds_pickle'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/nlp/',  # corresponds to root_prefix.
    eval_config_rel_path='adaptation/waterbirds_eval.yaml')

waterbirds_background = Dataset(
    name='waterbirds_background',
    overwrite_options=' --overwrite_dataset_name=waterbirds-background ',
    val_metric='test_acc/val',
    secondary_val_metrics=['LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    config_rel_path='adaptation/waterbirds.yaml',
    bundles=['waterbirds_pickle'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/nlp/',  # corresponds to root_prefix.
    eval_config_rel_path='adaptation/waterbirds_eval.yaml')

waterbirds_norm = Dataset(
    name='waterbirds_norm',
    val_metric='test_acc/val',
    secondary_val_metrics=['LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    config_rel_path='adaptation/waterbirds_norm.yaml',
    bundles=['waterbirds_pickle'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/nlp/',  # corresponds to root_prefix.
    eval_config_rel_path='adaptation/waterbirds_norm_eval.yaml')

waterbirds_augs = Dataset(
    name='waterbirds_augs',
    val_metric='test_acc/val',
    secondary_val_metrics=['LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/val',
        'test_acc/landbg-landbird-test', 'test_acc/landbg-waterbird-test',
        'test_acc/waterbg-landbird-test', 'test_acc/waterbg-waterbird-test'],
    config_rel_path='adaptation/waterbirds_augs.yaml',
    bundles=['waterbirds_pickle'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/nlp/',  # corresponds to root_prefix.
    eval_config_rel_path='adaptation/waterbirds_augs_eval.yaml')


living17_noaugs = Dataset(
    name='living17_noaugs',
    val_metric='test_acc/source_val_living',
    secondary_val_metrics=['test_acc/target_val_living', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    config_rel_path='adaptation/living17_noaugs.yaml',
    bundles=['imagenet'],
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh imagenet',
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/living17_eval.yaml')

living17_nonorm = Dataset(
    name='living17_nonorm',
    val_metric='test_acc/source_val_living',
    secondary_val_metrics=['test_acc/target_val_living', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/source_val_living',
        'test_acc/target_val_living'],
    config_rel_path='adaptation/living17_nonorm.yaml',
    bundles=['imagenet'],
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh imagenet',
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/living17_nonorm_eval.yaml')

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

imagenet = Dataset(
    name='imagenet',
    val_metric='test_acc/val',
    secondary_val_metrics=['test_acc/renditions' 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/val',
        'test_acc/renditions'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/val',
        'test_acc/renditions'],
    config_rel_path='adaptation/imagenet.yaml',
    bundles=['imagenet'],
    slurm_data_dir='',
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh imagenet',
    eval_config_rel_path='adaptation/imagenet_eval.yaml')

imagenet_augs = Dataset(
    name='imagenet_augs',
    val_metric='test_acc/val',
    secondary_val_metrics=['test_acc/renditions' 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/val',
        'test_acc/renditions'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/val',
        'test_acc/renditions'],
    config_rel_path='adaptation/imagenet_augs.yaml',
    bundles=['imagenet'],
    slurm_data_dir='',
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh imagenet',
    eval_config_rel_path='adaptation/imagenet_augs_eval.yaml')

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

cifar_stl_nonorm = Dataset(
    name='cifar_stl_nonorm',
    val_metric='test_acc/cifar10-test',
    secondary_val_metrics=['test_acc/stl-test', 'test_acc/imnet-n-cifar', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/cifar10-test',
        'test_acc/stl-test', 'test_acc/imnet-n-cifar'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/cifar10-test',
        'test_acc/stl-test', 'test_acc/imnet-n-cifar'],
    config_rel_path='adaptation/cifar_stl_nonorm.yaml',
    bundles=['cifar_stl'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/ananya/',
    eval_config_rel_path='adaptation/cifar_stl_nonorm_eval.yaml')


domainnet = Dataset(
    name='domainnet',
    val_metric='test_acc/sketch_val',
    secondary_val_metrics=['test_acc/real_val', 'test_acc/painting_val', 'test_acc/clipart_val', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/sketch_val',
        'test_acc/real_val', 'test_acc/painting_val', 'test_acc/clipart_val'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/sketch_val',
        'test_acc/real_val', 'test_acc/painting_val', 'test_acc/clipart_val'],
    config_rel_path='adaptation/domainnet.yaml',
    bundles=['domainnet'],
    slurm_data_cmd='source {scripts_dir}/copy_dataset.sh domainnet',
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/domainnet_eval.yaml')

fmow = Dataset(
    name='fmow',
    val_metric='test_acc/americas_val',
    secondary_val_metrics=['test_acc/africa_val', 'test_acc/europe_val', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/americas_val',
        'test_acc/africa_val', 'test_acc/europe_val'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/americas_val',
        'test_acc/africa_val', 'test_acc/europe_val'],
    config_rel_path='adaptation/fmow.yaml',
    bundles=['fmow'],
    slurm_data_cmd=None,
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/fmow_eval.yaml')

fmow_all = Dataset(
    name='fmow_all',
    val_metric='test_acc/ood_val',
    secondary_val_metrics=['test_acc/id_val', 'test_acc/test', 'test_acc/africa_test', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/id_val', 'test_acc/ood_val',
        'test_acc/test', 'test_acc/africa_test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/id_val', 'test_acc/ood_val',
        'test_acc/test', 'test_acc/africa_test'],
    config_rel_path='adaptation/fmow_all.yaml',
    bundles=['fmow'],
    slurm_data_cmd=None,
    slurm_data_dir='/scr/biggest/',
    eval_config_rel_path='adaptation/fmow_all_eval.yaml')

landcover = Dataset(
    name='landcover',
    val_metric='test_acc/nonafrica-val',
    secondary_val_metrics=['test_acc/africa', 'test_acc/nonafrica-test', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/nonafrica-val',
        'test_acc/africa', 'test_acc/nonafrica-test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/nonafrica-val',
        'test_acc/africa', 'test_acc/nonafrica-test'],
    config_rel_path='adaptation/landcover_to_africa.yaml',
    bundles=['landcover'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/nlp/eix/',
    eval_config_rel_path='adaptation/landcover_to_africa_eval.yaml')

landcover_auxin = Dataset(
    name='landcover_auxin',
    val_metric='test_acc/nonafrica-val',
    secondary_val_metrics=['test_acc/africa', 'test_acc/nonafrica-test', 'LAST'],
    output_metrics=['epoch', 'train/acc', 'test_acc/nonafrica-val',
        'test_acc/africa', 'test_acc/nonafrica-test'],
    linprobe_secondary_val_metrics=None,
    linprobe_output_metrics=['C', 'train/acc', 'test_acc/nonafrica-val',
        'test_acc/africa', 'test_acc/nonafrica-test'],
    config_rel_path='adaptation/landcover_auxin_to_africa.yaml',
    bundles=['landcover'],
    slurm_data_cmd=None,
    slurm_data_dir='/u/scr/nlp/eix/',
    eval_config_rel_path='adaptation/landcover_auxin_to_africa_eval.yaml')

names_to_datasets = {
    'living17': living17,
    'living17_nonorm': living17_nonorm,
    'living17_mixup': living17_mixup,
    'entity30': entity30,
    'imagenet': imagenet,
    'imagenet_augs': imagenet_augs,
    'cifar_stl': cifar_stl,
    'cifar_stl_nonorm': cifar_stl_nonorm,
    'domainnet': domainnet,
    'fmow': fmow,
    'fmow_all': fmow_all,
    'living17_noaugs': living17_noaugs,
    'celeba': celeba,
    'waterbirds': waterbirds,  # This dataset doesn't normalize.
    'waterbirds_augs': waterbirds_augs,  # This dataset doesn't normalize.
    'waterbirds_norm': waterbirds_norm,  # This dataset normalizes.
    'waterbirds_background': waterbirds_background,
    # 'landcover': landcover,
    # 'landcover_auxin': landcover_auxin,
}


############################################
## Models.
############################################

Model = namedtuple('Model', ['kwargs', 'bundles'])

mocotp_fmow_resnet50 = Model(
    kwargs={
        'classname': 'models.imnet_resnet.ResNet50',
        'args.pretrained': True,
        'args.pretrain_style': 'mocov2',
        'checkpoint_rel_path': 'mocotp_checkpoint_0200.pth.tar'
    },
    bundles=['simclr_weights']
)

scratch_resnet50 = Model(
    kwargs={
        'classname': 'models.imnet_resnet.ResNet50',
        'args.pretrained': False,
    },
    bundles=[]
)

moco_resnet50 = Model(
    kwargs={
        'classname': 'models.imnet_resnet.ResNet50',
        'args.pretrained': True,
        'args.pretrain_style': 'mocov2',
        'checkpoint_rel_path': 'moco_v2_800ep_pretrain.pth.tar'
    },
    bundles=['simclr_weights']
)

mocov1_resnet50 = Model(
    kwargs={
        'classname': 'models.imnet_resnet.ResNet50',
        'args.pretrained': True,
        'args.pretrain_style': 'mocov2',
        'checkpoint_rel_path': 'moco_v1_200ep_pretrain.pth.tar'
    },
    bundles=['simclr_weights']
)

swav_resnet50 = Model(
    kwargs={
        'classname': 'models.imnet_resnet.ResNet50',
        'args.pretrained': True,
        'args.pretrain_style': 'swav',
        'checkpoint_rel_path': 'swav_800ep_pretrain.pth.tar'
    },
    bundles=['simclr_weights']
)

sup_resnet50 = Model(
    kwargs={
        'classname': 'models.imnet_resnet.ResNet50',
        'args.pretrained': True,
        'args.pretrain_style': 'supervised',
    },
    bundles=[]
)

# Version where we normalize inside the model.
# TODO: ideally I think all normalization should be in the model...
sup_resnet50_norm = Model(
    kwargs={
        'classname': 'models.imnet_resnet.ResNet50',
        'args.pretrained': True,
        'args.pretrain_style': 'supervised',
        'args.normalize': True,
    },
    bundles=[]
)

clip_resnet50 = Model(
    kwargs={
        'classname': 'models.clip_model.ClipModel',
        'args.model_name': 'RN50',
    },
    bundles=[]
)

clip_vit_b16 = Model(
    kwargs={
        'classname': 'models.clip_model.ClipModel',
        'args.model_name': 'ViT-B/16',
    },
    bundles=[]
)

clip_vit_l14 = Model(
    kwargs={
        'classname': 'models.clip_model.ClipModel',
        'args.model_name': 'ViT-L/14',
    },
    bundles=[]
)

scratch_vit_b16_clipstyle = Model(
    kwargs={
        'classname': 'models.clip_model.ClipModel',
        'args.model_name': 'ViT-B/16',
        'args.scratch': True,
    },
    bundles=[]
)

scratch_vit_l14_clipstyle = Model(
    kwargs={
        'classname': 'models.clip_model.ClipModel',
        'args.model_name': 'ViT-L/14',
        'args.scratch': True,
    },
    bundles=[]
)

dino_vit_b16 = Model(
    kwargs={
        'classname': 'models.vit_model.VitModel',
        'args.model_name': 'dino_vitb16',
    },
    bundles=[]
)

bit_resnet_50 = Model(
    kwargs={
        'classname': 'models.bit_resnet.BitResNet',
        'args.model_name': 'BiT-M-R50x1',
        'checkpoint_rel_path': 'BiT-M-R50x1-ILSVRC2012.npz',
    },
    bundles=[]
)

landcover_baseline = Model(
    kwargs={
        'classname': 'models.innout_models.CNN1D',
        'args.in_channels': 8,
        'args.output_size': 6,
    },
    bundles=[]
)

landcover_auxin = Model(
    kwargs={
        'classname': 'models.innout_models.CNN1D',
        'args.in_channels': 14,
        'args.output_size': 6,
    },
    bundles=[]
)

names_to_model = {
    'scratch_resnet50': scratch_resnet50,
    'resnet50': moco_resnet50,
    'mocov1_resnet50': mocov1_resnet50,
    'swav_resnet50': swav_resnet50,
    'sup_resnet50': sup_resnet50,
    'sup_resnet50_norm': sup_resnet50_norm,
    'mocotp_fmow_resnet50': mocotp_fmow_resnet50,
    'clip_resnet50': clip_resnet50,
    'clip_vit_b16': clip_vit_b16,
    'clip_vit_l14': clip_vit_l14,
    'scratch_vit_b16_clipstyle': scratch_vit_b16_clipstyle,
    'dino_vit_b16': dino_vit_b16,
    'bit_resnet_50': bit_resnet_50,
    'landcover_baseline': landcover_baseline,
    'landcover_auxin': landcover_auxin,
}

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

SWEEP_LRS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

def get_datasets(args):
    print(args.datasets)
    if args.datasets is None:
        datasets = list(names_to_datasets.values())
    else:
        datasets = [names_to_datasets[n] for n in args.datasets]
    print(datasets)
    return datasets


def fine_tuning_celeba_single_experiment(args, attribute_name, num_replications=3,
                                  linear_probe=False, val_mode=False, final=True):
    adapt_name = 'full_ft'
    args.datasets = ['celeba']
    datasets = get_datasets(args)
    model = names_to_model[args.model_name]
    sweep_lrs = SWEEP_LRS
    if linear_probe:
        adapt_name = 'torch_linprobe'
        sweep_lrs = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
        if args.model_name == 'clip_vit_b16' and final:
            sweep_lrs = [0.01]  # We found this to be best on Wearing Earrings.
    elif args.model_name == 'clip_vit_b16':
        if final:
            sweep_lrs = [3e-06]  # We found this to be best on Wearing Earrings.
        else:
            sweep_lrs = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    elif args.model_name == 'scratch_vit_b16_clipstyle':
        if final:
            sweep_lrs = [0.003]  # We found this to be best on WearingEarrings.
        else:
            sweep_lrs = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    if args.epochs is not None:
        adapt_name += '_epochs' + str(args.epochs)
    adapt_name += '_' + attribute_name
    # Set hyperparameters
    if args.only_one_run:
        hyperparams_list = range_hyper('optimizer.args.lr', sweep_lrs[0])
        num_replications = 1
        # Would be num_replications = 0 if we used adaptation_experiment below.
    else:
        hyperparams_list = range_hyper('optimizer.args.lr', sweep_lrs)
    if val_mode:
        hyperparams_list = append_to_each(hyperparams_list, {'use_net_val_mode': True})
    hyperparams_list = append_to_each(hyperparams_list, {'seed': args.seed})
    if linear_probe:
        hyperparams_list = append_to_each(hyperparams_list, {'linear_probe': True})
    if args.epochs is not None:
        hyperparams_list = append_to_each(hyperparams_list, {'epochs': args.epochs})
        hyperparams_list = append_to_each(hyperparams_list, {'scheduler.args.T_max': args.epochs})
    if args.save_no_checkpoints:
        hyperparams_list = append_to_each(hyperparams_list, {'save_no_checkpoints': True})
    hyperparams_list = append_to_each(hyperparams_list, {'default_test_args.target_attribute': attribute_name})
    hyperparams_list = append_to_each(hyperparams_list, {'train_dataset.args.target_attribute': attribute_name})
    if args.no_replications:
        num_replications = 1
        # Would be num_replications = 0 if we used adaptation_experiment below.
    for dataset in datasets:
        all_ids = replicated_sweep(
            adapt_name=adapt_name, dataset=dataset, model=model, hyperparams_list=hyperparams_list,
            num_replications=num_replications, args=args, ignore_name_hypers={'checkpoint_path', 'default_test_args.target_attribute', 'train_dataset.args.target_attribute'})
        print('Job IDs: ' + ' '.join([str(id) for id in all_ids]))


def fine_tuning_celeba_experiments(args, linear_probe=True):
    # fine_tuning_celeba_single_experiment(args, 'Wearing_Earrings', linear_probe=True, val_mode=True)
    if linear_probe:
        fine_tuning_celeba_single_experiment(args, 'Wearing_Earrings', num_replications=5, linear_probe=True, val_mode=True, final=True)
        fine_tuning_celeba_single_experiment(args, 'Wearing_Necklace', num_replications=5, linear_probe=True, val_mode=True, final=True)
        fine_tuning_celeba_single_experiment(args, 'Wearing_Necktie', num_replications=5, linear_probe=True, val_mode=True, final=True)
        fine_tuning_celeba_single_experiment(args, 'Eyeglasses', num_replications=5, linear_probe=True, val_mode=True, final=True)
    else:
        fine_tuning_celeba_single_experiment(args, 'Wearing_Earrings', num_replications=5, final=True)
        fine_tuning_celeba_single_experiment(args, 'Wearing_Necklace', num_replications=5, final=True)
        fine_tuning_celeba_single_experiment(args, 'Wearing_Necktie', num_replications=5, final=True)
        fine_tuning_celeba_single_experiment(args, 'Eyeglasses', num_replications=5, final=True)



def fine_tuning_experiments(args, num_replications=3, linear_probe=False, batchnorm_ft=False, higher_linear_lr=False,
                            val_mode=False, no_augmentation=False, l2sp=False, side_tune=False,
                            imagenet_lp_ft_phase2=False, mixup_sweep=False):
    adapt_name = 'full_ft'
    datasets = get_datasets(args)
    model = names_to_model[args.model_name]
    sweep_lrs = SWEEP_LRS
    if len(args.datasets) > 1:
        print('WARNING: learning rates sweeps will not be done properly with > 1 dataset.\n\n')
    if 'imagenet' in args.datasets[0]:
        if len(args.datasets) > 1:
            raise ValueError('ImageNet uses custom learning rates, so launch it separately.')
        if imagenet_lp_ft_phase2:
            sweep_lrs = [0.00001, 0.00003, 0.0001]
        else:
            sweep_lrs = [0.0001, 0.0003, 0.001]
    if 'waterbirds' in args.datasets[0]:
        sweep_lrs = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    if side_tune:
        adapt_name += '_side_tune'
        sweep_lrs = sweep_lrs + [3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
        if val_mode:
            adapt_name += '_valmode'
            sweep_lrs = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    elif val_mode:
        adapt_name += '_valmode'
        if ('waterbirds' not in args.datasets[0] and 'imagenet' not in args.datasets[0]):
            sweep_lrs = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    if no_augmentation:
        adapt_name += '_no_augmentation'
    if linear_probe:
        adapt_name = 'torch_linprobe'
        # Linear probing needs a higher learning rate.
        if 'imagenet' in args.datasets or 'imagenet_augs' in args.datasets:
            sweep_lrs = [0.01, 0.03, 0.1]
        else:
            sweep_lrs = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    if batchnorm_ft:
        adapt_name = 'batchnorm_ft'
        # TODO: hacky / hardcoded.
        sweep_lrs = SWEEP_LRS[2:] + [0.03, 0.1, 0.3]
    if higher_linear_lr:
        adapt_name = 'full_ft_higherlinlr'
    if l2sp:
        adapt_name = 'l2sp'
    if args.epochs is not None:
        adapt_name += '_epochs' + str(args.epochs)
    # Set hyperparameters
    if args.only_one_run:
        if 'waterbirds' in args.datasets[0] and (args.freeze_bottom_k is None or
            args.freeze_bottom_k == 0):
            hyperparams_list = range_hyper('optimizer.args.lr', [1e-3])
        elif 1e-3 in sweep_lrs:
            # 1e-3 is a good default value for fine-tuning (with SGD).
            hyperparams_list = range_hyper('optimizer.args.lr', [1e-3])
        else:
            hyperparams_list = range_hyper('optimizer.args.lr', [sweep_lrs[0]])
        # TODO: remove this?
        num_replications = 1
        # Would be num_replications = 0 if we used adaptation_experiment below.
    elif mixup_sweep:
        lr_hypers = range_hyper('optimizer.args.lr', sweep_lrs)
        # TODO: add 1.0 to this as well.
        mixup_alpha_hypers = range_hyper('mixup_alpha', [0.5])
        hyperparams_list = product_dict_lists(lr_hypers, mixup_alpha_hypers)
    else:
        hyperparams_list = range_hyper('optimizer.args.lr', sweep_lrs)
    if side_tune:
        hyperparams_list = append_to_each(hyperparams_list, {'side_tune': True})
    if no_augmentation:
        hyperparams_list = append_to_each(hyperparams_list, {'no_augmentation': True})
    if val_mode:
        hyperparams_list = append_to_each(hyperparams_list, {'use_net_val_mode': True})
    if args.freeze_bottom_k is not None:
        hyperparams_list = append_to_each(
            hyperparams_list, {'freeze_bottom_k': args.freeze_bottom_k})
        adapt_name += '_freeze_bottom_' + str(args.freeze_bottom_k)
    if args.full_ft_epoch is not None:
        hyperparams_list = append_to_each(
            hyperparams_list, {'full_ft_epoch': args.full_ft_epoch})
        adapt_name += '_full_ft_epoch_' + str(args.full_ft_epoch)
    hyperparams_list = append_to_each(hyperparams_list, {'seed': args.seed})
    if linear_probe:
        hyperparams_list = append_to_each(hyperparams_list, {'linear_probe': True})
    if batchnorm_ft:
        hyperparams_list = append_to_each(hyperparams_list, {'batchnorm_ft': True})
    if higher_linear_lr:
        hyperparams_list = append_to_each(hyperparams_list, {'linear_layer_lr_multiplier': 10})
    if l2sp:
        # Note: tried 1.0, 0.1, 0.01, 0.001, 0.0001 on Living-17
        # 0.01 worked best ID, and 0.1 worked best OOD but did 0.4% worse than fine-tuning ID.
        hyperparams_list = append_to_each(hyperparams_list, {'l2sp_weight': 0.01})
    if args.epochs is not None:
        hyperparams_list = append_to_each(hyperparams_list, {'epochs': args.epochs})
        hyperparams_list = append_to_each(hyperparams_list, {'scheduler.args.T_max': args.epochs})
    if args.save_no_checkpoints:
        hyperparams_list = append_to_each(hyperparams_list, {'save_no_checkpoints': True})
    if imagenet_lp_ft_phase2:
        if 'imagenet_augs' in args.datasets:
            hyperparams_list = append_to_each(hyperparams_list,
                {'checkpoint_path':
                    '/u/scr/ananya/cifar_experiments/unlabeled_extrapolation/logs/'
                    'torch_linprobe_epochs5_imagenet_augs_clip_vit_b16/'
                    'epochs-5_linear_probe-True_optimizer.args.lr-0.1_scheduler.args.'
                    'T_max-5_seed-0_run0/checkpoints/ckp_best_val'})

        if 'imagenet' in args.datasets:
            hyperparams_list = append_to_each(hyperparams_list,
                {'checkpoint_path':
                    '/u/scr/ananya/cifar_experiments/unlabeled_extrapolation/logs/'
                    'torch_linprobe_epochs5_imagenet_clip_vit_b16/'
                    'epochs-5_linear_probe-True_optimizer.args.lr-0.1_scheduler.args.'
                    'T_max-5_seed-0_run0/checkpoints/ckp_best_val'})
    if args.no_replications:
        num_replications = 1
        # Would be num_replications = 0 if we used adaptation_experiment below.
    for dataset in datasets:
        all_ids = replicated_sweep(
            adapt_name=adapt_name, dataset=dataset, model=model, hyperparams_list=hyperparams_list,
            num_replications=num_replications, args=args, ignore_name_hypers={'checkpoint_path'})
        print('Job IDs: ' + ' '.join([str(id) for id in all_ids]))


def fine_tuning_mixup_sweep_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications,
                            mixup_sweep=True)


def ft_imnet_lp_ft_phase_2_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications,
                            imagenet_lp_ft_phase2=True)


def ft_imnet_lp_ft_phase_2_val_mode_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications,
                            val_mode=True, imagenet_lp_ft_phase2=True)


def fine_tuning_no_augmentation_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, no_augmentation=True)


def torch_linprobe_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, linear_probe=True)


def torch_linprobe_valmode_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, linear_probe=True,
                            val_mode=True)


def side_tune_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, side_tune=True)


def side_tune_val_mode_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, val_mode=True, side_tune=True)


def batchnorm_ft_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, batchnorm_ft=True)


def ft_higher_linear_lr_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, higher_linear_lr=True)


def l2sp_experiments(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, l2sp=True)


def ft_val_mode_experiment(args, num_replications=3):
    fine_tuning_experiments(args, num_replications=num_replications, val_mode=True)


def linprobe_experiments(args, num_replications=3, aug=True, train_mode=False, use_new_bn_stats=False):
    adapt_name = 'linprobe'
    if not(aug):
        adapt_name += '_noaug'
    if train_mode:
        adapt_name += '_trainmode'
    if use_new_bn_stats:
        if train_mode:
            raise ValueError('If use_new_bn_stats is True, train_mode must be False.')
        adapt_name += '_usenewbnstats'
    datasets = get_datasets(args)
    model = names_to_model[args.model_name]
    if args.no_replications or args.only_one_run:
        num_replications = 1
    for dataset in datasets:
        all_ids = linprobe_experiment(
            adapt_name=adapt_name, dataset=dataset, model=model, num_replications=num_replications,
            args=args, aug=aug, train_mode=train_mode, use_new_bn_stats=use_new_bn_stats)
        print('Job IDs: ' + ' '.join([str(id) for id in all_ids]))


def linprobe_experiments_no_aug(args, num_replications=3):
    linprobe_experiments(args, num_replications=num_replications, aug=False)


def linprobe_experiments_trainmode(args, num_replications=3):
    linprobe_experiments(args, num_replications=num_replications, train_mode=True)


def linprobe_experiments_usenewbnstats(args, num_replications=3):
    linprobe_experiments(args, num_replications=num_replications, use_new_bn_stats=True)


def lp_then_ft_experiments(args, num_replications=3, val_mode=False, train_mode=False, use_new_bn_stats=False):
    if args.epochs is not None:
        raise ValueError('Does not support epochs yet.')
    adapt_name = 'lp_then_ft'
    sweep_lrs = SWEEP_LRS
    if val_mode:
        adapt_name += '_valmode'
        sweep_lrs = [1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]
    linprobe_adapt_name = 'linprobe'
    datasets = get_datasets(args)
    model = names_to_model[args.model_name]
    if args.only_one_run:
        hyperparams_list = range_hyper('optimizer.args.lr', [sweep_lrs[0]])
        num_replications = 1
        # Would be num_replications = 0 if we used adaptation_experiment below.
    else:
        hyperparams_list = range_hyper('optimizer.args.lr', sweep_lrs)
    if val_mode:
        hyperparams_list = append_to_each(hyperparams_list, {'use_net_val_mode': True})
    if args.freeze_bottom_k is not None:
        hyperparams_list = append_to_each(
            hyperparams_list, {'freeze_bottom_k': args.freeze_bottom_k})
        adapt_name += '_freeze_bottom_' + str(args.freeze_bottom_k)
    if args.no_replications:
        num_replications = 1
        # Would be num_replications = 0 if we used adaptation_experiment below.
    if args.epochs is not None:
        hyperparams_list = append_to_each(hyperparams_list, {'epochs': args.epochs})
        hyperparams_list = append_to_each(hyperparams_list, {'scheduler.args.T_max': args.epochs})
    if args.save_no_checkpoints:
        hyperparams_list = append_to_each(hyperparams_list, {'save_no_checkpoints': True})
    for dataset in datasets:
        cur_hyperparams_list = deepcopy(hyperparams_list)
        linprobe_group_path = get_group_dir_path(linprobe_adapt_name, dataset.name, args.model_name, args)
        # cur_hyperparams_list = append_to_each(
        #     cur_hyperparams_list,
        #     {'linear_probe_checkpoint_path': linprobe_group_path + '/weights_0.pkl'})
        replication_hyperparams_list = []
        for i in range(num_replications):
            replication_hyperparams_list.append({
                'linear_probe_checkpoint_path': linprobe_group_path + '/weights_' + str(i) + '.pkl'})
        all_ids = replicated_sweep(
            adapt_name=adapt_name, dataset=dataset, model=model,
            hyperparams_list=cur_hyperparams_list, num_replications=num_replications,
            replication_hyperparams_list=replication_hyperparams_list, args=args,
            ignore_name_hypers={'linear_probe_checkpoint_path'})
        print('Job IDs: ' + ' '.join([str(id) for id in all_ids]))


def lp_then_ft_valmode_experiments(args, num_replications=3):
    lp_then_ft_experiments(args, num_replications=num_replications, val_mode=True)


def lp_then_ft_trainmode_experiments(args, num_replications=3):
    lp_then_ft_experiments(args, num_replications=num_replications, train_mode=True)


def lp_then_ft_usenewbnstats_experiments(args, num_replications=3):
    lp_then_ft_experiments(args, num_replications=num_replications, use_new_bn_stats=True)


############################################
## Functions to spray dataset on jags.
############################################

def spray_dataset_jags(copy_cmd):
    for i in range(10, 30):
        cmd = f'sbatch -p jag-lo --cpus-per-task=1 --gres=gpu:0 --mem=4G --nodelist=jagupard{i} ' +\
              f'scripts/run_sbatch.sh "{copy_cmd}"'
        subprocess.run(cmd, shell=True)


def spray_celeba_jags(args):
    spray_dataset_jags(
        f'source {args.scripts_dir}/copy_local.sh /u/scr/ananya/celeba.tar.gz')


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
        cmd += ' scripts/copy_dataset.sh domainnet'
        subprocess.run(shlex.split(cmd))


def main(args):
    experiment_to_fns = {
        'spray_celeba_jags': spray_celeba_jags,
        'spray_fmow_jags': spray_fmow_jags,
        'spray_imagenet_jags': spray_imagenet_jags,
        'spray_domainnet_jags': spray_domainnet_jags,
        'fine_tuning_experiments': fine_tuning_experiments,
        'fine_tuning_celeba_experiments': fine_tuning_celeba_experiments,
        'linprobe_experiments': linprobe_experiments,
        'linprobe_experiments_no_aug': linprobe_experiments_no_aug,
        'lp_then_ft_experiments': lp_then_ft_experiments,
        'linprobe_experiments_trainmode': linprobe_experiments_trainmode,
        'lp_then_ft_trainmode_experiments': lp_then_ft_trainmode_experiments,
        'linprobe_experiments_usenewbnstats': linprobe_experiments_usenewbnstats,
        'lp_then_ft_usenewbnstats_experiments': lp_then_ft_usenewbnstats_experiments,
        'lp_then_ft_valmode_experiments': lp_then_ft_valmode_experiments,
        'torch_linprobe_experiments': torch_linprobe_experiments,
        'torch_linprobe_valmode_experiments': torch_linprobe_valmode_experiments,
        'batchnorm_ft_experiments': batchnorm_ft_experiments,
        'ft_higher_linear_lr_experiments': ft_higher_linear_lr_experiments,
        'ft_val_mode_experiment': ft_val_mode_experiment,
        'fine_tuning_no_augmentation_experiments': fine_tuning_no_augmentation_experiments,
        'l2sp_experiments': l2sp_experiments,
        'side_tune_experiments': side_tune_experiments,
        'side_tune_val_mode_experiments': side_tune_val_mode_experiments,
        'ft_imnet_lp_ft_phase_2_experiments': ft_imnet_lp_ft_phase_2_experiments,
        'ft_imnet_lp_ft_phase_2_val_mode_experiments': ft_imnet_lp_ft_phase_2_val_mode_experiments,
        'fine_tuning_mixup_sweep_experiments': fine_tuning_mixup_sweep_experiments,
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
    parser.add_argument('--epochs', type=int, required=False, default=None,
                        help='Number of epochs to run job for.')
    # Note that store_true creates a default value of False.
    parser.add_argument('--codalab', action='store_true', help='run on CodaLab not slurm')
    parser.add_argument('--print_command', action='store_true', help='only print the python commands (dont run anything).')
    parser.add_argument('--save_no_checkpoints', action='store_true', help='run on CodaLab not slurm')
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
    parser.add_argument('--datasets', type=str, nargs='+',
                        help='Datasets to test on (if unspecified, run on all).', required=False)
    parser.add_argument('--model_name', type=str, default='resnet50',  # This is moco resnet.
                        help='Model to use', required=False)
    parser.add_argument('--freeze_bottom_k', type=int, required=False, default=None,
                        help='Freeze bottom k layers (if not specified, don\'t freeze).')
    parser.add_argument('--full_ft_epoch', type=int, required=False, default=None,
                        help='At what epoch should we unfreeze all weights and fine-tune.')
    parser.add_argument('--only_one_run', action='store_true',
                        help=('Only run one hyperparameter setting, e.g. for debugging'
                              '(also do not run replications).'), required=False)
    parser.add_argument('--no_replications', action='store_true',
                        help='Don\'t run replication runs, only sweep.', required=False)
    args, unparsed = parser.parse_known_args()
    fill_platform_specific_default_args(args)
    main(args)
