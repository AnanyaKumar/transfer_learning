import subprocess
import shlex
import argparse
from copy import deepcopy


def run_sbatch(cmd, job_name, args, exclude=None,
               nodes=1, gres='gpu:1', cpus_per_task=2, mem='16G'):
    output_path = args.output_dir + '/' + job_name
    sbatch_script_path = args.scripts_dir + '/' + args.sbatch_script_name 
    slurm_cmd = f'sbatch --partition={args.partition} --job-name={job_name} --output={output_path} ' +\
                f'--mail-type=END,FAIL --mail-user={args.mail_user} --nodes={nodes} ' +\
                f'--gres={gres} --cpus-per-task={cpus_per_task} --mem={mem} ' +\
                f'{sbatch_script_path} '
    slurm_cmd += f'"{cmd}"'
    print(slurm_cmd)
    subprocess.run(shlex.split(slurm_cmd))


def run_codalab(cmd, args):
    raise NotImplementedError


def run_job(cmd, job_name, args):
    if args.codalab:
        run_codalab(cmd, args)
    else:
        run_sbatch(cmd, job_name, args)


def format_value(v):
    if type(v) == list:
        if type(v[0]) == list:
            raise ValueError('We only support 1D lists.')
        return ' ' + ' '.join([str(e) for e in v])
    return '=' + str(v)


def get_python_cmd(code_path, python_path='python', kwargs=None):
    if kwargs is not None:
        opts = ''.join([f"--{k}{format_value(v)} " for k, v in kwargs.items()])
        # opts += ''.join([f"--{k} " for k, v in kwargs.items() if isinstance(v, bool) and v and '.' not in k])
    else:
        opts = ''
    python_cmd = python_path + ' ' + code_path + ' '
    python_cmd += opts
    return python_cmd


def get_baseline_experiment_cmd(config_path, run_name, group_name, project_name, kwargs, args):
    # config_path is relative to config_dir
    kwargs = deepcopy(kwargs)
    kwargs['config'] = args.config_dir + '/' + config_path
    if not(args.codalab):
        kwargs['log_dir'] = args.log_dir + '/' + group_name + '/' + run_name
        kwargs['tmp_par_ckp_dir'] = args.tmp_dir + '/' + run_name
    else:
        kwargs['log_dir'] = args.log_dir
    kwargs['root_prefix'] = args.data_dir
    kwargs['project_name'] = project_name
    kwargs['group_name'] = group_name
    kwargs['run_name'] = run_name
    code_path = args.code_dir + '/' + 'baseline_train.py'
    assert 'root_prefix' in kwargs and 'log_dir' in kwargs and 'config' in kwargs
    return get_python_cmd(code_path=code_path, python_path=args.python_path, kwargs=kwargs)


def config_run(args, kwargs, config_path, run_name, group_name, project_name,
               dataset_copy_cmd=None):
    cmd = get_baseline_experiment_cmd(
        config_path=config_path, run_name=run_name, group_name=group_name,
        project_name=project_name, kwargs=kwargs, args=args)
    if dataset_copy_cmd is not None and not args.codalab:
        cmd = dataset_copy_cmd + ' && ' + cmd
    job_name = group_name + '_' + run_name
    run_job(cmd, job_name, args)


def linprobe_run(args, rel_config_path, job_name, train_index, test_indices, num_reg_values=50,
                 extract_features_kwargs={}):
    feature_save_path = args.log_dir + '/extract_features/' + job_name + '.pkl'
    probe_stats_save_path = args.log_dir + '/linear_probe_sk/' + job_name + '.tsv'
    code_path = args.code_dir + '/extract_features.py'
    kwargs = deepcopy(extract_features_kwargs)
    kwargs['config'] = args.config_dir + '/' + rel_config_path
    kwargs['save_path'] = feature_save_path
    extract_cmd = get_python_cmd(code_path=code_path, python_path=args.python_path, kwargs=kwargs)
    code_path = args.code_dir + '/log_reg_sk.py'
    kwargs = {
        'load_path': feature_save_path, 'save_path': probe_stats_save_path,
        'train_index': train_index, 'test_indices': test_indices, 'num_reg_values': num_reg_values
    }
    log_reg_cmd =  get_python_cmd(code_path=code_path, python_path=args.python_path, kwargs=kwargs)
    cmd = extract_cmd + ' && ' + log_reg_cmd 
    run_job(cmd, job_name, args)

# FMOW.

def fmow_run(args, lr, seed, group_suffix, pretrain='moco'):
    run_name = str(lr) + '_' + str(seed)
    kwargs = {'optimizer.args.lr': lr, 'seed': seed}
    copy_cmd = f'source {args.scripts_dir}/copy_local.sh /u/scr/nlp/wilds/data/fmow_v1.1.tar.gz wilds/data'
    config_path = 'wilds/fmow_' + pretrain + '_ft_noaugment.yaml'
    group_name = 'fmow_' + pretrain + '_ft_noaugment_' + group_suffix
    config_run(
        args=args, kwargs=kwargs, config_path=config_path,
        run_name=run_name, group_name=group_name, project_name='fmow',
        dataset_copy_cmd=copy_cmd)


def fmow_moco_ft_noaugment_sweep(args):
    for lr in [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03]:
        fmow_run(args=args, lr=lr, seed=args.seed, group_suffix='sweep')


def fmow_mocotp_ft_noaugment_sweep(args):
    for lr in [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03]:
        fmow_run(args=args, lr=lr, seed=args.seed, group_suffix='sweep', pretrain='mocotp')


def fmow_moco_ft_noaugment_replication(args):
    for i in range(1,6):
        fmow_run(args=args, lr=None, seed=args.seed+i, group_suffix='sweep')


# Breeds.


def e30_moco_ft_augment_sweep_run(args, lr):
    run_name = str(lr) + '_' + str(args.seed)
    kwargs = {'optimizer.args.lr': lr, 'seed': args.seed}
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/e30_moco_ft_augment.yaml',
        run_name=run_name, group_name='e30_moco_ft_augment_sweep', project_name='entity30',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def e30_moco_ft_augment_sweep(args):
    for lr in [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03]:
        e30_moco_ft_augment_sweep_run(args, lr=lr)


def e30_moco_ft_augment_replication_run(args, seed):
    lr = 3e-4  # We found this to work best in the sweep.
    run_name = str(lr) + '_' + str(seed)
    kwargs = {'optimizer.args.lr': lr, 'seed': seed}
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/e30_moco_ft_augment.yaml',
        run_name=run_name, group_name='e30_moco_ft_augment_replication', project_name='entity30',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')

 
def e30_moco_ft_augment_replication(args):
    for seed_add in range(1,6):
        e30_moco_ft_augment_replication_run(args, seed=args.seed + seed_add)


def l17_moco_ft_augment_sweep_run(args, lr):
    run_name = str(lr) + '_' + str(args.seed)
    kwargs = {'optimizer.args.lr': lr, 'seed': args.seed}
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/l17_moco_ft_augment.yaml',
        run_name=run_name, group_name='l17_moco_ft_augment_sweep', project_name='living17',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def l17_moco_ft_augment_sweep(args):
    for lr in [1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03]:
        l17_moco_ft_augment_sweep_run(args, lr=lr)


def l17_moco_ft_augment_replication_run(args, seed):
    lr = 1e-4  # We found this to work best in the sweep.
    run_name = str(lr) + '_' + str(seed)
    kwargs = {'optimizer.args.lr': lr, 'seed': seed}
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/l17_moco_ft_augment.yaml',
        run_name=run_name, group_name='l17_moco_ft_augment_replication', project_name='living17',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def l17_moco_ft_augment_replication(args):
    for seed_add in range(5):
        l17_moco_ft_augment_replication_run(args, seed=args.seed + seed_add)


def landcover_run(args, lr, seed, group_suffix, tune_type='ft', split='to_africa', epochs=25):
    run_name = str(lr) + '_' + str(seed)
    kwargs = {'optimizer.args.lr': lr, 'seed': seed, 'epochs': epochs}
    if tune_type == 'probe':
        kwargs['linear_probe'] = True
        kwargs['use_net_val_mode'] = True
        config_path = 'innout/landcover_' + split + '_ft.yaml'
    else:
        config_path = 'innout/landcover_' + split + '_' + tune_type + '.yaml'
    group_name = 'landcover_' + split + '_' + tune_type + '_' + group_suffix
    config_run(
        args=args, kwargs=kwargs, config_path=config_path,
        run_name=run_name, group_name=group_name, project_name='landcover')


def landcover_to_africa_ft_sweep(args):
    for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        landcover_run(
            args, lr=lr, seed=args.seed, group_suffix='sweep', tune_type='ft',
            split='to_africa')


def landcover_to_africa_ft_replication(args):
    for i in range(5):
        landcover_run(
            args, lr=0.01, seed=args.seed+i, group_suffix='replication',
            tune_type='ft', split='to_africa')


def landcover_to_africa_scratch_sweep(args):
    for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        landcover_run(
            args, lr=lr, seed=args.seed, group_suffix='sweep', tune_type='scratch',
            split='to_africa', epochs=50)


def landcover_to_africa_scratch_replication(args):
    for i in range(5):
        landcover_run(
            args, lr=0.03, seed=args.seed+i, group_suffix='replication', tune_type='scratch',
            split='to_africa', epochs=50)


def landcover_to_africa_probe_sweep(args):
    for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        landcover_run(
            args, lr=lr, seed=args.seed, group_suffix='sweep', tune_type='probe',
            split='to_africa', epochs=50)


def landcover_to_africa_probe_replication(args):
    for i in range(5):
        landcover_run(
            args, lr=0.03, seed=args.seed+i, group_suffix='replication', tune_type='probe',
            split='to_africa', epochs=50)


def landcover_from_africa_ft_sweep(args):
    for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        landcover_run(
            args, lr=lr, seed=args.seed, group_suffix='sweep', tune_type='ft',
            split='from_africa')


def landcover_from_africa_ft_replication(args):
    for i in range(5):
        landcover_run(
            args, lr=0.03, seed=args.seed+i, group_suffix='replication', tune_type='ft',
            split='from_africa')


def landcover_from_africa_scratch_sweep(args):
    for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        landcover_run(
            args, lr=lr, seed=args.seed, group_suffix='sweep', tune_type='scratch',
            split='from_africa', epochs=50)


def landcover_from_africa_scratch_replication(args):
    for i in range(5):
        landcover_run(
            args, lr=0.01, seed=args.seed+i, group_suffix='replication', tune_type='scratch',
            split='from_africa', epochs=50)


def landcover_from_africa_probe_sweep(args):
    for lr in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
        landcover_run(
            args, lr=lr, seed=args.seed, group_suffix='sweep', tune_type='probe',
            split='from_africa', epochs=50)


def landcover_from_africa_probe_replication(args):
    for i in range(5):
        landcover_run(
            args, lr=0.1, seed=args.seed+i, group_suffix='replication', tune_type='probe',
            split='from_africa', epochs=50)




# Tune batchnorm parameters (+ linear classification layer).

def e30_moco_batchnorm_valmode_ft_sweep_run(args, lr):
    run_name = str(lr) + '_' + str(args.seed)
    kwargs = {
        'optimizer.args.lr': lr, 'seed': args.seed,
        'batchnorm_ft': True, 'use_net_val_mode': True
    }
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/e30_moco_ft_augment.yaml',
        run_name=run_name, group_name='e30_moco_batchnorm_valmode_ft_sweep', project_name='entity30',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def e30_moco_batchnorm_valmode_ft_sweep(args):
    for lr in [1e-4, 3e-4, 0.001, 0.003]:
        e30_moco_batchnorm_valmode_ft_sweep_run(args, lr=lr)


def l17_moco_batchnorm_valmode_ft_sweep_run(args, lr):
    run_name = str(lr) + '_' + str(args.seed)
    kwargs = {
        'optimizer.args.lr': lr, 'seed': args.seed,
        'batchnorm_ft': True, 'use_net_val_mode': True
    }
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/l17_moco_ft_augment.yaml',
        run_name=run_name, group_name='l17_moco_batchnorm_valmode_ft_sweep', project_name='living17',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def l17_moco_batchnorm_valmode_ft_sweep(args):
    for lr in [0.0003, 0.0001, 3e-4, 0.001]:
        l17_moco_batchnorm_valmode_ft_sweep_run(args, lr=lr)


# Use higher learning rate for top linear layer.

def e30_moco_lin_higher_lr_ft_sweep_run(args, lr, linear_layer_lr_multiplier=10):
    run_name = str(lr) + '_' + str(args.seed)
    kwargs = {
        'optimizer.args.lr': lr, 'seed': args.seed,
        'linear_layer_lr_multiplier': linear_layer_lr_multiplier
    }
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/e30_moco_ft_augment.yaml',
        run_name=run_name, group_name='e30_moco_lin_higher_lr_ft_sweep', project_name='entity30',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def e30_moco_lin_higher_lr_ft_sweep(args):
    for lr in [1e-4, 3e-4, 0.001, 0.003]:
        e30_moco_lin_higher_lr_ft_sweep_run(args, lr=lr)


def l17_moco_lin_higher_lr_ft_sweep_run(args, lr, linear_layer_lr_multiplier=10):
    run_name = str(lr) + '_' + str(args.seed)
    kwargs = {
        'optimizer.args.lr': lr, 'seed': args.seed,
        'linear_layer_lr_multiplier': linear_layer_lr_multiplier
    }
    config_run(
        args=args, kwargs=kwargs, config_path='breeds/l17_moco_ft_augment.yaml',
        run_name=run_name, group_name='l17_moco_lin_higher_lr_ft_sweep', project_name='living17',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def l17_moco_lin_higher_lr_ft_sweep(args):
    for lr in [3e-5, 1e-4, 3e-4, 0.001]:
        l17_moco_lin_higher_lr_ft_sweep_run(args, lr=lr)


# l2-sp runs.

def l17_moco_l2sp_augment_sweep_run(args, lr):
    # TODO: fix this
    ft_config_run(
        args, seed=args.seed, lr=lr, config_path='breeds/l17_moco_ft_augment.yaml',
        group_name='l17_moco_ft_augment_sweep', project_name='living17',
        dataset_copy_cmd=f'source {args.scripts_dir}/copy_dataset.sh imagenet')


def l17_moco_l2sp_augment_sweep(args):
    # TODO: fix this
    for lr in [3e-5, 1e-4, 3e-4, 0.001]:
        l17_moco_l2sp_augment_sweep_run(args, lr=lr)


# Linear probe runs.

def landcover_to_africa_linprobe(args):
    linprobe_run(
        args, rel_config_path='extract_features/landcover_cnn.yaml',
        job_name='landcover_to_africa_linprobe',
        train_index=0, test_indices=[0,1,2,3], num_reg_values=100)


def landcover_from_africa_linprobe(args):
    linprobe_run(
        args, rel_config_path='extract_features/landcover_cnn.yaml',
        job_name='landcover_from_africa_linprobe',
        train_index=2, test_indices=[0,1,2,3], num_reg_values=100)


def fmow_moco_linprobe(args):
    linprobe_run(
        args, rel_config_path='extract_features/fmow_moco.yaml',
        job_name='fmow_moco_linprobe',
        train_index=0, test_indices=list(range(11)), num_reg_values=40)


def fmow_mocotp_linprobe(args):
    extract_features_kwargs = {
        'config_paths': args.log_dir+'/fmow_mocotp_ft_noaugment_sweep/0.01_0/config.json',
        'checkpoint_paths': args.log_dir+'/fmow_mocotp_ft_noaugment_sweep/0.01_0/checkpoints/ckp_0',
    }
    linprobe_run(
        args, rel_config_path='extract_features/fmow_moco.yaml',
        job_name='fmow_mocotp_linprobe',
        train_index=0, test_indices=list(range(11)), num_reg_values=40,
        extract_features_kwargs=extract_features_kwargs)



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
        'fmow_moco_ft_noaugment_sweep': fmow_moco_ft_noaugment_sweep,
        'fmow_moco_ft_noaugment_replication': fmow_moco_ft_noaugment_replication,
        'spray_domainnet_jags': spray_domainnet_jags,
        'e30_moco_ft_augment_sweep': e30_moco_ft_augment_sweep,
        'e30_moco_ft_augment_replication': e30_moco_ft_augment_replication,
        'l17_moco_ft_augment_sweep': l17_moco_ft_augment_sweep,
        'l17_moco_ft_augment_replication': l17_moco_ft_augment_replication,
        'l17_moco_l2sp_augment_sweep': l17_moco_l2sp_augment_sweep,
        'e30_moco_batchnorm_valmode_ft_sweep': e30_moco_batchnorm_valmode_ft_sweep,
        'l17_moco_batchnorm_valmode_ft_sweep': l17_moco_batchnorm_valmode_ft_sweep,
        'e30_moco_lin_higher_lr_ft_sweep': e30_moco_lin_higher_lr_ft_sweep,
        'l17_moco_lin_higher_lr_ft_sweep': l17_moco_lin_higher_lr_ft_sweep,
        'landcover_to_africa_linprobe': landcover_to_africa_linprobe,
        'landcover_to_africa_ft_sweep': landcover_to_africa_ft_sweep,
        'landcover_to_africa_ft_replication': landcover_to_africa_ft_replication,
        'landcover_to_africa_scratch_sweep': landcover_to_africa_scratch_sweep,
        'landcover_to_africa_scratch_replication': landcover_to_africa_scratch_replication,
        'landcover_to_africa_probe_sweep': landcover_to_africa_probe_sweep,
        'landcover_to_africa_probe_replication': landcover_to_africa_probe_replication,
        'landcover_from_africa_linprobe': landcover_from_africa_linprobe,
        'landcover_from_africa_ft_sweep': landcover_from_africa_ft_sweep,
        'landcover_from_africa_ft_replication': landcover_from_africa_ft_replication,
        'landcover_from_africa_scratch_sweep': landcover_from_africa_scratch_sweep,
        'landcover_from_africa_scratch_replication': landcover_from_africa_scratch_replication,
        'landcover_from_africa_probe_sweep': landcover_from_africa_probe_sweep,
        'landcover_from_africa_probe_replication': landcover_from_africa_probe_replication,
        'fmow_moco_linprobe': fmow_moco_linprobe,
        'fmow_mocotp_linprobe': fmow_mocotp_linprobe,
        'fmow_mocotp_ft_noaugment_sweep': fmow_mocotp_ft_noaugment_sweep,
    }
    if args.experiment in experiment_to_fns:
        experiment_to_fns[args.experiment](args)
    else:
        raise ValueError(f'Experiment {args.experiment} does not exist.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run celeba experiments.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment to run.')
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help='Base seed, we may add to this seed for different runs.')
    parser.add_argument('--codalab', action='store_true', help='run on CodaLab not slurm')
    parser.add_argument('--partition', type=str, required=False, default='jag-standard',
                        help='(Slurm only) What priority to use.')
    parser.add_argument('--mail_user', type=str, required=False,
                        help='(Slurm only) Email if slurm job fails.', default=None)
    # Locations of folders and files.
    parser.add_argument('--data_dir', type=str, required=False, default='/scr/biggest/',
                        help='Dir (root_prefix in config) where the data is stored')
    parser.add_argument('--scripts_dir', type=str, required=False, default='scripts/',
                        help='Path to dir where scripts are stored.')
    parser.add_argument('--output_dir', type=str, required=False, default='slurm_outputs/',
                        help='Path to dir to store stdout for experiment.')
    parser.add_argument('--log_dir', type=str, required=False, default='logs/',
                        help='Path to dir where we save logs and checkpoints.')
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
    main(args)

