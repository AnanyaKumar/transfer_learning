# Thanks to Michael Xie for these utilities.

import ast
from copy import deepcopy
import json
import logging
import importlib
import inspect
import torch
from torchvision import transforms

def save_json(save_path, save_dict):
    with open(save_path, 'w') as outfile:
        json.dump(save_dict, outfile)


def initialize_obj(classname, args_dict=None):
    module_name, class_name = classname.rsplit(".", 1)
    Class = getattr(importlib.import_module(module_name), class_name)
    if not inspect.isclass(Class):
        raise ValueError("Can only initialize classes, are you passing in a function?")
    # filter by argnames
    if args_dict is not None:
        argspec = inspect.getfullargspec(Class.__init__)
        argnames = argspec.args
        for k, v in args_dict.items():
            if k not in argnames:
                raise ValueError(f"{k}, {v} not found in {Class}")
        args_dict = {k: v for k, v in args_dict.items()
                     if k in argnames}
        defaults = argspec.defaults
        # add defaults
        if defaults is not None:
            for argname, default in zip(argnames[-len(defaults):], defaults):
                if argname not in args_dict:
                    args_dict[argname] = default
        class_instance = Class(**args_dict)
    else:
        class_instance = Class()
    return class_instance


def initialize(obj_config, update_args=None):
    classname = obj_config['classname']
    kwargs = obj_config.get('args')
    if kwargs is None:
        kwargs = {}
    if update_args is not None:
        kwargs.update(update_args)
    return initialize_obj(classname, kwargs)


def get_params(m):
    if hasattr(m, "trainable_params"):
        # "trainable_params" is custom module function
        return m.trainable_params()
    return m.parameters()


def init_transform(config_transforms):
    transform_list = [initialize(trans) for trans in config_transforms]
    return transforms.Compose(transform_list)


def init_dataset(dataset_config):
    transform = init_transform(dataset_config['transforms'])
    dataset_config_copy = deepcopy(dataset_config)
    dataset_kwargs = {'transform': transform}
    dataset = initialize(dataset_config_copy, dataset_kwargs)
    return dataset


def save_ckp(epoch, model, optimizer, scheduler, model_dir, chkpt_name):
    checkpoint_fpath = str(model_dir / chkpt_name)
    logging.info(f"Saving to checkpoint {checkpoint_fpath}")
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, checkpoint_fpath)


def load_ckp(checkpoint_fpath, model, optimizer=None, scheduler=None, reset_optimizer=False):
    logging.info(f"Loading from checkpoint {checkpoint_fpath}")
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    epoch = 0
    if not reset_optimizer:
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            epoch = int(checkpoint_fpath.split('epoch')[1].split('.')[0])
    return epoch


def setup_logging(log_dir, level=logging.DEBUG):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S', level=level,
        filename=log_dir+'/logs.txt', force=True)


def update_config(unparsed, config):
    # handle unknown arguments that change yaml config components
    # Embedded arguments e.g. loss.args must already exist in config to be updated.
    for unparsed_option in unparsed:
        option_name, val = unparsed_option.split('=')
        # get rid of --
        option_name = option_name[2:].strip()
        # handle nesting
        option_name_list = option_name.split('.')

        # interpret the string as int, float, string, bool, etc
        try:
            val = ast.literal_eval(val.strip())
        except Exception:
            # keep as string
            val = val.strip()

        curr_dict = config
        for k in option_name_list[:-1]:
            try:
                curr_dict = curr_dict.get(k)
            except:
                raise ValueError(f"Dynamic argparse failed: Keys: {option_name_list} Dict: {config}")
        curr_dict[option_name_list[-1]] = val
    return config


def to_device(obj, device):
    '''
    Wrapper around torch.Tensor.to that handles the case when obj is a
    container.
    Parameters
    ----------
    obj : Union[torch.Tensor, List[torch.Tensor], Dict[Any, Any]]
        Object to move to the specified device.
    device : str
        Describes device to move to.
    Returns
    -------
    Same type as obj.
    '''
    if isinstance(obj, list):
        return [item.to(device) for item in obj]
    elif isinstance(obj, dict):
        res = {}
        for key in obj:
            value = obj[key]
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            res[key] = value
        return res
    else:
        return obj.to(device)

