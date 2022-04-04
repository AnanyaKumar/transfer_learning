import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def spawn_processes(worker_fn, args, mpargs):
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(worker_fn, nprocs=ngpus_per_node, args=(ngpus_per_node, *mpargs))
    else:
        # Simply call main_worker function
        worker_fn(*(args.gpu, ngpus_per_node, *mpargs))

def init_proc_group(args, ngpus_per_node):
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + args.gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

def init_data_parallel(args, model, ngpus_per_node):
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        else:
            model = model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model

def calculate_coeff(covariances):
    coeff = covariances ** (-2) # every element should have value bounded by sigma min/max
    coeff = torch.sqrt(torch.sum(coeff, [1, 2, 3]))
    return 1 / coeff

def grad_norm_for_loss(model, loss, to_filter=None):
    if to_filter is not None:
        params = [p for name, p in model.named_parameters() if p.requires_grad and to_filter in name]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        return 0
    model_grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        only_inputs=True,
        allow_unused=True)
    # filter out grads from unused parameters
    model_grads = [grad for grad in model_grads if grad is not None]
    if len(model_grads) == 0:
        return 0
    return torch.norm(torch.stack([torch.norm(m.detach()) for m in model_grads])).item()
