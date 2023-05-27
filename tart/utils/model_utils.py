from argparse import Namespace
from typing import Tuple, Iterator

import torch
import torch.optim as optim

DEVICE_CACHE = None


def get_device() -> torch.device:
    """get device (cpu or gpu)

    Returns:
        torch.device: available device
    """
    # get device (cpu or gpu)
    global DEVICE_CACHE

    if DEVICE_CACHE is None:
        if torch.cuda.is_available():
            # print("GPU is available!!!")
            DEVICE_CACHE = torch.device("cuda")
        else:
            DEVICE_CACHE = torch.device("cpu")

    return DEVICE_CACHE


def get_torch_tensor_type() -> torch.FloatTensor or torch.cuda.FloatTensor:
    """get torch tensor type (cpu or gpu)

    Returns:
        torch.dtype: correct torch tensor type
    """
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor


def build_model(model_type: torch.nn.Module, args: Namespace) -> torch.nn.Module:
    """build the user specified model

    Args:
        model_type (torch.nn.Module): model class
        args (Namespace): user defined configs

    Returns:
        torch.nn.Module: built model
    """
    # build model
    model = model_type(1, args.hidden_dim, args)
    model.to(get_device())

    if args.test and args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=get_device()))

    return model


def build_optimizer(args: Namespace, params: Iterator) -> Tuple:
    """build optimizer and scheduler

    Args:
        args (Namespace): user defined configs
        params (Iterator): model parameters

    Returns:
        Tuple[optim.lr_scheduler.LRScheduler, optim.Optimizer]: _description_
    """
    # build optimizer
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)

    if args.opt == "adam":
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.opt_scheduler == "none":
        return None, optimizer
    elif args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)

    return scheduler, optimizer
