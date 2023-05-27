import argparse
from typing import List, Tuple, Callable
from argparse import Namespace
from rich.progress import track, Progress, TextColumn, SpinnerColumn

import torch
import torch.multiprocessing as mp
from deepsnap.batch import Batch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from test_tube import HyperOptArgumentParser

from tart.representation.dataset import Corpus


def set_parser(tune):
    if tune:
        parser = HyperOptArgumentParser(strategy="grid_search")
    else:
        parser = argparse.ArgumentParser()
    return parser


def init_logger(args: Namespace) -> SummaryWriter:
    """Initialize tensorboard logger

    Args:
        args (Namespace): tart configs

    Returns:
        SummaryWriter: tensorboard logger
    """
    log_keys = [
        "conv_type",
        "n_layers",
        "hidden_dim",
        "margin",
        "dataset",
        "max_graph_size",
        "skip",
    ]
    log_str = ".".join(["{}={}".format(k, v) for k, v in sorted(vars(args).items()) if k in log_keys])
    return SummaryWriter(comment=log_str)


def start_workers(
    train_func: Callable,
    model: torch.nn.Module,
    corpus: Corpus,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    args: Namespace,
) -> List[mp.Process]:
    """Start workers for training

    Args:
        train_func (Callable): train function
        model (torch.nn.Module): tart model to train
        corpus (Corpus): dataset to train the model on
        in_queue (mp.Queue): mp queue for input
        out_queue (mp.Queue): mp queue for output
        args (Namespace): tart configs

    Returns:
        List[mp.Process]: list of workers
    """
    workers = []
    with Progress(
        SpinnerColumn(),
        TextColumn("Starting workers..{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)

        for _ in range(args.n_workers):
            worker = mp.Process(target=train_func, args=(args, model, corpus, in_queue, out_queue))
            worker.start()
            workers.append(worker)

    return workers


def make_validation_set(
    dataloader: DataLoader,
) -> List[Tuple[Batch, Batch, Batch, Batch]]:
    """Make validation set from dataloader

    Args:
        dataloader (DataLoader): dataloader for validation set

    Returns:
        List[Tuple[Batch, Batch, Batch, Batch]]:
            list of validation batches, each batch is a tuple of (pos_q, pos_t, neg_q, neg_t)
    """
    test_pts = []

    for batch in track(dataloader, total=len(dataloader), description="TestBatches"):
        pos_q, pos_t, neg_q, neg_t = zip(*batch)
        pos_q = Batch.from_data_list(pos_q)
        pos_t = Batch.from_data_list(pos_t)
        neg_q = Batch.from_data_list(neg_q)
        neg_t = Batch.from_data_list(neg_t)

        test_pts.append((pos_q, pos_t, neg_q, neg_t))

    return test_pts
