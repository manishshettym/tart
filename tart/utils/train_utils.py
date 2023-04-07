from rich.progress import track, Progress, TextColumn, SpinnerColumn

import torch.multiprocessing as mp
from deepsnap.batch import Batch
from torch.utils.tensorboard import SummaryWriter


def init_logger(args):
    log_keys = ["conv_type", "n_layers", "hidden_dim",
                "margin", "dataset", "max_graph_size", "skip"]
    log_str = ".".join(["{}={}".format(k, v)
                        for k, v in sorted(vars(args).items())
                        if k in log_keys])
    return SummaryWriter(comment=log_str)


def start_workers(train_func, model, corpus, in_queue, out_queue, args):
    workers = []
    with Progress(
        SpinnerColumn(),
        TextColumn("Starting workers..{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)

        for _ in range(args.n_workers):
            worker = mp.Process(
                target=train_func,
                args=(args, model, corpus, in_queue, out_queue)
            )
            worker.start()
            workers.append(worker)
    
    return workers


def make_validation_set(dataloader):
    test_pts = []

    for batch in track(dataloader, total=len(dataloader), description="TestBatches"):
        pos_q, pos_t, neg_q, neg_t = zip(*batch)
        pos_q = Batch.from_data_list(pos_q)
        pos_t = Batch.from_data_list(pos_t)
        neg_q = Batch.from_data_list(neg_q)
        neg_t = Batch.from_data_list(neg_t)
        
        test_pts.append((pos_q, pos_t, neg_q, neg_t))
    
    return test_pts
