import os
import glob
import json
import os.path as osp
from argparse import Namespace
from typing import Callable, List
from rich.progress import track, Progress, TextColumn, SpinnerColumn
from rich.console import Console

import networkx as nx
import torch
import torch.nn as nn
import argparse
from deepsnap.batch import Batch
import torch.multiprocessing as mp

from tart.representation.encoders import get_feature_encoder
from tart.representation import config, models
from tart.utils.model_utils import build_model, get_device
from tart.utils.graph_utils import read_graph_from_json, featurize_graph

console = Console()

# ########## MULTI PROC ##########


def start_workers_process(in_queue: mp.Queue, out_queue: mp.Queue, args: Namespace) -> List[mp.Process]:
    """Starts worker processes for generating neighborhoods

    Args:
        in_queue (mp.Queue): multiprocessing queue for input
        out_queue (mp.Queue): multiprocessing queue for output
        args (Namespace): tart configs

    Returns:
        List[mp.Process]: list of worker processes
    """
    workers = []
    with Progress(
        SpinnerColumn(),
        TextColumn("Starting workers..{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)

        for _ in range(args.n_workers):
            worker = mp.Process(target=generate_neighborhoods, args=(args, in_queue, out_queue))
            worker.start()
            workers.append(worker)

    return workers


def start_workers_embed(model: nn.Module, in_queue: mp.Queue, out_queue: mp.Queue, args: Namespace) -> List[mp.Process]:
    """Starts worker processes for generating embeddings

    Args:
        model (nn.Module): tart model to embed graphs
        in_queue (mp.Queue): multiprocessing queue for input
        out_queue (mp.Queue): multiprocessing queue for output
        args (Namespace): tart configs

    Returns:
        List[mp.Process]: list of worker processes
    """
    workers = []
    with Progress(
        SpinnerColumn(),
        TextColumn("Starting workers..{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)

        for _ in range(args.n_workers):
            worker = mp.Process(target=generate_embeddings, args=(args, model, in_queue, out_queue))
            worker.start()
            workers.append(worker)

    return workers


# ########## UTILITIES ##########


def get_neighborhoods(args: Namespace, graph: nx.Graph, feat_encoder: Callable) -> List:
    """Returns a featurized (sampled) radial neighborhood for all nodes in a graph

    Args:
        args (Namespace): tart configs
        graph (nx.Graph): graph to find neighborhoods for
        feat_encoder (Callable): feature encoder for graph nodes

    Returns:
        List: list of featurized neighborhoods
    """
    neighs = []

    # find each node's neighbors via SSSP
    for j, node in enumerate(graph.nodes):
        shortest_paths = sorted(nx.single_source_shortest_path_length(graph, node, cutoff=args.emb_sssp_radius).items(), key=lambda x: x[1])
        neighbors = list(map(lambda x: x[0], shortest_paths))

        if args.emb_subg_sample_size != 0:
            # NOTE: random sampling of radius-hop neighbors,
            # results in nodes w/o any edges between them!!
            # Instead, sort neighbors by hops and chose top-K closest neighbors
            neighbors = neighbors[: args.emb_subg_sample_size]

        if len(neighbors) > 1:
            # NOTE: G.subgraph([nodes]) returns the subG induced on [nodes]
            # i.e., the subG containing the nodes in [nodes] and
            # edges between these nodes => in this case, a (sampled) radial n'hood
            neigh = graph.subgraph(neighbors)

            neigh = featurize_graph(args, feat_encoder, neigh, anchor=0)
            neighs.append(neigh)

    return neighs


# ########## PIPELINE FUNCTIONS ##########


def generate_embeddings(args: Namespace, model: nn.Module, in_queue: mp.Queue, out_queue: mp.Queue):
    """Generates embeddings for each node in the graph.
    NOTE: This function is called by each worker process.

    Args:
        arg (Namespace): tart configs
        model (nn.Module): tart model to generate embeddings
        in_queue (mp.Queue): multiprocessing queue for input
        out_queue (mp.Queue): multiprocessing queue for output
    """
    done = False
    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        # read only graphs of processed programs
        try:
            neighs = torch.load(osp.join(args.proc_dir, f"data_{idx}.pt"))
        except:
            out_queue.put(("complete"))
            continue

        with torch.no_grad():
            emb = model.encoder(Batch.from_data_list(neighs).to(get_device()))
            torch.save(emb, osp.join(args.emb_dir, f"emb_{idx}.pt"))

        out_queue.put(("complete"))


def generate_neighborhoods(args: Namespace, in_queue: mp.Queue, out_queue: mp.Queue):
    """Generates neighborhoods for each node in the graph.
    NOTE: This function is called by each worker process.

    Args:
        args (Namespace): tart configs
        in_queue (mp.Queue): multiprocessing queue for input
        out_queue (mp.Queue): multiprocessing queue for output
    """
    done = False
    feat_encoder = get_feature_encoder(args.feat_encoder)

    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        raw_path = osp.join(args.raw_dir, f"example_{idx}.json")
        graph = read_graph_from_json(args, raw_path)

        if graph is None:
            out_queue.put(("complete"))
            continue

        # save graph object for future apps like search
        torch.save(graph, osp.join(args.graph_dir, f"data_{idx}.pt"))

        # get neighborhoods of each node in the graph
        neighs = get_neighborhoods(args, graph, feat_encoder)
        torch.save(neighs, osp.join(args.proc_dir, f"data_{idx}.pt"))

        del graph
        del neighs

        out_queue.put(("complete"))


# ########## MAIN ##########


def embed_main(args: Namespace):
    """Pipeline to generate embeddings for a dataset of graphs

    Args:
        args (Namespace): tart configs
    """
    assert osp.exists(osp.dirname(args.raw_dir)), "raw_dir does not exist!"

    if not osp.exists(args.graph_dir):
        os.makedirs(args.graph_dir)

    if not osp.exists(args.proc_dir):
        os.makedirs(args.proc_dir)

    if not osp.exists(args.emb_dir):
        os.makedirs(args.emb_dir)

    raw_paths = sorted(glob.glob(osp.join(args.raw_dir, "*.json")))

    # ######### PHASE1: PROCESS GRAPHS #########

    # util: to rename .py files into a standard filename format
    # TODO: write to a txt file the mapping between index and original filename
    # TODO: should we write the renamed files to a tmp folder?
    for idx, p in enumerate(raw_paths):
        os.rename(p, osp.join(args.raw_dir, f"example_{idx}.json"))

    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers_process(in_queue, out_queue, args)

    for i in range(0, len(raw_paths)):
        in_queue.put(("idx", i))

    for _ in track(range(0, len(raw_paths)), description="Processing graphs"):
        msg = out_queue.get()

    for _ in range(args.n_workers):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()

    # ######### EMBED GRAPHS #########

    model = build_model(models.SubgraphEmbedder, args)
    model.share_memory()

    console.print(f"\n[bright_green]\[tart] [/bright_green] Moving model to device: [bright_blue]{get_device()}[/bright_blue]\n")
    model = model.to(get_device())
    model.eval()

    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers_embed(model, in_queue, out_queue, args)

    for i in range(0, len(raw_paths)):
        in_queue.put(("idx", i))

    for _ in track(range(0, len(raw_paths)), description="Embedding graphs"):
        msg = out_queue.get()

    for _ in range(args.n_workers):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()


def tart_embed(user_config_file: str):
    """tart's embed API

    Args:
        user_config_file (str): config file path
    """
    console.print("[bright_green underline]Embedding Search Space[/ bright_green underline]\n")
    parser = argparse.ArgumentParser()

    # reading user config from json file
    with open(user_config_file) as f:
        config_json = json.load(f)

    # build configs and their defaults
    config.build_optimizer_configs(parser)
    config.build_model_configs(parser)
    config.build_feature_configs(parser)

    args = parser.parse_args()

    # set user defined configs
    args = config.init_user_configs(args, config_json)

    # set default file paths for results
    root_dir = osp.join(args.data_dir, "embed")
    args.raw_dir = osp.join(root_dir, "raw")
    args.graph_dir = osp.join(root_dir, "graphs")
    args.proc_dir = osp.join(root_dir, "processed")
    args.emb_dir = osp.join(root_dir, "embs")

    embed_main(args)
