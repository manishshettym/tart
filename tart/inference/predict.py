# predict API for tart

import os.path as osp
import json
import glob
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import numpy as np
import torch
from deepsnap.batch import Batch

from tart.representation.encoders import get_feature_encoder
from tart.representation import config, models
from tart.inference.embed import get_neighborhoods
from tart.utils.model_utils import build_model, get_device
from tart.utils.graph_utils import read_graph_from_json, featurize_graph
from tart.utils.tart_utils import print_header


console = Console()


def search_space_sample(src_dir: str, k=None, seed=24):
    np.random.seed(seed)
    files = [f for f in sorted(glob.glob(osp.join(src_dir, "*.pt")))]
    if k is None:
        k = len(files)
    random_files = np.random.choice(files, min(len(files), k))
    random_index = [f.split("_")[-1][:-3] for f in random_files]

    return random_files, random_index


def read_embedding(args, idx):
    emb_path = f"emb_{idx}.pt"
    emb_path = osp.join(args.emb_dir, emb_path)
    return torch.load(emb_path, map_location=torch.device("cpu"))


def load_search_space(args, file_indices):
    """load embeddings of search space graphs into a list of batches"""
    embs, batch_embs = [], []
    count = 0

    for i, idx in enumerate(file_indices):
        batch_embs.append(read_embedding(args, idx))

        if i > 0 and i % args.batch_size == 0:
            embs.append(torch.cat(batch_embs, dim=0))
            count += len(batch_embs)
            batch_embs = []

    # add remaining embs as a batch
    if len(batch_embs) > 0:
        embs.append(torch.cat(batch_embs, dim=0))
        count += len(batch_embs)

    assert count == len(file_indices)

    return embs


def predict_neighs_batched(model, search_embs, query_emb):
    """(batched) predict number of neighborhoods in which
    query graph (query_emb) is subgraph of search graphs (embs)

    Args:
        model (tart.representation.models.TART): trained TART model
        search_embs (list): list of embeddings of search graphs
        query_emb (torch.Tensor): embedding of query graph
    """
    score = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("Searching for subgraphs..{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=len(search_embs))

        for emb_batch in search_embs:
            with torch.no_grad():
                predictions, _ = model.predictv2((emb_batch.to(get_device()), query_emb))
                score += torch.sum(predictions).item()

    return score


def tart_predict(
    user_config_file,
    query_json,
    search_space_path,
    search_sample=None,
    outcome="count_subgraphs",
):
    """predict API for tart

    Args:
        user_config_file (_type_): json file containing user defined configs
        query_json (_type_): json file containing query graph
        search_space_path (_type_): path to either
            (1) a directory containing embeddings of search space
            (2) a single json file containing a search graph
        outcome (str, optional): prediction task = {count_subgraphs, is_subgraph}. Defaults to "count_subgraphs".
    """
    print_header()
    console.print("[bright_green underline]Prediction API[/ bright_green underline]\n")
    parser = argparse.ArgumentParser()

    # is_subgraph expects search_space to be a single graph
    if outcome == "is_subgraph" and osp.isdir(search_space_path):
        raise ValueError("is_subgraph expects search_space_path to point to a single graph")

    # count_subgraph expects search_space to be a directory
    if outcome == "count_subgraphs" and not osp.isdir(search_space_path):
        raise ValueError("count_subgraphs expects search_space_path to point to a directory")

    # reading user config from json file
    with open(user_config_file) as f:
        config_json = json.load(f)

    # build configs and their defaults
    config.build_optimizer_configs(parser)
    config.build_model_configs(parser)
    config.build_feature_configs(parser)

    args = parser.parse_args()

    # set to test mode
    args.test = True

    # set user defined configs
    args = config.init_user_configs(args, config_json)

    # set feature encoder
    feat_encoder = get_feature_encoder(args.feat_encoder)

    # set search space embeddings directory
    args.emb_dir = search_space_path

    # featurize the query graph
    query_graph = read_graph_from_json(args, query_json)
    console.print(f"Query graph: {query_graph}")
    query_feat = featurize_graph(args, feat_encoder, query_graph, anchor=0)
    query_tensor = Batch.from_data_list([query_feat]).to(get_device())

    # build model
    model = build_model(models.SubgraphEmbedder, args)

    # embed query graph
    query_emb = model.encoder(query_tensor)

    # ======= PREDICT =========
    if outcome == "count_subgraphs":
        # load search space embeddings
        _, file_indices = search_space_sample(search_space_path, k=search_sample, seed=4)
        search_embs = load_search_space(args, file_indices)
        console.print(f"Search space: {len(file_indices)} graphs loaded.")

        # predict number of neighborhoods
        score = predict_neighs_batched(model, search_embs, query_emb)
        console.print(f"Number of subgraph (neighs): {score}")

        # NOTE: returns number of nodes that have subgraphs
        # rooted at the node that are isomorphic to query graph.
        # TODO: count number of graphs in which query was found as a subG.

    elif outcome == "is_subgraph":  # predict if query graph is subgraph of search graph
        # featurize the search graph (like in embed.py)
        search_graph = read_graph_from_json(args, query_json)
        console.print(f"Search graph: {search_graph}")
        search_neighs = get_neighborhoods(args, search_graph, feat_encoder)
        search_embs = model.encoder(Batch.from_data_list(search_neighs).to(get_device()))

        # predict if subgraph
        score = predict_neighs_batched(model, [search_embs], query_emb)
        if score > 0.0:
            console.print(f"Query graph is a subgraph of search graph (score = {score}).")
        else:
            console.print(f"Query graph is not a subgraph of search graph!")
    else:
        raise ValueError("Invalid outcome. Please choose from: count_subgraphs, is_subgraph")
