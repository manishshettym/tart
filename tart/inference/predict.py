# predict API for tart

import os
import json
import argparse
import inspect
from rich.console import Console
from rich.progress import (Progress, SpinnerColumn, TextColumn)

import torch
from deepsnap.batch import Batch

from tart.representation import config, models, dataset
from tart.utils.model_utils import build_model, get_device
from tart.utils.config_utils import validate_feat_encoder

console = Console()


def predict_neighs_batched(model, search_embs, query_emb):
    '''predict number of neighborhoods in which
    query graph (query_emb) is subgraph of search graphs (embs)
    
    Args:
        model (tart.representation.models.TART): trained TART model
        search_embs (list): list of embeddings of search graphs
        query_emb (torch.Tensor): embedding of query graph
    '''
    score = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("Searching for subgraphs..{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=len(search_embs))

        for emb_batch in search_embs:
            with torch.no_grad():
                predictions, _ = model.predictv2((
                                    emb_batch.to(get_device()), 
                                    query_emb))
                score += torch.sum(predictions).item()
        
    return score


def tart_predict(user_config_file, feat_encoder, query_graph, search_graphs, outcome="count_subgraphs"):
    console.print("[bright_green underline]Prediction API[/ bright_green underline]\n")
    parser = argparse.ArgumentParser()

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

    # validate user defined feature encoder
    if inspect.isfunction(feat_encoder):
        feat_encoder = validate_feat_encoder(feat_encoder, config_json)

    # build model
    model = build_model(models.SubgraphEmbedder, args)

    # TODO: design loading strategy for dataset:
    # notes: should we just assume embeddings are available apriori?
    # or should we load the dataset and compute embeddings on the fly?

    # load dataset
    # TODO: add support for loading dataset from disk
    # TODO: create embeddings if not available

    # ======= PREDICT =========
    if outcome == "count_subgraphs":
        # predict number of neighborhoods
        # score = predict_neighs_batched(model, search_graphs, query_graph)
        # console.print(f"Number of neighborhoods: {score}")
        raise NotImplementedError("count_subgraphs not implemented yet")
    elif outcome == "is_subgraph":
        # predict if query graph is subgraph of search graph
        # score = predict_is_subgraph_batched(model, search_graphs, query_graph)
        # console.print(f"Is subgraph: {score}")
        raise NotImplementedError("is_subgraph not implemented yet")
    else:
        raise ValueError("Invalid outcome. Please choose from: num_neighs, is_subgraph")
