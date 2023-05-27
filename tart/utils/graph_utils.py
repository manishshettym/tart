import json
import networkx as nx
from argparse import Namespace
from typing import Callable

import torch
from deepsnap.graph import Graph as DSGraph


def read_graph_from_json(args: Namespace, path: str) -> nx.Graph:
    """Read a graph from a json file

    Args:
        args (Namespace): tart configs
        path (str): path to the json file

    Returns:
        nx.Graph: networkx graph
    """
    with open(path, "r") as f:
        data = json.load(f)

    if "directed" in data and data["directed"]:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for id, node_data in enumerate(data["nodes"]):
        node_attrs = {}
        for attr, value in zip(args.node_feats, node_data):
            node_attrs[attr] = value

        G.add_node(id, **node_attrs)

    for edge in data["edges"]:
        edge_attrs = {}
        for attr, value in zip(args.edge_feats, edge[2]):
            edge_attrs[attr] = value

        G.add_edge(edge[0], edge[1], **edge_attrs)

    return G


def featurize_graph(args: Namespace, feat_encoder: Callable, g: nx.DiGraph, anchor=None) -> DSGraph:
    """Featurize a networkx graph into a DeepSnap graph
    >> all features are converted to torch.tensor and added to the `{feat}_t` key
    >> string features are converted to torch.tensor by the encoder model

    Args:
        args (Namespace): tart configs
        feat_encoder (Callable):  encoder function that converts string to torch.tensor
        g (nx.DiGraph): networkx graph
        anchor (_type_, optional): anchor node id. Defaults to None.

    Returns:
        DSGraph: DeepSnap graph
    """
    # make a copy of the nx graphview because
    # we will be pickling the graph and we cannot pickle graphviews
    g = g.copy()

    assert len(g.nodes) > 0, "Oops, graph has no nodes!"
    assert len(g.edges) > 0, "Oops, graph has no edges!"

    pagerank = nx.pagerank(g)
    clustering_coeff = nx.clustering(g)

    for v in g.nodes:
        # anchor is the default node feature if set
        if anchor is not None:
            g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

        for f, t in zip(args.node_feats, args.node_feat_types):
            # previously featurized this node
            if f + "_t" in g.nodes[v]:
                continue

            if t == "str":
                g.nodes[v][f + "_t"] = feat_encoder(g.nodes[v][f])
                g.nodes[v].pop(f)  # remove the original feature
            elif f == "node_degree":
                g.nodes[v][f + "_t"] = torch.tensor([g.degree(v)])
            elif f == "node_pagerank":
                g.nodes[v][f + "_t"] = torch.tensor([pagerank[v]])
            elif f == "node_cc":
                g.nodes[v][f + "_t"] = torch.tensor([clustering_coeff[v]])
            else:
                g.nodes[v][f + "_t"] = torch.tensor([g.nodes[v][f]])
                g.nodes[v].pop(f)  # remove the original feature

    for e in g.edges:
        for f, t in zip(args.edge_feats, args.edge_feat_types):
            # previously featurized this edge
            if f + "_t" in g.edges[e]:
                continue

            if t == "str":
                g.edges[e][f + "_t"] = feat_encoder(g.edges[e][f])
            else:
                g.edges[e][f + "_t"] = torch.tensor([g.edges[e][f]])

            # remove the original feature
            g.edges[e].pop(f)

    return DSGraph(g)
