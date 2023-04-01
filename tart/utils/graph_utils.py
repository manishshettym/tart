import networkx as nx
import json

import torch
from deepsnap.graph import Graph as DSGraph


def read_graph_from_json(path: str) -> nx.Graph:
    with open(path, 'r') as f:
        data = json.load(f)
    
    if 'directed' in data and data['directed']:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    for node in data['nodes']:
        G.add_node(node['id'], **node['data'])
    for edge in data['edges']:
        G.add_edge(edge['source'], edge['target'], **edge['data'])

    return G


def featurize_graph(args, g: nx.DiGraph, encoder_func, anchor=None) -> DSGraph:
    """Featurize a networkx graph into a DeepSnap graph
    >> all features are converted to torch.tensor
    >> string features are converted to torch.tensor by the encoder model
    
    Args:
        g (nx.DiGraph): networkx graph
        anchor (int, optional): anchor node id. Defaults to None.
    """

    assert len(g.nodes) > 0
    assert len(g.edges) > 0

    pagerank = nx.pagerank(g)
    clustering_coeff = nx.clustering(g)

    for v in g.nodes:
        # anchor is the default node feature if set
        if anchor is not None:
            g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

        for f, t in zip(args.node_feat, args.node_feat_type):
            if t == 'str':
                g.nodes[v][f] = encoder_func(g.nodes[v][f])
            elif f == "node_degree":
                g.nodes[v][f] = torch.tensor([g.degree(v)])
            elif f == "node_pagerank":
                g.nodes[v][f] = torch.tensor([pagerank[v]])
            elif f == "node_cc":
                g.nodes[v][f] = torch.tensor([clustering_coeff[v]])
            else:
                g.nodes[v][f] = torch.tensor([g.nodes[v][f]])

    for e in g.edges:
        for f, t in zip(args.edge_feat, args.edge_feat_type):
            if t == 'str':
                g.edges[e][f] = encoder_func(g.edges[e][f])
            else:
                g.edges[e][f] = torch.tensor([g.edges[e][f]])
    
    return DSGraph(g)
