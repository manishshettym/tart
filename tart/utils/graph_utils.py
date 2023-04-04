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


def featurize_graph(args, feat_encoder, g: nx.DiGraph, anchor=None) -> DSGraph:
    """Featurize a networkx graph into a DeepSnap graph
    >> all features are converted to torch.tensor and added to the `{feat}_t` key
    >> string features are converted to torch.tensor by the encoder model
    
    Args:
        g (nx.DiGraph): networkx graph
        feat_encoder (function): encoder function that converts string to torch.tensor
        anchor (int, optional): anchor node id. Defaults to None.
    """

    assert len(g.nodes) > 0, "Oops, graph has no nodes!"
    assert len(g.edges) > 0, "Oops, graph has no edges!"

    pagerank = nx.pagerank(g)
    clustering_coeff = nx.clustering(g)

    for v in g.nodes:
        # anchor is the default node feature if set
        if anchor is not None:
            g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

        for f, t in zip(args.node_feat, args.node_feat_type):
            
            # previously featurized this node
            if f + '_t' in g.nodes[v]:
                continue

            if t == 'str':
                g.nodes[v][f + "_t"] = feat_encoder(g.nodes[v][f])
            elif f == "node_degree":
                g.nodes[v][f + "_t"] = torch.tensor([g.degree(v)])
            elif f == "node_pagerank":
                g.nodes[v][f + "_t"] = torch.tensor([pagerank[v]])
            elif f == "node_cc":
                g.nodes[v][f + "_t"] = torch.tensor([clustering_coeff[v]])
            else:
                g.nodes[v][f + "_t"] = torch.tensor([g.nodes[v][f]])
            
            # remove the original feature
            g.nodes[v].pop(f)

    for e in g.edges:
        for f, t in zip(args.edge_feat, args.edge_feat_type):
            
            # previously featurized this edge
            if f + '_t' in g.edges[e]:
                continue

            if t == 'str':
                g.edges[e][f + "_t"] = feat_encoder(g.edges[e][f])
            else:
                g.edges[e][f + "_t"] = torch.tensor([g.edges[e][f]])
            
            # remove the original feature
            g.edges[e].pop(f)

    return DSGraph(g)
