import os
import random
import glob
import os.path as osp
import networkx as nx
from argparse import Namespace

import torch
import numpy as np
from rich.progress import (
    track,
    Progress,
    TextColumn,
    MofNCompleteColumn,
    TimeRemainingColumn,
    BarColumn,
    TaskProgressColumn,
)
import scipy.stats as stats

from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch.utils.data import DataLoader
from typing import Optional, Callable, List, Tuple

from tart.utils.graph_utils import read_graph_from_json, featurize_graph


def my_collate(batch):
    """dummy collate fn to just return data as is"""
    return batch


class Corpus:
    def __init__(self, args: Namespace, feat_encoder: Callable, train: bool = True):
        """Initialize the corpus generator

        NOTE: new batch of graphs (positive and negative)
        are iteratively generated on the fly by sampling from the dataset.

        Args:
            args (Namespace): tart configs
            feat_encoder (Callable): function to encode graph features
            train (bool, optional): train/test corpus. Defaults to True.
        """
        if train:
            self.train_dataset = GraphDataset(
                root=f"{args.data_dir}/train",
                name=args.dataset,
                n_samples=args.n_train,
                feat_encoder=feat_encoder,
                args=args,
            )

        self.test_dataset = GraphDataset(
            root=f"{args.data_dir}/test",
            name=args.dataset,
            n_samples=args.n_test,
            feat_encoder=feat_encoder,
            args=args,
        )

    def gen_data_loader(self, batch_size: int, train: bool = True) -> DataLoader:
        """Initialize a data loader for the corpus

        Args:
            batch_size (int): batch size
            train (bool, optional): train/test corpus. Defaults to True.

        Returns:
            DataLoader: _description_
        """
        if train:
            return DataLoader(
                self.train_dataset,
                collate_fn=my_collate,
                batch_size=batch_size,
                sampler=None,
                shuffle=False,
            )
        else:
            return DataLoader(
                self.test_dataset,
                collate_fn=my_collate,
                batch_size=batch_size,
                sampler=None,
                shuffle=False,
            )


class GraphDataset(Dataset):
    def __init__(
        self,
        root: str,
        name: str,
        n_samples: int,
        feat_encoder: Callable,
        args: Namespace,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """Dataset of graphs

        Args:
            root (str): root directory of the dataset
            name (str): name of the dataset
            n_samples (int): number of samples to generate
            feat_encoder (Callable): function to encode graph str features
            args (Namespace): tart configs
            transform (Optional[Callable], optional): _description_. Defaults to None.
            pre_transform (Optional[Callable], optional): _description_. Defaults to None.
            pre_filter (Optional[Callable], optional): _description_. Defaults to None.
        """
        self.name = name
        self.graph_dir = osp.join(root, "graphs")

        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)

        self.n_samples = n_samples
        self.min_size = args.min_size
        self.max_size = args.max_size
        self.feat_encoder = feat_encoder

        # other args/configs
        self.args = args

        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        pass

    @property
    def num_node_labels(self) -> int:
        """Number of node labels

        Returns:
            int: number of node labels
        """
        return self.sizes["num_node_labels"]

    @property
    def num_node_attributes(self) -> int:
        """Number of node attributes

        Returns:
            int: number of node attributes
        """
        return self.sizes["num_node_attributes"]

    @property
    def num_edge_labels(self) -> int:
        """Number of edge labels

        Returns:
            int: number of edge labels
        """
        return self.sizes["num_edge_labels"]

    @property
    def num_edge_attributes(self) -> int:
        """Number of edge attributes

        Returns:
            int: number of edge attributes
        """
        return self.sizes["num_edge_attributes"]

    @property
    def raw_file_names(self) -> List[str]:
        """List of raw file names

        Returns:
            List[str]: list of raw file names
        """
        names = sorted(glob.glob(osp.join(self.raw_dir, "*.json")))
        names = [os.path.basename(file) for file in names]
        return names

    @property
    def processed_file_names(self) -> List[str]:
        """List of processed file names

        note: currently returns [], so that the dataset is
        reprocessed on every run.

        Returns:
            List[str]: list of processed file names
        """
        names = sorted(glob.glob(osp.join(self.processed_dir, "*.pt")))
        names = [os.path.basename(file) for file in names]
        # return names
        return []

    def pre_transform(self, data):
        return data

    def process(self) -> None:
        """process and store graphs as a Data object onto the disk"""
        count = 0
        graph_sizes = []

        # Preprocess each graph networkx json file to .pt file
        for raw_path in track(self.raw_paths, description="Processing graphs"):
            data = read_graph_from_json(self.args, raw_path)

            if data is None:
                continue

            if self.pre_filter is not None and self.prefilter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.graph_dir, f"data_{count}.pt"))
            graph_sizes.append(len(data))
            count += 1

        assert count > 1, f"No graphs found in the {self.raw_dir}"

        # Create randomly mutated examples and store in self.processed_dir
        self.random_samples_generator(graph_sizes, count)

    def random_samples_generator(self, graph_sizes: List[int], count: int) -> None:
        """Generate random samples of graphs

        Steps:
            1. choose a random size for graph
            2. choose a random target graph
            3. perform random bfs traversal in neighborhood = size
            4. choose a random but anchored query = positive e.g.
            5. repeat for negative e.g. but with different random graph for the query

        Args:
            graph_sizes (List[int]): distribution of graph sizes
            count (int): number of graphs
        """
        idx = 0

        with Progress(
            TextColumn("Sampling subgraphs"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TimeRemainingColumn(elapsed_when_finished=True),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("", total=self.n_samples)

            while idx < self.n_samples:
                size = random.randint(self.min_size + 1, self.max_size)
                graph, t = self.sample_neigh(graph_sizes, count, size)
                q = t[: random.randint(self.min_size, len(t) - 1)]

                anchor = list(graph.nodes)[0]
                pos_t_anchor = anchor
                pos_q_anchor = anchor
                pos_t, pos_q = graph.subgraph(t), graph.subgraph(q)

                size = random.randint(self.min_size + 1, self.max_size)
                graph_t, t = self.sample_neigh(graph_sizes, count, size)
                graph_q, q = self.sample_neigh(graph_sizes, count, random.randint(self.min_size, size - 1))

                neg_t_anchor = list(graph_t.nodes)[0]
                neg_q_anchor = list(graph_q.nodes)[0]
                neg_t, neg_q = graph_t.subgraph(t), graph_q.subgraph(q)

                # translate to DeepSnap Graph
                pos_t = featurize_graph(self.args, self.feat_encoder, pos_t, pos_t_anchor)
                pos_q = featurize_graph(self.args, self.feat_encoder, pos_q, pos_q_anchor)
                neg_t = featurize_graph(self.args, self.feat_encoder, neg_t, neg_t_anchor)
                neg_q = featurize_graph(self.args, self.feat_encoder, neg_q, neg_q_anchor)

                data = [pos_t, pos_q, neg_t, neg_q]
                torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
                idx += 1
                progress.update(task, advance=1)

    def sample_neigh(self, ps: List[int], count: int, size: int) -> Tuple[nx.Graph, List[int]]:
        """random bfs walk to find neighborhood graphs of a set size

        Args:
            ps (List[int]): distribution of graph sizes
            count (int): number of graphs
            size (int): size of subgraph to sample

        Returns:
            Tuple[nx.Graph, List[int]]: graph and list of nodes in subgraph
        """
        ps = np.array(ps, dtype=float)
        ps /= np.sum(ps)
        dist = stats.rv_discrete(values=(np.arange(count), ps))

        while True:
            idx = dist.rvs()
            graph = torch.load(osp.join(self.graph_dir, f"data_{idx}.pt"))

            start_node = random.choice(list(graph.nodes))
            neigh = [start_node]
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
            visited = set([start_node])

            while len(neigh) < size and frontier:
                new_node = random.choice(list(frontier))

                assert new_node not in neigh

                neigh.append(new_node)
                visited.add(new_node)
                frontier += list(graph.neighbors(new_node))
                frontier = [x for x in frontier if x not in visited]

            if len(neigh) == size:
                return graph, neigh

    def len(self) -> int:
        """Number of graphs in the dataset

        Returns:
            int: number of graphs in the dataset
        """
        return self.n_samples

    def get(self, idx: int) -> BaseData:
        """load a processed graph from disk.

        Args:
            idx (int): index of graph to load

        Raises:
            FileNotFoundError: if graph is not found

        Returns:
            BaseData: graph data object
        """
        try:
            data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        except FileNotFoundError:
            raise FileNotFoundError("data_{}.pt not found".format(idx))

        return data
