"""Configs for model and optimizer"""

from argparse import ArgumentParser, Namespace
from test_tube import HyperOptArgumentParser
from typing import Dict, List


def make_tunable(parser: HyperOptArgumentParser, tunable: List[str]) -> None:
    """make select arguments tunable

    Args:
        parser (ArgumentParser): argparse parser
        tunable (List[str]): list of arguments to make tunable
    """
    for arg in tunable:
        # remove arg from current parser
        parser._option_string_actions.pop("--" + arg)

        # add arg to test_tube parser
        if arg == "batch_size":
            parser.opt_list("--batch_size", type=int, help="Training batch size", tunable=True, default=64, options=[32, 64, 128])

        elif arg == "agg_type":
            parser.opt_list(
                "--agg_type", type=str, help="type of aggregation", tunable=True, default="GINE", options=["GINE", "GIN", "GCN"]
            )

        elif arg == "n_layers":
            parser.opt_list("--n_layers", type=int, help="Number of graph conv layers", tunable=True, default=7, options=[5, 7, 9, 11])

        elif arg == "hidden_dim":
            parser.opt_list("--hidden_dim", type=int, help="Training hidden size", tunable=True, default=64, options=[32, 64, 128])

        elif arg == "skip":
            parser.opt_list("--skip", type=str, help="skip connections", tunable=True, default="learnable", options=["all", "learnable"])

        else:
            raise ValueError("Argument {} is not tunable.".format(arg))


def build_model_configs(parser: ArgumentParser) -> None:
    """build model config arguments

    Args:
        parser (ArgumentParser): argparse parser
    """

    enc_args = parser.add_argument_group()

    enc_args.add_argument("--agg_type", type=str, help="type of aggregation/convolution")
    enc_args.add_argument("--batch_size", type=int, help="Training batch size")
    enc_args.add_argument("--n_layers", type=int, help="Number of graph conv layers")
    enc_args.add_argument("--hidden_dim", type=int, help="Training hidden size")
    enc_args.add_argument("--skip", type=str, help='"all" or "learnable"')
    enc_args.add_argument("--dropout", type=float, help="Dropout rate")
    enc_args.add_argument("--n_iters", type=int, help="Number of training iterations")
    enc_args.add_argument("--n_batches", type=int, help="Number of training minibatches")
    enc_args.add_argument("--margin", type=float, help="margin for loss")
    enc_args.add_argument("--dataset", type=str, help="Dataset name")
    enc_args.add_argument(
        "--data_dir",
        type=str,
        help="path to the root directory of the train/test sub-directories",
    )
    enc_args.add_argument("--test_set", type=str, help="test set filename")
    enc_args.add_argument("--eval_interval", type=int, help="how often to eval during training")
    enc_args.add_argument("--val_size", type=int, help="validation set size")
    enc_args.add_argument("--model_path", type=str, help="path to save/load model")
    enc_args.add_argument("--opt_scheduler", type=str, help="scheduler name")
    enc_args.add_argument(
        "--node_anchored",
        action="store_true",
        help="whether to use node anchoring in training",
    )
    enc_args.add_argument("--test", action="store_true")
    enc_args.add_argument("--n_workers", type=int)
    enc_args.add_argument("--tag", type=str, help="tag to identify the run")

    enc_args.set_defaults(
        agg_type="GINE",
        dataset="example",
        data_dir="../data/example",
        n_layers=7,
        batch_size=64,
        hidden_dim=64,
        skip="learnable",
        dropout=0.0,
        n_iters=5,
        n_batches=10000,
        opt="adam",
        opt_scheduler="none",
        opt_restart=100,
        weight_decay=0.0,
        lr=1e-4,
        margin=0.1,
        test_set="",
        eval_interval=1000,
        n_workers=4,
        model_path="ckpt/model.pt",
        tag="",
        val_size=4096,
        node_anchored=True,
    )


def build_optimizer_configs(parser: ArgumentParser) -> None:
    """build optimizer config arguments

    Args:
        parser (ArgumentParser): argparse parser
    """
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument("--opt", dest="opt", type=str, help="Type of optimizer")
    opt_parser.add_argument(
        "--opt-scheduler",
        dest="opt_scheduler",
        type=str,
        help="Type of optimizer scheduler. default: none",
    )
    opt_parser.add_argument(
        "--opt-restart",
        dest="opt_restart",
        type=int,
        help="Number of epochs before restart, default: 0",
    )
    opt_parser.add_argument(
        "--opt-decay-step",
        dest="opt_decay_step",
        type=int,
        help="Number of epochs before decay",
    )
    opt_parser.add_argument(
        "--opt-decay-rate",
        dest="opt_decay_rate",
        type=float,
        help="Learning rate decay ratio",
    )
    opt_parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    opt_parser.add_argument("--clip", dest="clip", type=float, help="Gradient clipping.")
    opt_parser.add_argument("--weight_decay", type=float, help="Optimizer weight decay.")

    opt_parser.set_defaults(opt="adam", opt_scheduler="none", opt_restart=100, weight_decay=0.0, lr=1e-4)


def build_feature_configs(parser: ArgumentParser) -> None:
    """build graph feature config arguments

    Args:
        parser (ArgumentParser): argparse parser
    """
    feat_parser = parser.add_argument_group()
    feat_parser.add_argument("--node_feats", nargs="+", help="node features to use in training")
    feat_parser.add_argument("--edge_feats", nargs="+", help="edge features to use in training")
    feat_parser.add_argument("--node_feat_dims", nargs="+", help="node feature dimension")
    feat_parser.add_argument("--edge_feat_dims", nargs="+", help="edge feature dimension")


def init_user_configs(args: Namespace, configs_json: Dict, tune: bool = False) -> Namespace:
    """initialize user defined configs

    Args:
        args (Namespace): argparse namespace
        configs_json (Dict): user defined configs

    Raises:
        ValueError: node_feats not provided in configs.json
        ValueError: edge_feats not provided in configs.json
        ValueError: node and edge feats names overlap

    Returns:
        Namespace: updated argparse namespace
    """

    # check if node_feats and edge_feats are provided
    if "node_feats" not in configs_json:
        raise ValueError("node_feats not provided in configs.json")
    if "edge_feats" not in configs_json:
        raise ValueError("edge_feats not provided in configs.json")

    # check if there is an overlap between node_feats and edge_feats
    feat_overlap = set(configs_json["node_feats"]) & set(configs_json["edge_feats"])
    if len(feat_overlap) > 0:
        raise ValueError("node and edge feats overlap on features: {}! Please rename them. ".format(feat_overlap))

    args.node_feats = configs_json["node_feats"] + [
        "node_degree",
        "node_pagerank",
        "node_cc",
    ]
    args.edge_feats = configs_json["edge_feats"]

    args.node_feat_dims = configs_json["node_feat_dims"] + [1, 1, 1]
    args.edge_feat_dims = configs_json["edge_feat_dims"]

    args.node_feat_types = configs_json["node_feat_types"] + ["int", "int", "int"]
    args.edge_feat_types = configs_json["edge_feat_types"]

    # other (assumes non list) features that were provided:
    for feat in set(configs_json) - set(
        [
            "node_feats",
            "edge_feats",
            "node_feat_dims",
            "edge_feat_dims",
            "node_feat_types",
            "edge_feat_types",
        ]
    ):
        if tune and feat in configs_json["tunable"]:
            raise ValueError(f"Feature {feat} is tunable. Please remove it from the config file.")

        setattr(args, feat, configs_json[feat])

    return args
