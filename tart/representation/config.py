"""Configs for model and optimizer"""

# TODO: Take a JSON as input when using the library
# Read the json and populate these args on the fly


def build_model_configs(parser):
    # Initialize encoder model configs
    enc_args = parser.add_argument_group()

    enc_args.add_argument(
        "--agg_type", type=str, help="type of aggregation/convolution"
    )
    enc_args.add_argument("--batch_size", type=int, help="Training batch size")
    enc_args.add_argument("--n_layers", type=int, help="Number of graph conv layers")
    enc_args.add_argument("--hidden_dim", type=int, help="Training hidden size")
    enc_args.add_argument("--skip", type=str, help='"all" or "last"')
    enc_args.add_argument("--dropout", type=float, help="Dropout rate")
    enc_args.add_argument("--n_iters", type=int, help="Number of training iterations")
    enc_args.add_argument(
        "--n_batches", type=int, help="Number of training minibatches"
    )
    enc_args.add_argument("--margin", type=float, help="margin for loss")
    enc_args.add_argument("--dataset", type=str, help="Dataset name")
    enc_args.add_argument("--data_dir", type=str, help="path to the root directory of the train/test sub-directories")
    enc_args.add_argument("--test_set", type=str, help="test set filename")
    enc_args.add_argument(
        "--eval_interval", type=int, help="how often to eval during training"
    )
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


def build_optimizer_configs(parser):
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
    opt_parser.add_argument(
        "--clip", dest="clip", type=float, help="Gradient clipping."
    )
    opt_parser.add_argument(
        "--weight_decay", type=float, help="Optimizer weight decay."
    )

    opt_parser.set_defaults(
        opt="adam", opt_scheduler="none", opt_restart=100, weight_decay=0.0, lr=1e-4
    )


def build_feature_configs(parser):
    feat_parser = parser.add_argument_group()
    feat_parser.add_argument(
        "--node_feat", nargs="+", help="node features to use in training")
    feat_parser.add_argument(
        "--edge_feat", nargs="+", help="edge features to use in training")
    feat_parser.add_argument(
        "--node_feat_dims", nargs="+", help="node feature dimension")
    feat_parser.add_argument(
        "--edge_feat_dims", nargs="+", help="edge feature dimension")
    
    feat_parser.set_defaults(
        node_feat=['node_degree', 'node_pagerank', 'node_cc'],
        edge_feat=['edge_type'],
        node_feat_dims=[1, 1, 1],
        edge_feat_dims=[1]
    )


def init_user_configs(args, configs_json):
    # add user defined features
    args = vars(args)

    # expected and required features
    # it will throw key error if not found
    args['node_feat'] = configs_json['node_feat'] + ['node_degree', 'node_pagerank', 'node_cc']
    args['edge_feat'] = configs_json['edge_feat']
    args['node_feat_dims'] = configs_json['node_feat_dims'] + [1, 1, 1]
    args['edge_feat_dims'] = configs_json['edge_feat_dims']

    # other (assumes non list) features that were provided:
    for feat in configs_json:
        args[feat] = configs_json[feat]
