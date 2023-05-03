import json
import argparse

from tart.utils.graph_utils import read_graph_from_json, featurize_graph
from tart.representation.encoders import get_feature_encoder
import tart.representation.config as config


def setup_test_args(user_config_file="./tests/data/sample_config.json"):
    parser = argparse.ArgumentParser()

    # add dummy argument for pytest
    parser.add_argument("-s", dest="s", type=str, help="pytest folder")

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

    return args


def test_read_graph():
    args = setup_test_args()
    sample_graph_json = "./tests/data/sample_graph.json"

    graph = read_graph_from_json(args, sample_graph_json)
    feat_encoder = get_feature_encoder(args.feat_encoder)

    assert feat_encoder is not None

    dsgraph = featurize_graph(args, feat_encoder, graph, anchor=None)
    assert dsgraph.num_nodes == 3
    assert dsgraph.num_edges == 3
