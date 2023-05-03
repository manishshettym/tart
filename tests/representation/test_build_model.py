import json
import argparse

from tart.representation import models
from tart.utils.model_utils import build_model
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


def test_model_build():
    args = setup_test_args()
    model = build_model(models.SubgraphEmbedder, args)
    model_dict = model.state_dict()

    assert len(model_dict.keys()) == 62
