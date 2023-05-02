import torch

from tart.representation.train import tart_train
from tart.representation.test import tart_test
from tart.inference.embed import tart_embed


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    # config file path
    config_file = "tart-config.json"

    # call train API
    tart_train(config_file)

    # call test API
    tart_test(config_file)

    # call embed API
    tart_embed(config_file)
