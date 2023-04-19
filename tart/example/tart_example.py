import torch

from tart.representation.encoders import get_feature_encoder
from tart.representation.train import tart_train
from tart.representation.test import tart_test

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    # config file path
    config_file = "tart-config.json"

    # feature encoder
    feat_encoder = get_feature_encoder("CodeBert")

    # call train API
    tart_train(config_file, feat_encoder)

    # call test API
    tart_test(config_file, feat_encoder)
