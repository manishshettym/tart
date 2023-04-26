import torch

from tart.inference.predict import tart_predict


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    # config file path
    config_file = "tart-config.json"

    # get query graph
    query_json = "../data/example/test/raw/g1.json"

    # locate search space embeddings
    search_embs_dir = "../data/example/embed/embs"

    # call predict API
    tart_predict(config_file, query_json, search_embs_dir,
                outcome="count_subgraphs")
