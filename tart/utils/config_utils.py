from importlib import import_module
import importlib.util


def get_feat_encoder(config_json):
    assert "feat_encoder" in config_json, "feat_encoder not found in config"

    spec = importlib.util.spec_from_file_location(
        "user_funcs", config_json["feat_encoder"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    user_feat_encoder = module.feat_encoder

    # check if the function takes a string and returns a torch.tensor using a sample input
    assert (
        user_feat_encoder.__code__.co_argcount == 1
    ), "feat_encoder must take a single (str) argument"

    # check if the function takes a string and returns a torch.tensor
    # expected shape = config_json['node_feat_dim'][0] (assumes all features use the same encoder)
    exp_dim = config_json["node_feat_dims"][0]
    recv_dim = user_feat_encoder("test").shape
    assert recv_dim == (1, exp_dim), f"feat_encoder must return a torch.tensor of shape ({exp_dim},) got {recv_dim}"

    return user_feat_encoder
