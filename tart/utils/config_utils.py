def validate_feat_encoder(user_feat_encoder, config_json):

    str_feat_idx = config_json["node_feat_types"].index("str")

    # check if the function takes a string and returns a torch.tensor using a sample input
    assert (
        user_feat_encoder.__code__.co_argcount == 1
    ), "feat_encoder must take a single (str) argument"

    # check if the function takes a string and returns a torch.tensor
    # expected shape = config_json['node_feat_dim'][0] (assumes all features use the same encoder)
    exp_dim = config_json["node_feat_dims"][str_feat_idx]
    recv_dim = user_feat_encoder("test").shape
    assert recv_dim == (
        1,
        exp_dim,
    ), f"feat_encoder must return a torch.tensor of shape ({exp_dim},) got {recv_dim}"

    return user_feat_encoder
