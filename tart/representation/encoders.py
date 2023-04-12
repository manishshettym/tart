import torch

def get_encoder(encoder_name: str):
    if encoder_name == "CodeBert":
        return CodeBertEncoder
    elif encoder_name == "BPE":
        return BPEEncoder
    else:
        raise ValueError("Invalid encoder name")


def CodeBertEncoder(x: str) -> torch.tensor:
    raise NotImplementedError


def BPEEncoder(x: str) -> torch.tensor:
    raise NotImplementedError
