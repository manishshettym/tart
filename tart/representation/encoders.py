from typing import Callable
from transformers import RobertaTokenizer, RobertaModel
import torch

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = None
tokenizer = None
model = None


def codebert_encoder(x: str) -> torch.tensor:
    """feature encoder using CodeBert model"""
    global tokenizer, model
    
    tokens_ids = tokenizer.encode(x, truncation=True)
    tokens_tensor = torch.tensor(tokens_ids, device=my_device)

    with torch.no_grad():
        context_embeddings = model(tokens_tensor[None, :])[0]

    encoding = torch.mean(context_embeddings, dim=1)

    return encoding


def codebert_bpe_encoder(x: str) -> torch.tensor:
    """feature encoder using CodeBert BPE tokenizer"""
    global tokenizer, max_len
    
    encoded_input = tokenizer(
        x,
        return_tensors="pt",
        max_length=max_len,
        padding="max_length",
        truncation=True,
    )
    encoding = encoded_input["input_ids"]
    return encoding


def get_feature_encoder(encoder_name: str, **kwargs) -> Callable[[str], torch.tensor]:
    global tokenizer, model, max_len

    if encoder_name == "CodeBert":
        codebert_name = "microsoft/codebert-base"
        tokenizer = RobertaTokenizer.from_pretrained(codebert_name)
        model = RobertaModel.from_pretrained(codebert_name).to(my_device)
        model.eval()

        return codebert_encoder

    elif encoder_name == "CodeBertBPE":
        try:
            max_len = kwargs["max_len"]
        except KeyError:
            raise ValueError(
                "max_len is required for CodeBertBPE encoder; please provide it as a keyword argument."
            )

        codebert_name = "microsoft/codebert-base"
        tokenizer = RobertaTokenizer.from_pretrained(codebert_name)

        return codebert_bpe_encoder

    elif encoder_name == "Bert":
        raise NotImplementedError

    elif encoder_name == "BertBPE":
        raise NotImplementedError

    else:
        # raise that this encoder is not supported
        raise "Encoder {} is not a default encoder! You can implement it as a custom encoder.".format(
            encoder_name
        )
