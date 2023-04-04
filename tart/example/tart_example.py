import torch
from transformers import RobertaTokenizer, RobertaModel

from tart.representation.train import tart_train
from tart.representation.test import tart_test

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

codebert_name = "microsoft/codebert-base"
CodeBertTokenizer = RobertaTokenizer.from_pretrained(codebert_name)
CodeBertModel = RobertaModel.from_pretrained(codebert_name).to(my_device)
CodeBertModel.eval()


def feat_encoder(x: str) -> torch.tensor:
    tokens_ids = CodeBertTokenizer.encode(
        x, truncation=True)
    
    tokens_tensor = torch.tensor(tokens_ids, device=my_device)
    
    with torch.no_grad():
        context_embeddings = CodeBertModel(
            tokens_tensor[None, :])[0]
    
    encoding = torch.mean(context_embeddings, dim=1)
    
    return encoding


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    
    # config file path
    config_file = "tart-config.json"

    # call train API
    # tart_train(config_file, feat_encoder)

    # call test API
    tart_test(config_file, feat_encoder)
