import torch
from transformers import RobertaTokenizer, RobertaModel
from tart.utils.train_utils import get_device


# EXAMPLE
codebert_name = "microsoft/codebert-base"
CodeBertTokenizer = RobertaTokenizer.from_pretrained(codebert_name)
CodeBertModel = RobertaModel.from_pretrained(codebert_name).to(get_device())
CodeBertModel.eval()


def my_encoder(x: str) -> torch.tensor:
    tokens_ids = CodeBertTokenizer.encode(
        x, truncation=True)
    
    tokens_tensor = torch.tensor(tokens_ids, device=get_device())
    
    with torch.no_grad():
        context_embeddings = CodeBertModel(
            tokens_tensor[None, :])[0]
    
    encoding = torch.mean(context_embeddings, dim=1)
    
    return encoding
