import os
import json
import argparse
from argparse import Namespace
from typing import List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import average_precision_score
from deepsnap.batch import Batch

from tart.representation.encoders import get_feature_encoder
from tart.representation import config, models, dataset
from tart.utils.model_utils import build_model, get_device
from tart.utils.config_utils import validate_feat_encoder

console = Console()


def precision(pred: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate precision for predictions.

    Args:
        pred (torch.Tensor): tensor of predicted labels
        labels (torch.Tensor): tensor of true labels

    Returns:
        float: average precision
    """
    if torch.sum(pred) > 0:
        return torch.sum(pred * labels).item() / torch.sum(pred).item()
    else:
        return float("NaN")


def recall(pred: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate recall for predictions.

    Args:
        pred (torch.Tensor): tensor of predicted labels
        labels (torch.Tensor): tensor of true labels

    Returns:
        float: average recall
    """
    if torch.sum(labels) > 0:
        return torch.sum(pred * labels).item() / torch.sum(labels).item()
    else:
        return float("NaN")


def test(model: nn.Module, dataloader: DataLoader):
    """Test the model on a corpus of graphs loaded by a dataloader.

    Args:
        model (nn.Module): tart model to test
        dataloader (DataLoader): dataloader for test data
    """
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []

    with Progress(
        SpinnerColumn(),
        TextColumn("Loading test data..{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("", total=len(dataloader))

        for batch in dataloader:
            pos_a, pos_b, neg_a, neg_b = zip(*batch)
            pos_a = Batch.from_data_list(pos_a)
            pos_b = Batch.from_data_list(pos_b)
            neg_a = Batch.from_data_list(neg_a)
            neg_b = Batch.from_data_list(neg_b)

            if pos_a:
                pos_a = pos_a.to(get_device())
                pos_b = pos_b.to(get_device())
            neg_a = neg_a.to(get_device())
            neg_b = neg_b.to(get_device())

            labels = torch.tensor([1] * (pos_a.num_graphs if pos_a else 0) + [0] * neg_a.num_graphs).to(get_device())

            with torch.no_grad():
                # forward pass through GNN layers
                emb_neg_a, emb_neg_b = (model.encoder(neg_a), model.encoder(neg_b))
                if pos_a:
                    emb_pos_a, emb_pos_b = (model.encoder(pos_a), model.encoder(pos_b))
                    emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                    emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
                else:
                    emb_as, emb_bs = emb_neg_a, emb_neg_b

                # prediction from GNN
                pred = model(emb_as, emb_bs)
                raw_pred = model.predict(pred)

                # prediction from classifier
                pred = model.classifier(raw_pred.unsqueeze(1)).argmax(dim=-1)

            all_raw_preds.append(raw_pred)
            all_preds.append(pred)
            all_labels.append(labels)

    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)

    # metrics
    acc = torch.mean((pred == labels).type(torch.float))
    prec = precision(pred, labels)
    rec = recall(pred, labels)

    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    auroc = roc_auc_score(labels, pred)
    avg_prec = average_precision_score(labels, pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    console.print(
        "\nTest. Count: {}. Acc: {:.4f}.\n"
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n"
        "TN: {}. FP: {}. FN: {}. TP: {}\n".format(len(pred), acc, prec, rec, auroc, avg_prec, tn, fp, fn, tp)
    )


def validation(args: Namespace, model: nn.Module, test_pts: List, logger: SummaryWriter, batch_n: int, epoch: int):
    """validate the model on the validation set

    Args:
        args (Namespace): tart configs
        model (nn.Module): tart model
        test_pts (List): validation set
        logger (SummaryWriter): tensorboard logger
        batch_n (int): batch number
        epoch (int): epoch number
    """
    model.eval()
    all_raw_preds, all_preds, all_labels = [], [], []

    for pos_a, pos_b, neg_a, neg_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(get_device())
            pos_b = pos_b.to(get_device())
        neg_a = neg_a.to(get_device())
        neg_b = neg_b.to(get_device())

        labels = torch.tensor([1] * (pos_a.num_graphs if pos_a else 0) + [0] * neg_a.num_graphs).to(get_device())

        with torch.no_grad():
            # forward pass through GNN layers
            emb_neg_a, emb_neg_b = (model.encoder(neg_a), model.encoder(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (model.encoder(pos_a), model.encoder(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b

            # prediction from GNN
            pred = model(emb_as, emb_bs)
            raw_pred = model.predict(pred)

            # prediction from classifier
            pred = model.classifier(raw_pred.unsqueeze(1)).argmax(dim=-1)

        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)

    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)

    # metrics
    acc = torch.mean((pred == labels).type(torch.float))
    prec = precision(pred, labels)
    rec = recall(pred, labels)

    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()

    auroc = roc_auc_score(labels, pred)
    avg_prec = average_precision_score(labels, pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    console.print(
        "Validation. Epoch {}. Count: {}. Acc: {:.4f}.\n"
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n"
        "TN: {}. FP: {}. FN: {}. TP: {}".format(epoch, len(pred), acc, prec, rec, auroc, avg_prec, tn, fp, fn, tp)
    )

    if not args.test:
        logger.add_scalar("Accuracy/test", acc, batch_n)
        logger.add_scalar("Precision/test", prec, batch_n)
        logger.add_scalar("Recall/test", rec, batch_n)
        logger.add_scalar("AUROC/test", auroc, batch_n)
        logger.add_scalar("AvgPrec/test", avg_prec, batch_n)
        logger.add_scalar("TP/test", tp, batch_n)
        logger.add_scalar("TN/test", tn, batch_n)
        logger.add_scalar("FP/test", fp, batch_n)
        logger.add_scalar("FN/test", fn, batch_n)
        console.print("\n[italic]Saving {}[/ italic]\n".format(args.model_path))
        torch.save(model.state_dict(), args.model_path)


def tart_test(user_config_file: str):
    """tart's test API

    Args:
        user_config_file (str): config file path
    """
    console.print("[bright_green underline]Testing Model[/ bright_green underline]\n")
    parser = argparse.ArgumentParser()

    # reading user config from json file
    with open(user_config_file) as f:
        config_json = json.load(f)

    # build configs and their defaults
    config.build_optimizer_configs(parser)
    config.build_model_configs(parser)
    config.build_feature_configs(parser)

    args = parser.parse_args()

    # set to test mode
    args.test = True

    # set user defined configs
    args = config.init_user_configs(args, config_json)

    # set feature encoder
    feat_encoder = get_feature_encoder(args.feat_encoder)
    validate_feat_encoder(feat_encoder, config_json)

    args.n_train = args.n_batches * args.batch_size
    args.n_test = int(0.2 * args.n_train)

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    # build model
    model = build_model(models.SubgraphEmbedder, args)
    model.share_memory()

    # print("Moving model to device:", get_device())
    model = model.to(get_device())

    # create a corpus for train and test
    corpus = dataset.Corpus(args, feat_encoder, train=False)

    # create validation points
    loader = corpus.gen_data_loader(args.batch_size, train=(not args.test))

    # ====== TESTING ======
    test(model, loader)
