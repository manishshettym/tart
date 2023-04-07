import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from deepsnap.batch import Batch

from tart.representation.test import validation, test
from tart.representation import config, models, dataset
from tart.utils.model_utils import build_model, build_optimizer, get_device
from tart.utils.train_utils import init_logger, start_workers, make_validation_set
from tart.utils.config_utils import validate_feat_encoder
from tart.utils.tart_utils import print_header, summarize_tart_run

torch.multiprocessing.set_sharing_strategy('file_system')


def train(args, model, corpus, in_queue, out_queue):
    """
    Train the model that was initialized
    
    Args:
        args: Commandline arguments
        model: GNN model
        corpus: dataset to train the GNN model
        in_queue: input queue to an intersection computation worker
        out_queue: output queue to an intersection computation worker
    """
    scheduler, opt = build_optimizer(args, model.parameters())
    clf_opt = optim.Adam(model.classifier.parameters(), lr=args.lr)

    done = False
    while not done:
        dataloader = corpus.gen_data_loader(args.batch_size, train=True)

        for batch in dataloader:
            msg, _ = in_queue.get()
            if msg == "done":
                done = True
                break

            model.train()
            model.zero_grad()

            pos_a, pos_b, neg_a, neg_b = zip(*batch)

            # convert to DeepSnap Batch
            pos_a = Batch.from_data_list(pos_a).to(get_device())
            pos_b = Batch.from_data_list(pos_b).to(get_device())
            neg_a = Batch.from_data_list(neg_a).to(get_device())
            neg_b = Batch.from_data_list(neg_b).to(get_device())

            # get embeddings
            emb_pos_a = model.encoder(pos_a)  # pos target
            emb_pos_b = model.encoder(pos_b)  # pos query
            emb_neg_a = model.encoder(neg_a)  # neg target
            emb_neg_b = model.encoder(neg_b)  # neg query

            # concatenate
            emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
            emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)

            labels = torch.tensor(
                [1] * pos_a.num_graphs
                + [0] * neg_a.num_graphs).to(get_device())
            
            # make predictions
            pred = model(emb_as, emb_bs)
            loss = model.criterion(pred, labels)

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if scheduler:
                scheduler.step()
            
            with torch.no_grad():
                pred = model.predict(pred)

            model.classifier.zero_grad()
            pred = model.classifier(pred.unsqueeze(1))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            clf_opt.step()
            
            # metrics
            pred = pred.argmax(dim=-1)
            acc = torch.mean((pred == labels).type(torch.float))
            train_loss = loss.item()
            train_acc = acc.item()

            out_queue.put(("step", (train_loss, train_acc)))


def train_loop(args, feat_encoder):
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    
    in_queue, out_queue = mp.Queue(), mp.Queue()

    # init logger
    logger = init_logger(args)

    # build model
    model = build_model(models.SubgraphEmbedder, args)
    model.share_memory()
    
    # print("Moving model to device:", get_device())
    model = model.to(get_device())

    # create a corpus for train and test
    corpus = dataset.Corpus(args, feat_encoder, train=(not args.test))

    # create validation points
    loader = corpus.gen_data_loader(args.batch_size, train=False)

    summarize_tart_run(args)

    # ====== TRAINING ======
    validation_pts = make_validation_set(loader)

    assert args.n_iters > 0, "Number of iterations must be greater than 0"
    assert args.n_batches // args.eval_interval > 0, "Number of epochs per iteration must be greater than 0"
    
    for iter in range(args.n_iters):
        print(f"Iteration #{iter}")
        workers = start_workers(train, model, corpus, in_queue, out_queue, args)

        batch_n = 0
        for epoch in range(args.n_batches // args.eval_interval):
            print(f"Epoch #{epoch}")

            for _ in range(args.eval_interval):
                in_queue.put(("step", None))
            
            # loop over mini-batches in an epoch
            for _ in range(args.eval_interval):
                _, result = out_queue.get()
                train_loss, train_acc = result
                print(f"Batch {batch_n}. Loss: {train_loss:.4f}. \
                    Train acc: {train_acc:.4f}\n")
                
                logger.add_scalar("Loss(train)", train_loss, batch_n)
                logger.add_scalar("Acc(train)", train_acc, batch_n)
                batch_n += 1

            # validation after an epoch
            validation(args, model, validation_pts, logger, batch_n, epoch)
    
        for _ in range(args.n_workers):
            in_queue.put(("done", None))
        for worker in workers:
            worker.join()


def tart_train(user_config_file, feat_encoder):
    print_header()
    parser = argparse.ArgumentParser()

    # reading user config from json file
    with open(user_config_file) as f:
        config_json = json.load(f)

    # build configs and their defaults
    config.build_optimizer_configs(parser)
    config.build_model_configs(parser)
    config.build_feature_configs(parser)

    args = parser.parse_args()

    # set user defined configs
    args = config.init_user_configs(args, config_json)

    # validate user defined feature encoder
    feat_encoder = validate_feat_encoder(feat_encoder, config_json)
    
    args.n_train = args.n_batches * args.batch_size
    args.n_test = int(0.2 * args.n_train)

    train_loop(args, feat_encoder)
