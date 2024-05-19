import argparse
import os

import transformers
import copy
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from datasets import load_dataset

import utils as utils
import modeling as modeling
import matplotlib.pyplot as plt
import numpy as np

torch.set_num_threads(1)


def init_process(local_rank, fn, config, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size, config)

def average_gradients(model, num_accum_steps=1):
    size = float(dist.get_world_size())
    for param in model.named_parameters():
        if param[1].grad is None:
            continue

        dist.all_reduce(param[1].grad.data, op=dist.ReduceOp.SUM)
        param[1].grad.data /= (size * num_accum_steps)


def plot_losses(losses, mode):
    plt.figure
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.scatter(list(range(len(losses))), losses, c='black')
    plt.savefig(mode + "_losses.png")
    plt.close()


def run_training(rank, size, config):
    torch.manual_seed(0)

    tokenizer = modeling.GPT2TokenizerFixed.from_pretrained("openai-community/gpt2")
    special_tokens = {
        "bos_token": "<BOS>",
    }
    tokenizer.add_special_tokens(special_tokens)
    tmp_weights = modeling.Gpt2Kostil.from_pretrained("openai-community/gpt2")

    model_config = copy.copy(tmp_weights.config)
    model_config.add_cross_attention = True
    model_config.cross_attention_hidden_size = 1024
    decoder = modeling.Gpt2Kostil(model_config)
    decoder.config.bos_token_id = tokenizer.bos_token_id
    missing_keys = decoder.load_state_dict(tmp_weights.state_dict(), strict=False)
    decoder.resize_token_embeddings(len(tokenizer))
    decoder.load_state_dict(torch.load("../pretrain_decoder/model_state_dict_e5_lr0.0001", map_location=torch.device('cpu')), strict=False)
    
    if rank == 0:
        print("Loading model")
    model, feature_extractor, _ = modeling.get_model(decoder)
    if rank == 0:
        print("Model loaded")

    train_ds = utils.SongsDataset("/home/b6361nik/kursach/htdemucs", "/home/b6361nik/kursach/annotations", None, feature_extractor, resample=True)
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        2,
        collate_fn=utils.CollateSongs(tokenizer, num_splits=20),
        sampler=DistributedSampler(train_ds, size, rank),
    )
    val_ds = utils.SongsDataset("/home/b6361nik/kursach/val_songs", "/home/b6361nik/kursach/val_texts", None, feature_extractor, resample=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, 
        2,
        collate_fn=utils.CollateSongs(tokenizer, num_splits=20),
        sampler=DistributedSampler(val_ds, size, rank),
    )

    device = torch.device("cuda:{}".format(rank))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    optim, scaler, schedular = utils.get_optim(model, config.lr, config.num_epochs, len(train_dl) // config.accum_steps)

    num_train_batches = len(train_dl)

    train_losses = []
    test_losses = []
    optim.zero_grad()
    for epo in range(config.num_epochs):
        model.train()
        tmp = []
        for i, (song, text) in enumerate(train_dl):
            optim.zero_grad()
            song = {key: val.to(device) for key, val in song.items()}
            text = {key: val.to(device) for key, val in text.items()}

            with torch.cuda.amp.autocast():
                loss, _ = model(song, text)
            scaler.scale(loss).backward()

            if (i + 1) % config.accum_steps == 0 or i == len(train_dl) - 1:
                average_gradients(model, config.accum_steps)
                scaler.step(optim)
                scaler.update()
                schedular.step()
                optim.zero_grad()
            if rank == 0:
                tmp.append(loss.item())
                print(f"Batch {i + 1}/{num_train_batches}, loss: {loss.detach()}")
        
        if rank == 0:
            train_losses.append(np.mean(tmp))

        # if rank == 0:
        #     model.eval()
        #     tmp = []
        #     with torch.no_grad():
        #         for i, (song, text) in enumerate(val_dl):
        #             song = {key: val.to(device) for key, val in song.items()}
        #             text = {key: val.to(device) for key, val in text.items()}
        #             with torch.cuda.amp.autocast():
        #                 loss, _ = model(song, text)
        #             tmp.append(loss.item())
        #     test_losses.append(np.mean(tmp))
        #     print(test_losses[-1])
        # torch.distributed.barrier()
    
    if rank == 0:
        plot_losses(train_losses, "train")
        # plot_losses(test_losses, "test")

    if dist.get_rank() == 0:
        torch.save(model.state_dict(), "./model_state_dict_e{}_lr{}".format(config.num_epochs, config.lr))


def ParseArgs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-a', '--accum_steps', type=int)
    parser.add_argument('-e', '--num_epochs', type=int)
    parser.add_argument('--lr', type=float)
    
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = ParseArgs()
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, config=config, backend="nccl")

