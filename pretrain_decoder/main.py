import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

import utils as utils
import modeling as modeling

torch.set_num_threads(1)


def plot_losses(losses, mode):
    plt.figure
    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.scatter(list(range(len(losses))), losses, c='black')
    plt.savefig(mode + "_losses.png")
    plt.close()

def init_process(local_rank, fn, config, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size, config)

def average_gradients(model, num_accum_steps=1):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= (size * num_accum_steps)

def run_training(rank, size, config):
    torch.manual_seed(0)

    model, tokenizer = modeling.get_decoder()
    
    dataset = load_dataset("sadFaceEmoji/english-poems", split="train")
    dataset = dataset.map(lambda examples: tokenizer(examples["poem"], return_tensors="pt"), remove_columns=["id", "poem"])
    dataset = dataset.train_test_split(test_size=0.1)

    collate_fn = utils.PretrainCollector(tokenizer.eos_token_id)
    train_loader = DataLoader(
        dataset["train"],
        sampler=DistributedSampler(dataset["train"], size, rank),
        batch_size=32,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset["test"],
        sampler=DistributedSampler(dataset["test"], size, rank),
        batch_size=64,
        collate_fn=collate_fn
    )

    print("cuda:{}".format(rank + 4))
    device = torch.device("cuda:{}".format(rank + 4))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = True

    optim, scaler, schedular = utils.get_optim(model, config.lr, config.num_epochs, len(train_loader) // config.accum_steps)

    train_losses = []
    test_losses = []
    optim.zero_grad()
    for epo in range(config.num_epochs):
        epoch_loss = torch.zeros((1,), device=device)
        tmp_losses = []
        model.train()
        for i, data in enumerate(train_loader):
            data["input_ids"] = data["input_ids"].to(device)
            data["attention_mask"] = data["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                output = model(**data, labels=data["input_ids"])
            epoch_loss += output.loss.detach()

            scaler.scale(output.loss).backward()
            if (i + 1) % config.accum_steps == 0:
                average_gradients(model, config.accum_steps)
                scaler.step(optim)
                scaler.update()
                schedular.step()
                optim.zero_grad()
            
            tmp_losses.append(output.loss.detach().item())
        train_losses.append(np.mean(tmp_losses))

        if rank == 0:
            print("Epo {} train loss {}".format(epo, train_losses[-1]))
            tmp_losses = []
            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    data["input_ids"] = data["input_ids"].to(device)
                    data["attention_mask"] = data["attention_mask"].to(device)
                    output = model(**data, labels=data["input_ids"])
                    tmp_losses.append(output.loss.detach().item())
            test_losses.append(np.mean(tmp_losses))
            print("Epo {} val loss {}".format(epo, test_losses[-1]))
        
        dist.barrier()
    
    if rank == 0:
        plot_losses(train_losses, "train")
        plot_losses(test_losses, "test")
    
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

