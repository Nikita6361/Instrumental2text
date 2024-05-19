import torch
import json
import copy
import torchaudio
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm
from os import listdir
from os.path import isfile, join


class PretrainCollector:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, data):
        dict_data = {"input_ids": [], "attention_mask":  []}
        for sample in data:
            for key in sample:
                dict_data[key].append(torch.tensor(sample[key][0]))
        
        dict_data["input_ids"] = torch.nn.utils.rnn.pad_sequence(dict_data["input_ids"], batch_first=True, padding_value=self.pad_token_id)
        dict_data["attention_mask"] = torch.nn.utils.rnn.pad_sequence(dict_data["attention_mask"], batch_first=True, padding_value=0)
        return dict_data


def get_optim(model, lr, num_epochs, train_dl_len):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    total_iters = num_epochs * train_dl_len
    warmup_iters = int(0.1 * total_iters)
    post_iters = total_iters - warmup_iters
    schedular1 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.00001, total_iters=warmup_iters)
    schedular2 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=0, total_iters=post_iters)
    schedular = torch.optim.lr_scheduler.SequentialLR(optim, [schedular1, schedular2], milestones=[warmup_iters])
    scaler = torch.cuda.amp.GradScaler()
    return optim, scaler, schedular

    
def train(model, optim, schedular, scaler, train_dl, num_epo, device):
    model.train()
    loss_ema = None
    for _ in range(num_epo):
        for song, text in train_dl:
            optim.zero_grad()
            song = {key: val.to(device) for key, val in song.items()}
            text = {key: val.to(device) for key, val in text.items()}
            with torch.cuda.amp.autocast():
                output = model(song, text)
            scaler.scale(output[0]).backward()
            scaler.step(optim)
            scaler.update()
            schedular.step()
            loss_ema = output[0].item() if loss_ema is None else 0.6 * loss_ema + 0.4 * output[0].item()
            print(loss_ema)

def val(model, val_dl, device):
    pass
