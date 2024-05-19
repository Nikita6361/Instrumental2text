import torch
import json
import copy
import torchaudio
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm
import os
from os.path import isfile, join
import librosa


class SongsDataset(Dataset):
    def __init__(self, path_to_songs, path_to_lyrics, tokenizer, processor, resample=None):
        self.path_to_songs = path_to_songs
        self.path_to_lyrics = path_to_lyrics
        files = [x[0] for x in os.walk(path_to_songs) if x[0] != path_to_songs]
        self.songs = []
        self.lyrics = []
        lens = []
        i = 0
        for file in tqdm(files):
            # if i > 10:
            #     break
            # i += 1
            song_hash = file[len(self.path_to_songs):]
            tmp, sr = librosa.load(file + "/no_vocals.mp3", sr=16000)
            arr = torch.tensor(tmp)
            song = processor(arr, sampling_rate=16000, return_tensors="pt")
            lens.append(song["input_values"].shape[1])
            self.songs.append(song)
            with open(self.path_to_lyrics + "/" + song_hash + ".txt") as f:
                data = " ".join(f.readlines())
            self.lyrics.append(data)
        
        self.max_len = int(np.percentile(lens, q=60))
        print("Max song len reduced to:", self.max_len)
        for i in range(len(self.songs)):
            self.songs[i]["input_values"] = self.songs[i]["input_values"][:, :min(self.max_len, self.songs[i]["input_values"].shape[1])]

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        return self.songs[idx], self.lyrics[idx]


class CollateSongs:
    def __init__(self, tokenizer, num_splits):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_splits = num_splits
    
    def __call__(self, batch):
        N = max(batch, key=lambda x: x[0]['input_values'].shape[1])[0]['input_values'].shape[1]

        songs = []
        texts = []
        for song, text in batch:
            songs.append(self.split(song['input_values'], N))
            texts.append(text)
        songs_part = {'input_values': torch.cat(songs, dim=0)}
        texts_part = self.tokenizer(texts, padding=True, return_tensors="pt", max_length=986, truncation=True)
        labels = copy.deepcopy(texts_part["input_ids"])
        labels[texts_part["attention_mask"] == 0] = -100
        labels = labels[:, 1:]
        labels = torch.cat((labels, -100 * torch.ones((labels.shape[0], 1), dtype=torch.int32)), dim=-1)
        texts_part.update({'labels': labels})
        return songs_part, texts_part
    
    def split(self, tensor, N):
        padding = (0, N - tensor.shape[1])
        tensor = torch.nn.functional.pad(tensor, padding, value=0.0)
        return tensor


def get_optim(model, lr, num_epochs, train_dl_len):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    total_iters = num_epochs * train_dl_len
    warmup_iters = int(0.1 * total_iters)
    post_iters = total_iters - warmup_iters
    schedular1 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.00001, total_iters=warmup_iters)
    schedular2 = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=0, total_iters=post_iters)
    schedular = torch.optim.lr_scheduler.SequentialLR(optim, [schedular1, schedular2], milestones=[warmup_iters])
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)
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
