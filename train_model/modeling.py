from typing import Tuple
import torch
import copy
import numpy as np
import transformers

from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class GPT2TokenizerFixed(transformers.GPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_ids = [self.bos_token_id]
        output = bos_token_ids + token_ids_0 + [self.eos_token_id]
        return output

class Agregator(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, batch, attention_mask=None, output_attentions=False):
        batch = batch.mean(dim=1)
        return batch


class Gpt2Kostil(transformers.GPT2LMHeadModel):
    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states)

class Checkpoint(torch.nn.Module):
  def __init__(self, layer):
    super().__init__()
    self.layer = layer

  def forward(self, *args, **inputs):
    return checkpoint(self.layer.forward, *args, use_reentrant=False, **inputs)

class Wav2Vec2GPT2(torch.nn.Module):
    def __init__(self, encoder, decoder, num_splits=20):
        super().__init__()
        # self.encoder = Checkpoint(encoder)
        self.encoder = encoder
        self.encoder.gradient_checkpointing_enable()
        self.decoder = decoder
        self.num_splits = num_splits
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    @staticmethod
    def create_custom_matrix(rows, columns):
        cols_per_group = columns // rows
        matrix = torch.zeros(rows, rows * cols_per_group)
        idx = torch.arange(rows * cols_per_group).reshape(rows, cols_per_group)
        matrix.scatter_(1, idx, 1)
        
        if columns % rows != 0:
            to_concat = torch.zeros(rows, columns % rows)
            to_concat[-1, :] = 1
            matrix = torch.cat([matrix, to_concat], dim=-1)
        
        return matrix / 512
    
    def forward(self, songs_part, lyrics_part):
        labels = lyrics_part["labels"]
        lyrics_part.pop("labels")
        encoder_output = self.encoder(songs_part["input_values"])

        encoder_output = encoder_output.last_hidden_state
        my_conv = self.create_custom_matrix(512, encoder_output.shape[1]).to(encoder_output.device)
        encoder_output = my_conv[None, :, :] @ encoder_output

        output = self.decoder(lyrics_part["input_ids"], lyrics_part["attention_mask"], encoder_output)
        loss = self.loss_fn(output.logits.transpose(1, 2), labels)
        return loss, output.logits
    
    def prepare_inputs(self, tokens):
        prepared_tokens = {
            "input_ids": torch.tensor([[tokens]]).to(self.encoder.device), 
            "attention_mask": torch.ones((1, len(tokens))).to(self.encoder.device)
        }
        return prepared_tokens
    
    def generate_next_sample(self, encoder_part, generated_tokens, temp):
        logits = self.decoder(generated_tokens["input_ids"], generated_tokens["attention_mask"], encoder_part).logits
        logits = logits[0][0][-1]
        inds = torch.argsort(-logits)
        return inds[np.random.choice(5, 1, p=torch.nn.functional.softmax(logits[inds[:5]] / temp).cpu().data.numpy()).item()]

    def generate(self, song_part, max_len, bos_token_id, temp=1):
        song_part = {key: val.to(self.encoder.device) for key, val in song_part.items()}
        encoder_output = self.encoder(**song_part)
        encoder_output = encoder_output.last_hidden_state
        my_conv = self.create_custom_matrix(512, encoder_output.shape[1]).to(encoder_output.device)
        encoder_output = my_conv[None, :, :] @ encoder_output
        generated_tokens = [bos_token_id]
        for i in range(max_len):
            new_token = self.generate_next_sample(encoder_output, self.prepare_inputs(generated_tokens), temp)
            generated_tokens.append(new_token)
        return generated_tokens


def get_decoder():
    tokenizer = GPT2TokenizerFixed.from_pretrained("openai-community/gpt2")
    special_tokens = {
        "bos_token": "<BOS>",
    }
    tokenizer.add_special_tokens(special_tokens)
    tmp_weights = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

    config = copy.copy(tmp_weights.config)
    config.add_cross_attention = True
    config.cross_attention_hidden_size = 1024
    decoder = transformers.GPT2LMHeadModel(config)
    decoder.config.bos_token_id = tokenizer.bos_token_id
    missing_keys = decoder.load_state_dict(tmp_weights.state_dict(), strict=False)
    decoder.resize_token_embeddings(len(tokenizer))
    
    return decoder, tokenizer


def get_model(decoder=None):
    tokenizer = None
    if decoder is None:
        decoder, tokenizer = get_decoder()

    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    encoder = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2GPT2(encoder, decoder)
    
    return model, feature_extractor, tokenizer

