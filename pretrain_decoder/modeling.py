import torch
import copy
import transformers


class GPT2TokenizerFixed(transformers.GPT2Tokenizer):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_ids = [self.bos_token_id]
        output = bos_token_ids + token_ids_0 + [self.eos_token_id]
        return output

class Agregator(torch.nn.Module):
    def __init__(self,):
        super().__init__()
    
    def forward(self, batch, attention_mask=None, output_attentions=False):
        print(batch.shape)
        batch = batch.mean(dim=1)
        return batch

class Wav2Vec2GPT2(torch.nn.Module):
    def __init__(self, encoder, decoder, num_splits=20):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.agregator = Agregator()
        self.num_splits = num_splits
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, songs_part, lyrics_part):
        labels = lyrics_part["labels"]
        lyrics_part.pop("labels")
        encoder_output = self.encoder(**songs_part)
        # encoder_output = self.agregator(encoder_output.last_hidden_state)
        encoder_output = encoder_output.last_hidden_state
        encoder_output = encoder_output.reshape(len(encoder_output) // self.num_splits, self.num_splits * encoder_output.shape[1], encoder_output.shape[-1])
        # print(encoder_output[:, 8:12])
        lyrics_part.update({"encoder_hidden_states": encoder_output})
        output = self.decoder(**lyrics_part)
        print(output.logits[0][:5][-6:])
        loss = self.loss_fn(output.logits.transpose(1, 2), labels)
        return loss, output.logits
    
    def prepare_inputs(self, tokens):
        prepared_tokens = {
            "input_ids": torch.tensor([[tokens]]).to(self.encoder.device), 
            "attention_mask": torch.ones((1, len(tokens))).to(self.encoder.device)
        }
        return prepared_tokens
    
    def generate_next_sample(self, encoder_part, generated_tokens):
        generated_tokens.update({"encoder_hidden_states": encoder_part})
        logits = self.decoder(**generated_tokens)[1]
        print(logits.shape)
        return torch.argmax(logits[0][-1]).item()

    def generate(self, song_part, max_len):
        song_part = {key: val.to(self.encoder.device) for key, val in song_part.items()}
        encoder_output = self.encoder(**song_part)
        encoder_output = self.agregator(encoder_output.last_hidden_state)
        encoder_output = encoder_output.reshape(len(encoder_output) // self.num_splits, self.num_splits, encoder_output.shape[-1])
        print(encoder_output.shape)
        generated_tokens = [self.decoder.config.bos_token_id]
        for i in range(max_len):
            new_token = self.generate_next_sample(encoder_output, self.prepare_inputs(generated_tokens))
            generated_tokens.append(new_token)
        return generated_tokens


def get_decoder():
    tokenizer = GPT2TokenizerFixed.from_pretrained("openai-community/gpt2")
    special_tokens = {
        "bos_token": "<BOS>",
    }
    tokenizer.add_special_tokens(special_tokens)
    decoder = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    decoder.config.bos_token_id = tokenizer.bos_token_id
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

