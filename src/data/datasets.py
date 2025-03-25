import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len, clip=False):
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.clip = clip

        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def clip_data(self, tokens):
        if len(tokens) > (self.seq_len - 2):
            return tokens[0:(self.seq_len - 2)]
        else:
            return tokens

    def __getitem__(self, idx):
        src_target_data = self.dataset[idx]
        src_text = src_target_data[self.src_lang + "_text"]
        tgt_text = src_target_data[self.tgt_lang + "_text"]

        # Convert text into tokens
        src_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        # Clip data if longer than seq_length
        if self.clip:
            src_tokens = self.clip_data(src_tokens)
            tgt_tokens = self.clip_data(tgt_tokens)

        # Find padding tokens
        src_padding_count = self.seq_len - len(src_tokens) - 2  # <sos> and <eos>
        tgt_padding_count = self.seq_len - len(tgt_tokens) - 1  # <sos> in target and </eos> in label

        if src_padding_count < 0 or tgt_padding_count < 0:
            raise ValueError(f"Sentence length longer than sequence length: {self.seq_len}")

        encoder_input = torch.cat([self.sos_token, torch.tensor(src_tokens, dtype=torch.int64), self.eos_token,
                                   torch.tensor([self.pad_token] * src_padding_count, dtype=torch.int64)], dim=0)

        decoder_input = torch.cat([self.sos_token, torch.tensor(tgt_tokens, dtype=torch.int64),
                                   torch.tensor([self.pad_token] * tgt_padding_count, dtype=torch.int64)], dim=0)

        target_output = torch.cat([torch.tensor(tgt_tokens, dtype=torch.int64), self.eos_token,
                                   torch.tensor([self.pad_token] * tgt_padding_count, dtype=torch.int64)], dim=0)

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert target_output.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_len) & (1, seq_len, seq_len),
            "target_output": target_output,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return (mask == 0).int()