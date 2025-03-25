import glob
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchinfo import summary

from src.models.model import InputEmbeddings, PositionalEncoding, MultiHeadAttention, FeedForwardBlock, EncoderBlock, Encoder, DecoderBlock, Decoder, \
    ProjectionLayer, Transformer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import (
    WhitespaceSplit,
    Punctuation,
    Sequence as PreSequence,
)

logger = logging.getLogger(__name__)


def get_all_sentences(dataset, lang):
    lang_sentences = []
    for entry in dataset:
        lang_sentences.append(entry[lang + "_text"])
    return lang_sentences


def get_or_build_tokenizer(dataset, config, lang):
    is_new_tokenizer = False
    if config["tkn_build_scratch"]:
        tokenizer_path = Path(config["output_dir"], config["model_folder"], config['tokenizer_file'].format(lang))
        if not Path.exists(tokenizer_path):
            logger.info(f"Building tokenizer from scratch...")
            is_new_tokenizer = True
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"],
                                 min_frequency=config["tkn_min_freq"], vocab_size=config['tkn_max_vocab_size'])
            tokenizer.pre_tokenizer = PreSequence([WhitespaceSplit(), Punctuation()])

            if lang == "en":
                tokenizer.normalizer = Sequence(
                    [
                        NFD(),
                        StripAccents(),
                        Lowercase(),
                    ]
                )
            else:
                tokenizer.normalizer = Sequence(
                    [
                        NFD(),
                        Lowercase(),
                    ]
                )

            # Use train_from_iterator instead of train
            sentences = get_all_sentences(dataset, lang)
            tokenizer.train_from_iterator(sentences, trainer=trainer)
            tokenizer.save(str(tokenizer_path))
            return tokenizer, is_new_tokenizer
        else:
            logger.info(f"Loading {tokenizer_path}")
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            return tokenizer, is_new_tokenizer
    else:
        raise ValueError("Pre-built tokenizer not set")


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, d_ff: int = 2048, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1):
    # Input Embeddings
    src_embeddings = InputEmbeddings(d_model, src_vocab_size)
    tgt_embeddings = InputEmbeddings(d_model, tgt_vocab_size)
    # Positional Encodings
    src_pos_encoding = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Define Encoder
    encoder_blocks = []
    for _ in range(n_layers):
        encoder_self_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Define decoder
    decoder_blocks = []
    for _ in range(n_layers):
        decoder_mask_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
        decoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_mask_attention_block, decoder_cross_attention_block,
                                     decoder_feed_forward, dropout)
        decoder_blocks.append(decoder_block)
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Define Projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Define Transformer
    transformer = Transformer(encoder, decoder, src_embeddings, tgt_embeddings, src_pos_encoding,
                              tgt_pos_encoding, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def build_model(src_tkn_size, tgt_tkn_size, config):
    model = build_transformer(src_vocab_size=src_tkn_size, tgt_vocab_size=tgt_tkn_size, src_seq_len=config['model_seq_len'],
                              tgt_seq_len=config['model_seq_len'], d_model=config['model_d_model'], d_ff=config['model_d_ff'],
                              n_layers=config['model_layers'], n_heads=config['model_heads'], dropout=config['model_dropout'])
    return model


def save_checkpoint(model, optimizer, scheduler, epoch, model_dir, model_basename):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir + "/epochs", exist_ok=True)

    model_filename = f"epochs/{model_basename}_epoch{epoch}.pth"
    model_path = os.path.join(model_dir, model_filename)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # Add scheduler state if it exists
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, model_path)
    logger.info(f"Model saved: {model_path}")


def load_checkpoint(model, optimizer, model_path, device='cpu', scheduler=None):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    # Load model checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Restore scheduler state if it exists
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']

    logger.info(f"Model loaded from {model_path}, last trained epoch: {epoch}")
    return epoch


def get_checkpoint_path(config):
    model_dir = os.path.join(config['output_dir'], config['model_folder'])
    model_filename = f"**/{config['model_basename']}*"
    matching_files = glob.glob(os.path.join(model_dir, model_filename), recursive=True)

    if len(matching_files) == 0:
        return None
    matching_files.sort()
    return str(matching_files[-1])


def generate_summary(model, seq_len):
    logger.info("Generating model summary")
    src = torch.randint(0, 100, (1, seq_len))
    src_mask = torch.randint(0, 2, (seq_len, seq_len))
    tgt = torch.randint(0, 100, (1, seq_len))
    tgt_mask = torch.randint(0, 2, (seq_len, seq_len))

    summary(model, input_data=(src, src_mask, tgt, tgt_mask))
