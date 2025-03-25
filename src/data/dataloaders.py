import json
import logging

from src.data.datasets import TranslationDataset
from src.models.setup import get_or_build_tokenizer

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_loaders(config):
    data_path = config["data_path"]
    logger.info(f"Loading dataset: {data_path}")
    with open(data_path, 'r') as file:
        dataset = json.load(file)
    logger.info(f"Dataset loaded with total entries: {len(dataset)}")

    # Generate tokenizers
    src_tokenizer, created_now = get_or_build_tokenizer(dataset, config, config["src_lang"])
    tgt_tokenizer, _ = get_or_build_tokenizer(dataset, config, config["tgt_lang"])

    # Split into train and val
    train_data, val_data = train_test_split(dataset, train_size=config["data_train_split"])
    logger.info(f"Train set: {len(train_data)}, Val set: {len(val_data)}")

    # Init dataset class
    train_ds = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, config["src_lang"], config["tgt_lang"], config["model_seq_len"], clip=True)
    val_ds = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, config["src_lang"], config["tgt_lang"], config["model_seq_len"], clip=True)

    if created_now:
        # Find the maximum length of each sentence in the source and target sentence
        max_len_src = 0
        max_len_tgt = 0
        for item in dataset:
            src_ids = src_tokenizer.encode(item[config['src_lang'] + "_text"]).ids
            tgt_ids = tgt_tokenizer.encode(item[config['tgt_lang'] + "_text"]).ids
            max_len_src = max(max_len_src, len(src_ids))
            max_len_tgt = max(max_len_tgt, len(tgt_ids))

        logger.info(f'Max length of source sentence: {max_len_src}')
        logger.info(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer
