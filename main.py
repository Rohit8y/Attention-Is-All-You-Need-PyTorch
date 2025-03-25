import argparse
import logging
import os

import torch

from src.data.dataloaders import get_loaders
from src.models.setup import build_model, generate_summary, get_checkpoint_path, load_checkpoint, save_checkpoint
from src.training.trainer import train_block
from src.utils.helpers import set_seed, read_config_file, setup_wandb
from src.utils.schedulers import NoamScheduler

# Call this function at the beginning of your script
set_seed(42)  # You can use any integer seed you prefer

# Initialize logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Parser for Transformer Implementation")
parser.add_argument('--config', type=str, default="src/config/base_transformer.toml")
parser.add_argument('--skip_wandb', action='store_true', default=True)


def main(args):
    config = read_config_file(args.config)

    # Define the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    device = torch.device(device)

    # Setup W&B
    if not args.skip_wandb:
        setup_wandb(config)

    # Make relevant output dirs
    os.makedirs(config['output_dir'], exist_ok=True)
    model_dir = os.path.join(config['output_dir'], config['model_folder'])
    os.makedirs(model_dir, exist_ok=True)

    # Get tokenizer and model
    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_loaders(config)

    # Build model
    model = build_model(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), config)
    generate_summary(model, config['model_seq_len'])
    model = model.to(device)

    # Model attributes
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    # init scheduler
    scheduler = NoamScheduler(optimizer, config['model_d_model'], warmup_steps=config['scheduler_warmup_steps'])

    # Training
    latest_epoch = 0

    if config['load_checkpoint']:
        logger.info("Checking, if we need to load previous checkpoint...")
        checkpoint_path = get_checkpoint_path(config)
        if checkpoint_path is not None:
            latest_epoch = load_checkpoint(model, optimizer, checkpoint_path, device, scheduler)
            latest_epoch += 1
        else:
            logger.info("No checkpoint found...")

    logger.info(f"Training starting from epoch: {latest_epoch}")

    # Training
    for epoch in range(latest_epoch, config['num_epochs']):
        train_block(model, optimizer, scheduler, tgt_tokenizer, train_dataloader, epoch, device, loss_fn, args)
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, model_dir, config['model_basename'])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
