import os
import tomli

import torch
import numpy as np
import random
import wandb
from datetime import datetime


# Set a fixed seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = "false"


def read_config_file(file_path):
    try:
        with open(file_path, 'rb') as toml_file:
            toml_dict = tomli.load(toml_file)
        return toml_dict
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except tomli.TOMLDecodeError as e:
        print(f"Error parsing TOML file: {e}")
        raise


def setup_wandb(config):
    # Initialize W&B
    wandb.init(
        entity=config['wandb_entity'],
        project=config['wandb_project'],
        name=config['wandb_run_name'] + '-' + datetime.now().strftime('%Y%m%d-%H%M%S'),
        config=config
    )
