import torch
import numpy as np
import gc
import random
from random import sample
from pathlib import Path

from config import settings

def set_up_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clear_cuda_memory():
    torch.cuda.empty_cache()
    gc.collect()

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)
    
    # Create model save path
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model.state_dict(), f=model_save_path)
    

def load_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    target_dir_path = Path(target_dir)
    model_save_path = target_dir_path / model_name

    # Load the model state dict()
    print(f"[INFO] Loading model: {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))