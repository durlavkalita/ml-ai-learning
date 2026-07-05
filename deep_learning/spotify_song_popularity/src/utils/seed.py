import torch

def set_seed(seed: int):
    torch.manual_seed(seed)
