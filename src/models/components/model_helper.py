import numpy as np
import torch

def positional_embedding(n, dim):
    pe = torch.zeros(n, dim)
    position = torch.arange(0, n).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * -(np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe