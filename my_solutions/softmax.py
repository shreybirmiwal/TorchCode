import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:

    max_val = torch.max(x, dim=dim, keepdim=True).values

    safe = torch.sub(x, max_val)
    exp = torch.exp(safe)

    total_sum = torch.sum(exp, dim=dim, keepdim=True)

    return torch.div(exp, total_sum)
