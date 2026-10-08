import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features, requires_grad=True))
        self.bias nn.Parameter(torch.zeros(out_features, requires_grad=True))
        
        # Initialize weight and bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x @ torch.transpose(self.weight) + self.bias
        # Compute y = x @ W^T + b
