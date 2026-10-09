import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):

        if not self.training:
            return x

        # else we are in train mode
        rand = torch.rand(x.shape)
        x_masked = torch.where(rand > self.p, x / (1-self.p) , 0)

        return x_masked

        
