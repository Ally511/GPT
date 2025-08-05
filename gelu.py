import torch
import torch.nn as nn
import math

class Gelu(nn.Module):
    """GELU (Gaussian Error Linear Units) activation function"""
    def forward(self,x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))