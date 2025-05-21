import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

def series_decomp(x: Tensor) -> Tuple[Tensor, Tensor]:
    """分解时间序列为趋势和季节分量"""
    mean = torch.mean(x, dim=1, keepdim=True)
    seasonal = x - mean
    return seasonal, mean

class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
        
    def forward(self, x):  # x: [bs x n_vars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
