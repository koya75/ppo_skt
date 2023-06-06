import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):

  def __init__(self, dim, dropout = 0.1, max_len = 6000, device="cuda"):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1).to(device)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)).to(device)
    pe = torch.zeros(max_len, 1, dim).to(device)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)