import torch
from torch import nn

from repos.pfrl.pfrl.distributions import Delta


class DeterministicHead(nn.Module):
    """Head module for a deterministic policy."""

    def forward(self, loc):
        return torch.distributions.Independent(Delta(loc=loc), 1)
