import torch
from torch import nn

from attention.attention import Attention
from attention.config import AttentionConfig


class MultiHeadAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.heads = nn.ModuleList([Attention(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
