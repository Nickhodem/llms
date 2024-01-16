import torch
from transformer.attention import Attention
from transformer.config import AttentionConfig


def test_head_attention():
    config = AttentionConfig(n_embd=64, block_size=8, n_head=4, dropout=0.1)
    batch_size = 2
    attention = Attention(config)
    x = torch.rand(batch_size, config.block_size, config.n_embd)
    y = attention(x)
    assert y.shape == (batch_size, config.block_size, config.head_size)
