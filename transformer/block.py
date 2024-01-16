from torch import nn

from transformer.attention import CausalSelfAttention
from transformer.config import AttentionConfig
from transformer.mlp import MLP
from transformer.norm import LayerNorm


class Block(nn.Module):

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x