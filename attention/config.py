from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    n_embd: int
    block_size: int
    n_heads: int
    dropout: float
    head_size: Optional[int] = None

    def __post_init__(self):
        if self.head_size is None:
            self.head_size = self.n_embd // self.n_heads
        else:
            assert self.n_embd // self.n_heads == self.head_size
