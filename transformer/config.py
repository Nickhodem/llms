from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionConfig:
    vocab_size: int
    n_layer: int
    n_embd: int
    block_size: int
    n_head: int
    dropout: float
    head_size: Optional[int] = None
    bias: bool = True


    def __post_init__(self):
        if self.head_size is None:
            self.head_size = self.n_embd // self.n_head
        else:
            assert self.n_embd // self.n_head == self.head_size
