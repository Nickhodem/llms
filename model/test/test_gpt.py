from model.gpt import GPT
from transformer.config import AttentionConfig


def test_gpt():
    config = AttentionConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=16,
        n_embd=1600,
        dropout=0.1,
        bias=True,
    )
    gpt = GPT(config)
    gpt.to("cuda")
    print(f"Number of parameters, {gpt.get_num_params()}")
    print(f"Number of flops, {gpt.estimate_mfu(40, 1.0)}")
