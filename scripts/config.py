from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    path: str = "gpt2"
    min_lr_factor: float = 0.1
    warmup_steps: int = 27 # need to explain why
    warmdown_steps: int = 214 # need to explain why
    lr: float = 16e-4
