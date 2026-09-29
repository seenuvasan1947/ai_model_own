from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 4000        # set after tokenizer training
    context_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    n_experts: int = 4            # MoE experts per feedforward block
    bias: bool = True


@dataclass
class TrainConfig:
    data_path: str = "data/corpus.txt"
    tokenizer_path: str = "data/tokenizer.json"
    out_dir: str = "out"
    batch_size: int = 32
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"          # falls back to cpu automatically in train.py if unavailable
