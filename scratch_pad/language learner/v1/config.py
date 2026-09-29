from dataclasses import dataclass


@dataclass
class Pair1Config:
    """Language foundation: decoder-only transformer, GELU, backprop-trained.

    Scaled down from the original 384/6/6 spec (prototype_spec.md) so a full
    training run completes in a few minutes on CPU-only hardware -- see
    piece_collect.md for why architecture, not scale, was the point here.
    """
    vocab_size: int = 4000        # set after tokenizer training
    context_size: int = 32
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    book_path: str = "data/alice_clean.txt"
    qa_path: str = "data/qa_behavior.txt"
    tokenizer_path: str = "data/tokenizer.json"
    out_dir: str = "out"
    batch_size: int = 4
    max_iters: int = 4000
    eval_interval: int = 400
    eval_iters: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"          # falls back to cpu automatically if unavailable


@dataclass
class Pair2Config:
    """Short-term memory: local Hebbian update on Pair 1's output vectors."""
    dim: int = 384
    lr: float = 5e-4
    decay: float = 0.02           # per-step decay applied to the short-term trace
    bound: float = 3.0            # tanh-style magnitude cap


@dataclass
class Pair3Config:
    """Long-term memory: sparse quantized key-value store."""
    dim: int = 384
    codebook_size: int = 512
    similarity_threshold: float = 0.85   # snap-to-existing-entry threshold
    promotion_loss_threshold: float = 3.0  # surprise trigger for promotion
    repetition_min: int = 2               # times a concept must recur to count as reinforced
    capacity: int = 2000
