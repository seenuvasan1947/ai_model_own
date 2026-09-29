from dataclasses import dataclass


@dataclass
class Pair1Config:
    """Language foundation: decoder-only transformer, GELU, backprop-trained.

    Scaled down from the original 384/6/6 spec (prototype_spec.md) so a full
    training run completes in a few minutes on CPU-only hardware -- see
    piece_collect.md for why architecture, not scale, was the point here.
    """
    vocab_size: int = 4000        # set after tokenizer training
    context_size: int = 64
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    book_path: str = "data/combined_corpus.txt"   # v3: Alice + Sherlock Holmes (~97.5k words, vs 26.5k in v1/v2)
    qa_path: str = "data/qa_behavior.txt"
    tokenizer_path: str = "data/tokenizer.json"
    out_dir: str = "out"
    batch_size: int = 8
    max_iters: int = 6000
    eval_interval: int = 500
    eval_iters: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"          # falls back to cpu automatically if unavailable

    # Added retroactively to re-measure v3's original run with a held-out
    # split (see v4/UPDATES.md fix #3) -- v3's own architecture/vocab/data
    # are otherwise left exactly as originally configured.
    val_split: float = 0.1


@dataclass
class Pair2Config:
    """Short-term memory: local Hebbian update on Pair 1's output vectors.

    `dim` should normally be set to match Pair1Config.n_embd at construction
    time (see ask.py / continual_read.py), not left at this default --
    dimension and layer count are adjustable per-run via Pair1Config, and
    this module's `dim` tracks whatever Pair 1 is currently using.
    """
    dim: int = 384

    # Item 4 (GAP_VS_TITANS.md): surprise and momentum are separate terms
    # instead of one blended Hebbian update.
    surprise_lr: float = 5e-4      # how strongly a fresh co-activation registers
    momentum_decay: float = 0.85   # how much of the accumulated direction persists per step

    # Item 2: adaptive decay range. Actual decay each step is interpolated
    # between these based on how dissimilar the new input is to the current
    # trace (dissimilar -> decay_max, similar/reinforcing -> decay_min),
    # instead of one fixed constant.
    decay_min: float = 0.005
    decay_max: float = 0.08

    bound: float = 3.0             # tanh-style magnitude cap


@dataclass
class Pair3Config:
    """Long-term memory: sparse quantized key-value store.

    `dim` should be set to match Pair1Config.n_embd, same note as Pair2Config.
    """
    dim: int = 384
    codebook_size: int = 512
    similarity_threshold: float = 0.85   # snap-to-existing-entry threshold

    # Item 1: momentum-based surprise instead of a hard instantaneous cutoff.
    promotion_loss_threshold: float = 3.0    # accumulated-surprise trigger for promotion
    surprise_momentum_decay: float = 0.7     # how much running surprise persists per exposure

    repetition_min: int = 2               # times a concept must recur to count as reinforced
    capacity: int = 2000
