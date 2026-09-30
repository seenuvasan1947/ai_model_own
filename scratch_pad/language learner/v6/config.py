"""v6 config: Layer 1 (frozen transformer, larger + trained on domain-neutral
English) + Layer 2 (short-term memory) + Layer 3 (long-term memory), vertical
stack. See V6_CHANGES.md and V5_DESIGN.md (architecture unchanged from v5)."""

from dataclasses import dataclass


@dataclass
class Layer1Config:
    """Language substrate. v6 scales this up appropriately: a proper English
    substrate needs enough capacity to actually generalize across corpora,
    not just memorize one book. See V6_CHANGES.md."""
    vocab_size: int = 10000            # v6: bigger corpus needs bigger vocab (was 6000)
    context_size: int = 256             # v6: 64 was too short for real prose (was 64)
    n_layer: int = 6                    # v6: was 3
    n_head: int = 8                     # v6: was 4
    n_embd: int = 256                   # v6: was 128
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    # v6: unified corpus of TinyStories + Simple Wikipedia + synthetic +
    # QA-behavior, built by data_prep/build_corpus.py. See V6_CHANGES.md.
    book_path: str = "data/foundation_corpus.txt"
    qa_path: str = "data/qa_behavior.txt"
    tokenizer_path: str = "data/tokenizer.json"
    out_dir: str = "out"
    batch_size: int = 16                # v6: was 8
    max_iters: int = 70000              # v6 corpus-bump: 40K -> 70K, matches ~45M-token corpus
    eval_interval: int = 1000           # v6: bumped so eval overhead stays small over longer run
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"
    val_split: float = 0.05             # v6: 5% of a ~15M-token corpus is still 750K tokens

    # v6 NEW: early stopping. Save the best-val-loss checkpoint as
    # layer1_final.pt; stop training if no improvement for `patience`
    # eval checkpoints.
    patience: int = 5


@dataclass
class Layer2Config:
    """Short-term memory: sparse growing units, distributed activation.
    v6: `dim` bumped to match Layer1Config.n_embd=256. Rate constants unchanged.
    See V5_DESIGN.md for the mechanism."""
    dim: int = 256                     # v6: matches Layer1Config.n_embd (was 128)

    # Unit creation / activation
    novelty_threshold: float = 0.75
    activity_floor: float = 0.30
    max_units: int = 4000               # v6: doubled (was 2000) -- bigger substrate warrants bigger memory
    unit_lifetime_steps: int = 400

    # Connection strengthening / decay
    strengthen_rate: float = 0.15
    decay_rate: float = 0.01
    connection_floor: float = 0.02


@dataclass
class Layer3Config:
    """Long-term memory. Same mechanism as Layer 2, slower rates. Only
    receives activation from units that already have strong connections
    in Layer 2 -- so promotion is emergent, not hand-coded."""
    dim: int = 256                      # v6: matches Layer1Config.n_embd

    novelty_threshold: float = 0.80
    activity_floor: float = 0.30
    max_units: int = 10000              # v6: doubled (was 5000)
    unit_lifetime_steps: int = 50_000

    strengthen_rate: float = 0.02
    decay_rate: float = 0.0005
    connection_floor: float = 0.005

    promotion_activation: float = 0.5


@dataclass
class StackConfig:
    """Wiring between layers. See stack.py."""
    memory_feedback_strength: float = 0.5
