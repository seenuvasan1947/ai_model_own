"""v5 config: Layer 1 (frozen transformer) + Layer 2 (short-term memory)
+ Layer 3 (long-term memory), vertical stack. See V5_DESIGN.md."""

from dataclasses import dataclass


@dataclass
class Layer1Config:
    """Language substrate. Same standard transformer as v4's Pair1."""
    vocab_size: int = 6000
    context_size: int = 64
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True


@dataclass
class TrainConfig:
    book_path: str = "data/combined_corpus.txt"
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
    device: str = "cuda"
    val_split: float = 0.1


@dataclass
class Layer2Config:
    """Short-term memory: sparse growing units, distributed activation.

    Layer 2 starts with zero units. A hidden vector from Layer 1 either
    reinforces existing units (when at least one unit's similarity clears
    novelty_threshold) or spawns a new unit (when no existing unit matches
    well enough). Connections between co-activated units strengthen; unused
    connections decay and prune; long-unused units fade and are removed.

    See V5_DESIGN.md for the full mechanism.
    """
    dim: int = 128                    # matches Layer1Config.n_embd

    # Unit creation / activation
    novelty_threshold: float = 0.75   # if max similarity to existing units < this, spawn a new unit
    activity_floor: float = 0.30      # units with similarity below this are considered inactive this step
    max_units: int = 2000              # hard cap to prevent runaway growth
    unit_lifetime_steps: int = 400     # units inactive for this many steps fade and are removed

    # Connection strengthening / decay
    strengthen_rate: float = 0.15      # how much co-activation adds to a connection
    decay_rate: float = 0.01           # per-step decay of every connection
    connection_floor: float = 0.02     # connections below this are pruned


@dataclass
class Layer3Config:
    """Long-term memory. Same mechanism as Layer 2, slower rates. Only
    receives activation from units that already have strong connections
    in Layer 2 -- so promotion is emergent (repeatedly reinforced L2
    patterns cross the promotion_activation threshold and drive L3), not
    hand-coded."""
    dim: int = 128

    novelty_threshold: float = 0.80
    activity_floor: float = 0.30
    max_units: int = 5000
    unit_lifetime_steps: int = 50_000   # near-permanent once formed

    strengthen_rate: float = 0.02       # ~10x slower than L2
    decay_rate: float = 0.0005          # ~20x slower than L2
    connection_floor: float = 0.005

    # Minimum connection strength in L2 required to feed activation down to L3.
    # This is what makes L3 emergent-consolidation: only well-reinforced L2
    # patterns ever reach L3.
    promotion_activation: float = 0.5


@dataclass
class StackConfig:
    """Wiring between layers. See stack.py."""
    # How strongly memory activity (L2 + L3 recall) biases Layer 1's residual
    # stream during generation. 0.0 = memory has no effect on generation
    # (bypass mode, useful for ablation). 1.0 = full mix.
    memory_feedback_strength: float = 0.5
