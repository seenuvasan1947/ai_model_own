"""dict-approach config: Layer 1 trained ONLY on a closed-vocabulary
dictionary + grammar + numbers + punctuation corpus (no TinyStories, no
Wikipedia, no narrative/encyclopedic content at all). Everything else
(world facts, named entities, domain content) is meant to be fed later
through Layer 2/3 via continual_read.py, never baked into Layer 1's
frozen weights. See data_prep/gen_dict_vocab.py and gen_synthetic.py.

Architecture (layer counts, dims) is intentionally unchanged from v6 --
only the TRAINING DATA and the sizes that must track its much smaller
scale (vocab_size, max_iters) differ."""

from dataclasses import dataclass


@dataclass
class Layer1Config:
    """Language substrate. Much smaller closed vocabulary than v6 because
    the corpus itself is a small, deliberately controlled word set
    (gen_dict_vocab.WORDS + gen_synthetic's closed sets), not bulk prose."""
    vocab_size: int = 1500              # dict-approach: small closed-vocab corpus (v6 used 10000);
                                         # train_layer1.py overrides this with the tokenizer's actual
                                         # size anyway (measured ~1300 for the current word list)
    context_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True


def _default_sources():
    # Must match data_prep/build_corpus.py's SOURCES.
    return [
        "data/synthetic.txt",     # numbers/arithmetic/calendar/opposites/grammar patterns
        "data/dict_vocab.txt",    # closed-vocabulary dictionary definitions + usage + punctuation
    ]


@dataclass
class TrainConfig:
    # dict-approach: corpus is ONLY synthetic.txt + dict_vocab.txt -- no
    # TinyStories, no Wikipedia. Built by data_prep/build_corpus.py.
    book_path: str = "data/foundation_corpus.txt"
    book_source_paths: list = None  # set in __post_init__ to _default_sources()
    qa_path: str = "data/qa_behavior.txt"
    tokenizer_path: str = "data/tokenizer.json"
    out_dir: str = "out"
    batch_size: int = 16
    max_iters: int = 8000                # dict-approach: much smaller corpus than v6's 70K-iter run
    eval_interval: int = 200
    eval_iters: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"
    val_split: float = 0.1              # dict-approach: small corpus, hold out more for a stable val signal

    # early stopping: save best-val-loss checkpoint as layer1_final.pt;
    # stop if no improvement for `patience` eval checkpoints. Bumped from
    # 5 -> 8: with this small a corpus the val split is tiny and noisy,
    # so a single bad checkpoint shouldn't end the run.
    patience: int = 8

    def __post_init__(self):
        if self.book_source_paths is None:
            self.book_source_paths = _default_sources()


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
