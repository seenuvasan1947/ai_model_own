from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- transformer shape ---
    vocab_size: int = 256          # byte-level
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2             # grouped-query attention
    seq_len: int = 128
    rope_base: float = 10000.0
    fast_rank: int = 8             # rank of the fast-memory weights per layer

    # --- growth ---
    max_layer: int = 12            # RTX 2050 4 GB limit at d_model up to 768
    grow_window: int = 300         # steps per loss window when checking for a plateau
    grow_min_gain: float = 0.01    # grow if loss improved less than 1% between windows

    # --- learning ---
    lr: float = 1e-3               # slow memory (cortex)
    fast_lr: float = 1e-2          # fast memory (hippocampus)
    chat_lr_scale: float = 0.3     # learn more gently from chat input
    batch_size: int = 32
    grad_clip: float = 1.0
    replay_ratio: float = 0.25     # share of each batch that is revision
    grammar_ratio: float = 0.15    # share of each batch that is grammar practice (base English)

    # --- plasticity (strength / resistance / fade) ---
    strength_keep: float = 0.999   # how slowly strength is lost when unused
    resist: float = 4.0            # how hard strong weights resist change
    fade: float = 1e-5             # per-step fading of weak weights
    fast_decay: float = 0.01       # per-step fading of fast memory

    # --- sleep / episodic memory ---
    sleep_every: int = 500
    sleep_steps: int = 50
    min_repeats: int = 2           # episodes seen at least this often get consolidated
    forget_after: int = 5000       # drop single-seen episodes older than this many steps
    max_episodes: int = 20000
    sleep_fast_keep: float = 0.5   # fast memory shrinks to this after sleep

    # --- curriculum ---
    pass_score: float = 0.9
    keep_score: float = 0.8        # earlier classes must stay at least this good to move on
    eval_every: int = 200
    quiz_size: int = 40

    def to_dict(self):
        return asdict(self)
