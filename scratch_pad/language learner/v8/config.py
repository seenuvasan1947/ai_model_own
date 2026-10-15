from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- transformer shape ---
    vocab_size: int = 256          # byte-level
    d_model: int = 256
    n_head: int = 4
    n_kv_head: int = 2             # grouped-query attention
    rope_base: float = 100000.0    # larger base for long context
    qk_norm: bool = True
    fast_rank: int = 8             # fast-memory rank per layer

    # --- sections ---
    # each entry is one layer: English section -> [is_global]; knowledge section -> [is_global, n_experts]
    lang_layers: list = field(default_factory=lambda: [False, True, False, True])
    know_layers: list = field(default_factory=lambda: [[False, 1], [True, 1]])
    max_lang_layer: int = 8
    max_total_layer: int = 30
    max_experts: int = 8
    moe_aux: float = 0.01          # load-balance loss weight

    # --- context / attention ---
    max_ctx: int = 30000           # longest text it can read
    train_ctx: int = 1024          # lesson window for normal learning
    local_window: int = 512        # exact attention: this block + the previous one
    comp_block: int = 64           # global layers: one summary per 64 bytes of older text
    grad_ckpt: bool = True
    ckpt_min_len: int = 2048       # only recompute layers for long texts (saves time on short lessons)

    # --- learning ---
    lr: float = 1e-3
    fast_lr: float = 1e-2
    warmup: int = 100              # steps of warmup (also after every growth)
    lang_lr_normal: float = 0.0    # English-section lr scale during feed/chat (0 = frozen)
    chat_lr_scale: float = 0.3
    batch_size: int = 32
    batch_tokens: int = 16384      # cap on bytes per step (long documents use fewer rows)
    grad_clip: float = 1.0
    replay_ratio: float = 0.25
    grammar_ratio: float = 0.25    # English anchor during knowledge learning (half learned English lessons, half grammar)
    anchor_kl: float = 5.0         # how strongly knowledge layers must keep the English section's output on English text
    english_gap_ok: float = 0.05   # feed is done only when English output drift is below this

    # --- plasticity ---
    strength_keep: float = 0.999
    resist: float = 4.0
    fade: float = 1e-5
    fast_decay: float = 0.01

    # --- sleep / episodes ---
    sleep_every: int = 500
    sleep_steps: int = 50
    min_repeats: int = 2
    forget_after: int = 5000
    max_episodes: int = 30000
    sleep_fast_keep: float = 0.5

    # --- growth ---
    grow_window: int = 300
    grow_min_gain: float = 0.01

    # --- dictionary ---
    dict_path: str = ""
    word_known: int = 5            # seen this many times = known, no more lookups
    dict_max_refs: int = 20        # max lookups per word

    # --- curriculum ---
    pass_score: float = 0.9
    keep_score: float = 0.8
    eval_every: int = 200
    quiz_size: int = 40

    @property
    def n_layer(self):
        return len(self.lang_layers) + len(self.know_layers)

    def to_dict(self):
        return asdict(self)
