from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- transformer shape ---
    vocab_size: int = 256          # byte-level: any new word is letters it can already read
    d_model: int = 256
    n_head: int = 4
    n_kv_head: int = 2             # grouped-query attention
    rope_base: float = 100000.0    # larger base for long context
    qk_norm: bool = True
    fast_rank: int = 8             # fast-memory rank per layer

    # --- master (English) and slave (knowledge) ---
    # each entry is one layer: English -> [is_global]; knowledge -> [is_global, n_experts]
    lang_layers: list = field(default_factory=lambda: [False, True, False, True])
    know_layers: list = field(default_factory=lambda: [[False, 1], [True, 1]])
    know_tap: int = 2              # the doorway: knowledge reads/writes after this many English layers
    max_lang_layer: int = 8        # new English layers are added above the doorway
    max_total_layer: int = 30
    max_experts: int = 8
    moe_aux: float = 0.01

    # --- context / attention ---
    max_ctx: int = 30000
    train_ctx: int = 1024
    local_window: int = 512
    comp_block: int = 64
    grad_ckpt: bool = True
    ckpt_min_len: int = 2048

    # --- learning rates ---
    lr: float = 1e-3
    fast_lr: float = 1e-2
    warmup: int = 100
    low_lr_scale: float = 0.2      # master layers below the doorway learn slowly once knowledge exists
    practice_lr_scale: float = 0.3 # English practice from known-word sentences while reading subjects
    dict_lr_scale: float = 0.3     # dictionary lessons learn gently
    chat_lr_scale: float = 0.3
    batch_size: int = 32
    batch_tokens: int = 16384
    grad_clip: float = 1.0
    replay_ratio: float = 0.25     # revision share of every step

    # --- master -> slave linkage ---
    english_anchor_ratio: float = 0.25  # English lessons mixed into knowledge steps
    gate_penalty: float = 1.0      # knowledge gate pushed to 0 on plain English
    english_gate_ok: float = 0.05  # feed is done only when the gate on English is below this
    doorway_weight: float = 1.0    # doorway check: master keeps old understanding steady at the doorway
    refile_drift: float = 0.05     # doorway drift above this -> sleep re-reads old notes longer
    refile_steps: int = 100        # extra knowledge re-reading steps during such a sleep

    # --- plasticity ---
    strength_keep: float = 0.999
    resist: float = 4.0
    low_resist_mult: float = 3.0   # extra resistance below the doorway once knowledge exists
    fade: float = 1e-5
    fast_decay: float = 0.01

    # --- sleep / episodes / importance ---
    sleep_every: int = 500
    sleep_steps: int = 50
    min_repeats: int = 2
    forget_after: int = 5000
    max_episodes: int = 30000
    sleep_fast_keep: float = 0.5
    strong_weight: float = 0.5     # each strong word adds this much importance
    max_importance: float = 3.0    # cap; /important in chat gives the max

    # --- growth ---
    grow_window: int = 300
    grow_min_gain: float = 0.01

    # --- dictionary / words ---
    dict_path: str = ""
    strong_words_path: str = ""
    word_known: int = 5            # seen this many times = known
    dict_max_refs: int = 20

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
