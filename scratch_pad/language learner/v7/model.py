"""Decoder-only transformer: byte-level, RoPE, GQA, SwiGLU, RMSNorm,
plus a small fast-memory path per layer and identity-preserving depth growth."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(seq_len, head_dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd)
    T = x.shape[2]
    c, s = cos[:T], sin[:T]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.nh, self.nkv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.nkv, self.hd).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = 64 * ((int(8 * d / 3) + 63) // 64)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model)
        # fast memory: low-rank path that learns quickly and fades quickly
        self.fast_A = nn.Parameter(torch.randn(cfg.fast_rank, cfg.d_model) * 0.02)
        self.fast_B = nn.Parameter(torch.zeros(cfg.d_model, cfg.fast_rank))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        h = self.norm2(x)
        return x + self.mlp(h) + (h @ self.fast_A.t()) @ self.fast_B.t()


class LanguageModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)
        self.set_seq_len(cfg.seq_len)

    def set_seq_len(self, seq_len):
        """Change the reading window; weights do not depend on it."""
        self.cfg.seq_len = seq_len
        dev = self.emb.weight.device
        cos, sin = rope_cache(seq_len, self.cfg.d_model // self.cfg.n_head, self.cfg.rope_base, dev)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        logits = self.norm(x) @ self.emb.weight.t()   # tied output head
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def grow_depth(self):
        """Add a layer at the top that starts as identity, so nothing already learned changes."""
        dev = self.emb.weight.device
        blk = Block(self.cfg).to(dev)
        blk.apply(self._init)
        nn.init.zeros_(blk.attn.o.weight)
        nn.init.zeros_(blk.mlp.down.weight)
        self.blocks.append(blk)
        self.cfg.n_layer += 1

    @torch.no_grad()
    def generate(self, idx, max_new=80, temperature=0.8, top_k=20, stop_byte=10):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.seq_len:])
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            if stop_byte is not None and nxt.item() == stop_byte:
                break
            idx = torch.cat([idx, nxt], 1)
        return idx

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def is_fast(name):
    return "fast_" in name
