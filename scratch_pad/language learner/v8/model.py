"""v8 model: English section + knowledge section (topic experts), long-context attention
(local window + compressed memory), QK-norm, RoPE, GQA, SwiGLU, RMSNorm, fast memory, KV cache."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import Config


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(n, head_dim, base, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    f = torch.outer(torch.arange(n, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); cos/sin: (T, hd/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2).type_as(x)


# ---------------------------------------------------------------- attention
class Attention(nn.Module):
    """Local-window attention (this block + previous block). Global layers add a compressed
    branch: one summary key/value per comp_block bytes of all older text."""

    def __init__(self, cfg: Config, is_global: bool):
        super().__init__()
        self.cfg, self.is_global = cfg, is_global
        self.nh, self.nkv = cfg.n_head, cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.q = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.kv = nn.Linear(cfg.d_model, 2 * self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        if is_global:
            self.gate = nn.Parameter(torch.zeros(self.nh))

    def _proj(self, x):
        B, T, _ = x.shape
        q = self.q_norm(self.q(x).view(B, T, self.nh, self.hd)).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.nkv, self.hd).unbind(2)
        k = self.k_norm(k).transpose(1, 2)
        return q, k, v.transpose(1, 2)

    def _rep(self, t):
        return t.repeat_interleave(self.nh // self.nkv, 1)

    def forward(self, x, cos, sin, cache=None):
        if cache is not None and "k" in cache:
            return self._step(x, cos, sin, cache)
        B, T, _ = x.shape
        q, k, v = self._proj(x)
        qr, kr = apply_rope(q, cos[:T], sin[:T]), apply_rope(k, cos[:T], sin[:T])
        if cache is not None:  # prefill: keep keys/values for generation
            cache.update(k=k, kr=kr, v=v)
        y = self._local(qr, self._rep(kr), self._rep(v))
        if self.is_global:
            y = y + torch.sigmoid(self.gate)[None, :, None, None] * self._compressed(q, self._rep(k), self._rep(v))
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

    def _local(self, q, k, v):
        W = self.cfg.local_window
        B, H, T, hd = q.shape
        if T <= W:   # fits in one block: plain causal attention, no padding (same result)
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        pad = (-T) % W
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        nb = (T + pad) // W

        def blocks(t):
            return t.view(B, H, nb, W, hd)

        qb, kb, vb = blocks(q), blocks(k), blocks(v)
        kk = torch.cat([F.pad(kb, (0, 0, 0, 0, 1, 0))[:, :, :-1], kb], 3)   # previous block + this block
        vv = torch.cat([F.pad(vb, (0, 0, 0, 0, 1, 0))[:, :, :-1], vb], 3)
        mask = torch.ones(W, 2 * W, dtype=torch.bool, device=q.device)
        mask[:, W:] = torch.tril(mask[:, W:])
        mask = mask.expand(nb, W, 2 * W).clone()
        mask[0, :, :W] = False                                                 # block 0 has no previous block

        def flat(t):
            return t.transpose(1, 2).reshape(B * nb, H, t.shape[3], hd)

        y = F.scaled_dot_product_attention(flat(qb), flat(kk), flat(vv),
                                           attn_mask=mask.unsqueeze(1).repeat(B, 1, 1, 1))
        return y.view(B, nb, H, W, hd).transpose(1, 2).reshape(B, H, nb * W, hd)[:, :, :T]

    def _summaries(self, k, v, n):
        cb = self.cfg.comp_block
        B, H, _, hd = k.shape
        z = k.new_zeros(B, H, 1, hd)            # null slot so every query has something to attend to
        if n == 0:
            return z, z
        ks = k[:, :, :n * cb].reshape(B, H, n, cb, hd).mean(3)
        vs = v[:, :, :n * cb].reshape(B, H, n, cb, hd).mean(3)
        return torch.cat([z, ks], 2), torch.cat([z, vs], 2)

    def _compressed(self, q, k, v):
        cb, W = self.cfg.comp_block, self.cfg.local_window
        T = q.shape[2]
        n = T // cb
        ks, vs = self._summaries(k, v, n)
        pos = torch.arange(T, device=q.device)
        seen_until = ((pos // W) - 1) * W                                    # older than the local window
        j = torch.arange(n + 1, device=q.device)
        mask = (j[None, :] * cb <= seen_until[:, None]) & (j[None, :] > 0)
        mask[:, 0] = True
        return F.scaled_dot_product_attention(q, ks, vs, attn_mask=mask)

    def _step(self, x, cos, sin, cache):
        """Generate one new byte using the cache (same attention pattern as training)."""
        W, cb = self.cfg.local_window, self.cfg.comp_block
        t = cache["k"].shape[2]                                              # position of the new byte
        q, k, v = self._proj(x)
        qr, kr = apply_rope(q, cos[t:t + 1], sin[t:t + 1]), apply_rope(k, cos[t:t + 1], sin[t:t + 1])
        for name, new in (("k", k), ("kr", kr), ("v", v)):
            cache[name] = torch.cat([cache[name], new], 2)
        start = max(0, (t // W - 1) * W)
        y = F.scaled_dot_product_attention(qr, self._rep(cache["kr"][:, :, start:]), self._rep(cache["v"][:, :, start:]))
        if self.is_global:
            n = max(0, (t // W - 1) * W) // cb
            ks, vs = self._summaries(self._rep(cache["k"]), self._rep(cache["v"]), n)
            y = y + torch.sigmoid(self.gate)[None, :, None, None] * F.scaled_dot_product_attention(q, ks, vs)
        return self.o(y.transpose(1, 2).reshape(x.shape[0], 1, -1))


# ---------------------------------------------------------------- feed-forward
class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = 64 * ((int(8 * d / 3) + 63) // 64)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TopicMoE(nn.Module):
    """Top-1 experts routed by the running (causal) average of the text so far = a topic signal."""

    def __init__(self, cfg: Config, n_experts: int):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model) for _ in range(n_experts)])
        self.router = nn.Linear(cfg.d_model, n_experts, bias=False)
        self.register_buffer("usage", torch.zeros(n_experts), persistent=False)

    def forward(self, h, cache=None):
        B, T, d = h.shape
        csum = h.float().cumsum(1)
        if cache is not None:
            prev_sum, prev_n = cache.get("hsum"), cache.get("n", 0)
            if prev_sum is not None:
                csum = csum + prev_sum
            cache["hsum"], cache["n"] = csum[:, -1:], prev_n + T
            n0 = prev_n
        else:
            n0 = 0
        topic = csum / torch.arange(n0 + 1, n0 + T + 1, device=h.device)[None, :, None]
        probs = F.softmax(self.router(topic.to(h.dtype)).float(), -1)
        p, choice = probs.max(-1)
        weight = (p / p.detach()).to(h.dtype)                # =1 in value, still trains the router
        flat_h, flat_c, flat_w = h.reshape(-1, d), choice.reshape(-1), weight.reshape(-1)
        out = torch.zeros_like(flat_h)
        for e, expert in enumerate(self.experts):
            idx = (flat_c == e).nonzero(as_tuple=True)[0]
            if idx.numel():
                out[idx] = (expert(flat_h[idx]) * flat_w[idx, None]).to(out.dtype)
        with torch.no_grad():
            self.usage += torch.bincount(flat_c, minlength=len(self.experts)).float()
        n = len(self.experts)
        frac = torch.bincount(flat_c, minlength=n).float() / flat_c.numel()
        aux = n * (frac * probs.reshape(-1, n).mean(0)).sum() if n > 1 else h.new_zeros((), dtype=torch.float32)
        return out.view(B, T, d), aux


# ---------------------------------------------------------------- block / model
class Block(nn.Module):
    def __init__(self, cfg: Config, is_global: bool, n_experts: int = 0):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, is_global)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = TopicMoE(cfg, n_experts) if n_experts else SwiGLU(cfg.d_model)
        self.fast_A = nn.Parameter(torch.randn(cfg.fast_rank, cfg.d_model) * 0.02)
        self.fast_B = nn.Parameter(torch.zeros(cfg.d_model, cfg.fast_rank))

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, None if cache is None else cache.setdefault("attn", {}))
        h = self.norm2(x)
        if isinstance(self.ffn, TopicMoE):
            f, aux = self.ffn(h, None if cache is None else cache.setdefault("moe", {}))
        else:
            f, aux = self.ffn(h), x.new_zeros((), dtype=torch.float32)
        return x + f + (h @ self.fast_A.t()) @ self.fast_B.t(), aux

    @torch.no_grad()
    def make_identity(self):
        nn.init.zeros_(self.attn.o.weight)
        downs = [e.down for e in self.ffn.experts] if isinstance(self.ffn, TopicMoE) else [self.ffn.down]
        for d in downs:
            nn.init.zeros_(d.weight)


def _init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, std=0.02)


class LanguageModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lang = nn.ModuleList([Block(cfg, g) for g in cfg.lang_layers])
        self.know = nn.ModuleList([Block(cfg, g, n) for g, n in cfg.know_layers])
        self.norm = RMSNorm(cfg.d_model)
        self.apply(_init)
        cos, sin = rope_cache(cfg.max_ctx + 1, cfg.d_model // cfg.n_head, cfg.rope_base, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def blocks(self):
        return list(self.lang) + list(self.know)

    def forward(self, idx, targets=None, cache=None, english_only=False):
        x = self.emb(idx)
        aux_total = x.new_zeros((), dtype=torch.float32)
        for i, b in enumerate(list(self.lang) if english_only else self.blocks()):
            if cache is not None:
                x, aux = b(x, self.cos, self.sin, cache.setdefault(i, {}))
            elif self.training and self.cfg.grad_ckpt and idx.shape[1] > self.cfg.ckpt_min_len:
                x, aux = checkpoint(b, x, self.cos, self.sin, use_reentrant=False)
            else:
                x, aux = b(x, self.cos, self.sin)
            aux_total = aux_total + aux
        logits = self.norm(x) @ self.emb.weight.t()
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1)
        return logits, loss + self.cfg.moe_aux * aux_total

    # ---------------- growth (each keeps the model's output unchanged) ----------------
    def _new_block(self, is_global, n_experts=0):
        blk = Block(self.cfg, is_global, n_experts).to(self.emb.weight.device)
        blk.apply(_init)
        blk.make_identity()
        return blk

    @torch.no_grad()
    def grow_lang(self):
        g = not self.cfg.lang_layers[-1]
        self.lang.append(self._new_block(g))
        self.cfg.lang_layers.append(g)

    @torch.no_grad()
    def grow_know(self):
        g = not self.cfg.know_layers[-1][0]
        self.know.append(self._new_block(g, 1))
        self.cfg.know_layers.append([g, 1])

    @torch.no_grad()
    def split_expert(self, layer):
        """Copy the busiest expert of a knowledge layer; the two copies then specialise."""
        moe = self.know[layer].ffn
        busiest = int(moe.usage.argmax())
        new = SwiGLU(self.cfg.d_model).to(self.emb.weight.device)
        new.load_state_dict(moe.experts[busiest].state_dict())
        moe.experts.append(new)
        old_w = moe.router.weight
        router = nn.Linear(self.cfg.d_model, len(moe.experts), bias=False).to(old_w.device)
        router.weight[:-1] = old_w
        router.weight[-1] = old_w[busiest] + torch.randn_like(old_w[busiest]) * 0.01
        moe.router = router
        moe.usage = torch.zeros(len(moe.experts), device=old_w.device)
        self.cfg.know_layers[layer][1] += 1

    # ---------------- generation with KV cache ----------------
    @torch.no_grad()
    def generate(self, idx, max_new=200, temperature=0.8, top_k=20, stop_byte=10):
        idx = idx[:, -self.cfg.max_ctx + max_new:]
        cache = {}
        logits, _ = self(idx, cache=cache)                    # prefill: prompt processed once
        out = []
        for _ in range(max_new):
            last = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = last.argmax(-1, keepdim=True)
            else:
                last = last / temperature
                if top_k:
                    v, _ = torch.topk(last, top_k)
                    last[last < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(last, -1), 1)
            if stop_byte is not None and nxt.item() == stop_byte:
                break
            out.append(nxt.item())
            if idx.shape[1] + len(out) >= self.cfg.max_ctx:
                break
            logits, _ = self(nxt, cache=cache)                # one byte at a time
        return out

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def section_of(name):
    if "fast_" in name:
        return "fast_lang" if name.startswith("lang.") else "fast_know"
    return "know" if name.startswith("know.") else "lang"   # emb, lang.*, final norm = English section
