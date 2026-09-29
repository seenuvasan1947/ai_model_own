"""Pair 1 — Language Foundation.

Standard decoder-only transformer, GELU activation, trained once with
backprop on raw book text + a small generic Q&A-behavior dataset (see
architecture_design.md, Section 2). This is the only backprop-trained
module in the design; Pair 2 and Pair 3 (pair2_shortterm.py,
pair3_longterm.py) read this module's output vectors but never modify it.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Pair1Config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Pair1Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.context_size, config.context_size))
        self.register_buffer("mask", mask.view(1, 1, config.context_size, config.context_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """Standard dense feedforward block with GELU (see piece_collect.md)."""

    def __init__(self, config: Pair1Config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: Pair1Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Pair1Foundation(nn.Module):
    """Decoder-only transformer. `forward(..., return_hidden=True)` exposes
    the final hidden-state vectors (before the output head) so Pair 2/Pair 3
    can read the full 384-dim representation, per architecture_design.md."""

    def __init__(self, config: Pair1Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.context_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        assert T <= self.config.context_size, "sequence longer than context_size"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        hidden = self.ln_f(x)          # (B, T, n_embd) -- this is what Pair 2/3 read
        logits = self.head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def forward_with_memory(self, idx, memory_vecs=None, targets=None):
        """v2, item 3 (GAP_VS_TITANS.md): "Memory as Context" -- splice
        Pair 2/3's vectors directly into the hidden-state sequence as extra
        leading positions, instead of decoding them to text and pasting
        them into the prompt. The causal mask still applies, so real tokens
        can attend back to these memory positions but not vice versa.

        `memory_vecs`: (M, n_embd) tensor, or None. Pair 1's weights are
        never modified by this -- it's a forward-pass-only extension of
        the input sequence, consistent with the frozen-foundation design.
        """
        B, T = idx.shape

        if memory_vecs is not None and memory_vecs.size(0) > 0:
            M = memory_vecs.size(0)
            assert M + T <= self.config.context_size, (
                f"memory ({M}) + tokens ({T}) exceeds context_size ({self.config.context_size})"
            )
            mem = memory_vecs.unsqueeze(0).expand(B, M, self.config.n_embd)
            mem_pos = torch.arange(0, M, device=idx.device).unsqueeze(0)
            mem_x = mem + self.pos_emb(mem_pos)

            tok_pos = torch.arange(M, M + T, device=idx.device).unsqueeze(0)
            tok_x = self.tok_emb(idx) + self.pos_emb(tok_pos)

            x = self.drop(torch.cat([mem_x, tok_x], dim=1))
        else:
            M = 0
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        hidden = self.ln_f(x)
        logits = self.head(hidden)

        loss = None
        if targets is not None:
            logits_tok = logits[:, M:, :]  # loss only over the real-token part
            loss = F.cross_entropy(
                logits_tok.reshape(-1, logits_tok.size(-1)), targets.reshape(-1), ignore_index=-1
            )

        return logits, loss, M

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, memory_vecs=None):
        """memory_vecs (v2): optional (M, n_embd) tensor spliced in as extra
        leading context on every step, via forward_with_memory."""
        m = memory_vecs.size(0) if memory_vecs is not None else 0
        budget = self.config.context_size - m

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -budget:]
            logits, _, _ = self.forward_with_memory(idx_cond, memory_vecs=memory_vecs)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
