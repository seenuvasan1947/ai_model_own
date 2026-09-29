"""Decoder-only transformer with a Mixture-of-Experts feedforward block.

The MoE-FFN replaces the usual dense feedforward layer so that separate
"expert" sub-spaces exist in the architecture from the start (see
prototype_spec.md, Section 2) even though Phase 1 trains everything with
plain backprop, same as a normal transformer.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
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


class Expert(nn.Module):
    """A single feedforward expert (standard transformer MLP block)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.gelu(self.fc(x))
        x = self.proj(x)
        return self.dropout(x)


class MoEFeedForward(nn.Module):
    """Top-1 routed mixture of experts. Each token is sent to exactly one expert.

    This gives a separable, inspectable "sub-space per concept-type" structure
    (see prototype_spec.md) that a later continual-learning phase can attach
    localized, per-expert updates to.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.n_experts = config.n_experts

    def forward(self, x):
        B, T, C = x.shape
        logits = self.router(x)                       # (B, T, n_experts)
        weights = F.softmax(logits, dim=-1)
        top1 = torch.argmax(weights, dim=-1)            # (B, T)

        out = torch.zeros_like(x)
        flat_x = x.view(-1, C)
        flat_top1 = top1.view(-1)
        flat_out = out.view(-1, C)
        flat_gate = torch.gather(weights.view(-1, self.n_experts), 1, flat_top1.unsqueeze(-1)).squeeze(-1)

        for expert_id, expert in enumerate(self.experts):
            token_mask = flat_top1 == expert_id
            if token_mask.any():
                selected = flat_x[token_mask]
                expert_out = expert(selected)
                flat_out[token_mask] = expert_out * flat_gate[token_mask].unsqueeze(-1)

        return flat_out.view(B, T, C), top1  # return routing decisions for inspection/logging


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.moe = MoEFeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        moe_out, routing = self.moe(self.ln2(x))
        x = x + moe_out
        return x, routing


class BrainInspiredGPT(nn.Module):
    def __init__(self, config: GPTConfig):
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.context_size, "sequence longer than context_size"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        all_routing = []
        for block in self.blocks:
            x, routing = block(x)
            all_routing.append(routing)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, all_routing

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
