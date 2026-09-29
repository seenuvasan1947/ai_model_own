"""Small, faithful Titans (Behrouz et al. 2024): Memory-as-Context (MAC).

Per segment of C tokens, in every block:
  1. retrieve   h = M*(q)                       (Eq. 15, 21)  from the neural memory
  2. attend     y = Attn([persistent ; h ; segment])         (Eq. 22-23)
  3. update     M <- update(M, y)               (Eq. 24) at TEST time too:
                surprise = grad of ||M(k)-v||^2, momentum, forgetting (Eq. 8-13)
  4. output     o = y (x) M*(y)                 (Eq. 25)
Outer loop (normal backprop) trains: embeddings, attention, persistent tokens,
W_K/W_V/W_Q, the memory's initial weights, and the data-dependent gates
alpha (forget), theta (step), eta (momentum). Gradients flow through the
inner-loop memory updates (second order), unlike titan_way/neural_memory.py.

Variants for comparison (mode=): "mac" (memory), "local" (same but NO memory:
attention only inside each segment), "full" (ordinary causal transformer over
the whole sequence).
Simplifications vs the paper: chunk-level gates (one alpha/theta/eta per
segment, from its mean), no 1-D convolution, no parallel scan, tiny sizes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def sinus(pos, d):
    i = torch.arange(0, d, 2).float()
    ang = pos[:, None].float() / (10000 ** (i / d))
    return torch.cat([ang.sin(), ang.cos()], -1)


def logit(p):
    return math.log(p / (1 - p))


def mem_forward(x, Ws):
    for i, W in enumerate(Ws):
        x = torch.bmm(x, W)
        if i < len(Ws) - 1:
            x = F.silu(x)
    return x


class NeuralMemory(nn.Module):
    def __init__(self, d, depth=2, alpha0=0.01, theta0=0.1, eta0=0.8):
        super().__init__()
        self.W0 = nn.ParameterList([nn.Parameter(torch.randn(d, d) / d ** 0.5 * 0.5) for _ in range(depth)])
        self.Wk, self.Wv, self.Wq = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.gate = nn.Linear(d, 3)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias.copy_(torch.tensor([logit(alpha0), logit(theta0), logit(eta0)]))

    def init_state(self, B):
        Ws = [w[None].expand(B, -1, -1).contiguous() for w in self.W0]
        return Ws, [torch.zeros_like(w) for w in Ws]

    def retrieve(self, state, x):
        q = F.normalize(self.Wq(x), dim=-1)
        return mem_forward(q, state[0])

    def update(self, state, x):
        Ws, S = state
        if not Ws[0].requires_grad:
            Ws = [w.detach().requires_grad_(True) for w in Ws]
        k, v = F.normalize(self.Wk(x), dim=-1), F.normalize(self.Wv(x), dim=-1)
        with torch.enable_grad():
            loss = ((mem_forward(k, Ws) - v) ** 2).sum(-1).mean(-1).sum()   # per-sequence memories
            grads = torch.autograd.grad(loss, Ws, create_graph=self.training)
        g = torch.sigmoid(self.gate(x.mean(1)))
        a, th, et = (g[:, i, None, None] for i in range(3))
        S2 = [et * s - th * gr for s, gr in zip(S, grads)]
        W2 = [(1 - a) * w + s for w, s in zip(Ws, S2)]
        if not self.training:
            W2, S2 = [w.detach() for w in W2], [s.detach() for s in S2]
        return W2, S2


class Block(nn.Module):
    def __init__(self, d, heads, C, Np, use_memory, theta0=0.1):
        super().__init__()
        self.C, self.Np, self.use_memory = C, Np, use_memory
        self.P = nn.Parameter(torch.randn(Np, d) * 0.02)          # persistent memory
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.mem = NeuralMemory(d, theta0=theta0) if use_memory else None
        nm = C if use_memory else 0
        T = Np + nm + C
        mask = torch.ones(T, T, dtype=torch.bool)                 # True = blocked
        mask[:, :Np + nm] = False                                 # everyone sees persistent + memory tokens
        mask[Np + nm:, Np + nm:] = torch.triu(torch.ones(C, C, dtype=torch.bool), 1)   # segment is causal
        self.register_buffer("mask", mask)

    def forward(self, x, state, update=True):
        B = x.size(0)
        parts = [self.P[None].expand(B, -1, -1)]
        if self.use_memory:
            parts.append(self.mem.retrieve(state, x))
        z = torch.cat(parts + [x], 1)
        n = self.ln1(z)
        with sdpa_kernel(SDPBackend.MATH):   # flash kernels have no second-order gradient (needed by the memory)
            y = z + self.attn(n, n, n, attn_mask=self.mask, need_weights=False)[0]
        y = y[:, -self.C:]
        y = y + self.ff(self.ln2(y))
        if self.use_memory:
            if update:
                state = self.mem.update(state, y)                 # learn at test time
            y = y * torch.sigmoid(self.mem.retrieve(state, y))    # o = y (x) M*(y)
        return y, state


class Titans(nn.Module):
    def __init__(self, vocab, mode="mac", d=64, layers=2, heads=4, C=16, Np=4, theta0=0.1):
        super().__init__()
        self.mode, self.C, self.d = mode, C, d
        self.emb = nn.Embedding(vocab, d)
        if mode == "full":
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.blocks = nn.ModuleList([Block(d, heads, C, Np, mode == "mac", theta0) for _ in range(layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, tok, update=True):
        B, T = tok.shape
        x = self.emb(tok)
        if self.mode == "full":
            x = x + sinus(torch.arange(T), self.d)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            return self.head(self.core(x, mask=mask))
        assert T % self.C == 0
        pos = sinus(torch.arange(self.C), self.d)
        states = [b.mem.init_state(B) if b.use_memory else None for b in self.blocks]
        outs = []
        for s in range(0, T, self.C):
            h = x[:, s:s + self.C] + pos
            for j, b in enumerate(self.blocks):
                h, states[j] = b(h, states[j], update)
            outs.append(h)
        return self.head(torch.cat(outs, 1))
