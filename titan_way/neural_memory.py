"""Mini-Titans neural long-term memory (Behrouz et al. 2024, sec. 3.1).

The memory M is a small MLP. Its WEIGHTS are the memory: they are updated
at test time by gradient descent on an associative loss
    l(M; x) = || M(k) - v ||^2 ,  k = W_K x,  v = W_V x
with
    surprise  = grad of l w.r.t. M's weights           (Eq. 8)
    S_t       = eta * S_{t-1} - theta * grad           (Eq. 10, momentum)
    M_t       = (1 - alpha) * M_{t-1} + S_t            (Eq. 13, forgetting)
Recall is a forward pass without update:  y = M(q), q = W_Q x   (Eq. 15).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    def __init__(self, in_dim, dim=64, depth=2, hidden=None, theta=0.1,
                 eta=0.8, alpha=0.01, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        hidden = hidden or dim * 2
        self.dim, self.depth = dim, depth
        self.theta, self.eta, self.alpha = theta, eta, alpha
        # projections (outer-loop parameters; fixed random here, trainable later)
        self.W_K = nn.Parameter(torch.randn(in_dim, dim, generator=g) / in_dim ** 0.5)
        self.W_V = nn.Parameter(torch.randn(in_dim, dim, generator=g) / in_dim ** 0.5)
        self.W_Q = nn.Parameter(self.W_K.detach().clone())  # tied to W_K
        # the memory MLP: weights live in a plain list, updated by hand
        sizes = [dim] + [hidden] * (depth - 1) + [dim]
        self.M = [torch.randn(a, b, generator=g) / a ** 0.5 for a, b in zip(sizes, sizes[1:])]
        self.S = [torch.zeros_like(w) for w in self.M]  # momentum ("past surprise")
        for p in (self.W_K, self.W_V, self.W_Q):
            p.requires_grad_(False)

    # ---- the memory network ------------------------------------------
    def _mlp(self, x, weights):
        for i, w in enumerate(weights):
            x = x @ w
            if i < len(weights) - 1:
                x = F.silu(x)
        return x

    def _kv(self, x):
        return F.normalize(x @ self.W_K, dim=-1), F.normalize(x @ self.W_V, dim=-1)

    # ---- write: learn at test time -------------------------------------
    def update(self, x, y=None, decay=True, momentum=True):
        """x: (n, in_dim) keys' source; y: optional values' source (default x,
        as in the paper). One gradient step on the batch. Returns mean surprise."""
        k, v = self._kv(x)
        if y is not None:
            v = F.normalize(y @ self.W_V, dim=-1)
        ws = [w.clone().requires_grad_(True) for w in self.M]
        loss = ((self._mlp(k, ws) - v) ** 2).sum(-1).mean()
        grads = torch.autograd.grad(loss, ws)
        eta = self.eta if momentum else 0.0
        a = self.alpha if decay else 0.0
        with torch.no_grad():
            for i, g in enumerate(grads):
                self.S[i] = eta * self.S[i] - self.theta * g
                self.M[i] = (1 - a) * self.M[i] + self.S[i]
        return loss.item()

    @torch.no_grad()
    def surprise(self, x):
        k, v = self._kv(x)
        return ((self._mlp(k, self.M) - v) ** 2).sum(-1)

    @torch.no_grad()
    def surprise_pair(self, x, y):
        """Surprise of a (key source, value source) pair, without updating."""
        k, v = self._kv(x)
        v = F.normalize(y @ self.W_V, dim=-1)
        return ((self._mlp(k, self.M) - v) ** 2).sum(-1)

    # ---- read: forward pass, no update ---------------------------------
    @torch.no_grad()
    def retrieve(self, x):
        q = F.normalize(x @ self.W_Q, dim=-1)
        return self._mlp(q, self.M)

    @torch.no_grad()
    def target(self, x):  # the value vector an input WOULD be stored as
        return F.normalize(x @ self.W_V, dim=-1)

    # ---- persistence: the memory IS weights, so save weights -----------
    def save(self, path):
        torch.save({"M": self.M, "S": self.S, "W_K": self.W_K, "W_V": self.W_V,
                    "W_Q": self.W_Q}, path)

    def load(self, path):
        d = torch.load(path)
        self.M, self.S = d["M"], d["S"]
        self.W_K.data, self.W_V.data, self.W_Q.data = d["W_K"], d["W_V"], d["W_Q"]
