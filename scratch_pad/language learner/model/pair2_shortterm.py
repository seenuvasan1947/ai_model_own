"""Pair 2 — Short-Term Memory.

Reads Pair 1's full 384-dim hidden vectors (never modifies Pair 1's weights)
and maintains a fast-decaying trace via a local Hebbian-style update:
no backward pass, no global error signal -- only "these vectors were
co-active, strengthen the association between them" (see
architecture_design.md, Section 2 and 5).
"""

import torch

from config import Pair2Config


class ShortTermMemory:
    def __init__(self, cfg: Pair2Config):
        self.cfg = cfg
        self.trace = torch.zeros(cfg.dim)   # the current short-term state
        self.last_vec = None

    @torch.no_grad()
    def step(self, hidden_vec: torch.Tensor):
        """hidden_vec: a single (dim,) vector from Pair 1's output for the
        current token/position. Updates the short-term trace locally."""
        hidden_vec = hidden_vec.detach()

        if self.last_vec is not None:
            # Hebbian co-activation: reinforce the trace in the direction
            # shared by the last vector and the current one.
            co_activation = (self.last_vec + hidden_vec) / 2.0
            self.trace = self.trace + self.cfg.lr * co_activation

        # decay -- "use it or lose it" (see architecture_design.md, Section 8
        # notes on the missing three-factor upgrade: this is plain Hebbian,
        # not yet reward/surprise-modulated)
        self.trace = self.trace * (1.0 - self.cfg.decay)

        # bound the magnitude so the trace can't grow unboundedly
        self.trace = torch.tanh(self.trace / self.cfg.bound) * self.cfg.bound

        self.last_vec = hidden_vec
        return self.trace

    def current(self):
        return self.trace

    def reset(self):
        self.trace = torch.zeros(self.cfg.dim)
        self.last_vec = None

    def save(self, path: str):
        torch.save({"trace": self.trace, "last_vec": self.last_vec}, path)

    def load(self, path: str):
        data = torch.load(path, weights_only=True)
        self.trace = data["trace"]
        self.last_vec = data["last_vec"]
