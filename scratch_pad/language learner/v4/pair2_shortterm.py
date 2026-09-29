"""Pair 2 — Short-Term Memory (v2).

Reads Pair 1's full hidden vectors (never modifies Pair 1's weights) and
maintains a fast-decaying trace via a local, no-backprop update -- see
architecture_design.md, Section 2 and 5.

v2 changes vs. v1 (see GAP_VS_TITANS.md, items 2 and 4):
  - Surprise (instantaneous co-activation) and momentum (accumulated
    direction) are tracked as two explicit terms instead of one blended
    Hebbian update.
  - Decay is adaptive: it scales between cfg.decay_min and cfg.decay_max
    based on how dissimilar the new input is to the current trace,
    instead of a single fixed constant -- a heuristic, non-learned stand-in
    for Titans' learned forgetting gate.
"""

import torch
import torch.nn.functional as F

from config import Pair2Config


class ShortTermMemory:
    def __init__(self, cfg: Pair2Config):
        self.cfg = cfg
        self.trace = torch.zeros(cfg.dim)      # combined short-term state (surprise + momentum)
        self.momentum = torch.zeros(cfg.dim)   # accumulated direction, separate from instant surprise
        self.last_vec = None

    def _adaptive_decay(self, hidden_vec: torch.Tensor) -> float:
        """Dissimilar input -> decay_max (forget faster, topic likely shifted).
        Similar/reinforcing input -> decay_min (forget slower, still relevant)."""
        if self.trace.norm().item() < 1e-8:
            return self.cfg.decay_min
        sim = F.cosine_similarity(self.trace.unsqueeze(0), hidden_vec.unsqueeze(0)).item()
        sim = max(-1.0, min(1.0, sim))
        # sim=1 (identical direction) -> decay_min; sim=-1 (opposite) -> decay_max
        weight = (1.0 - sim) / 2.0
        return self.cfg.decay_min + weight * (self.cfg.decay_max - self.cfg.decay_min)

    @torch.no_grad()
    def step(self, hidden_vec: torch.Tensor):
        """hidden_vec: a single (dim,) vector from Pair 1's output for the
        current token/position. Updates the short-term trace locally."""
        hidden_vec = hidden_vec.detach()

        # Instantaneous surprise: co-activation between the last vector and
        # this one (same Hebbian rule as v1, kept as its own term).
        surprise = torch.zeros_like(hidden_vec)
        if self.last_vec is not None:
            surprise = (self.last_vec + hidden_vec) / 2.0

        # Momentum: accumulated direction, decayed each step but never
        # reset -- this is what lets a sustained pattern build influence
        # even if no single exposure is individually large (GAP item 1's
        # sibling idea, applied here to the short-term side).
        self.momentum = self.cfg.momentum_decay * self.momentum + (1 - self.cfg.momentum_decay) * surprise

        decay = self._adaptive_decay(hidden_vec)
        self.trace = self.trace * (1.0 - decay) + self.cfg.surprise_lr * surprise + self.momentum * decay

        # bound the magnitude so the trace can't grow unboundedly
        self.trace = torch.tanh(self.trace / self.cfg.bound) * self.cfg.bound

        self.last_vec = hidden_vec
        return self.trace

    def current(self):
        return self.trace

    def reset(self):
        self.trace = torch.zeros(self.cfg.dim)
        self.momentum = torch.zeros(self.cfg.dim)
        self.last_vec = None

    def save(self, path: str):
        torch.save(
            {"trace": self.trace, "momentum": self.momentum, "last_vec": self.last_vec}, path
        )

    def load(self, path: str):
        data = torch.load(path, weights_only=True)
        self.trace = data["trace"]
        self.momentum = data["momentum"]
        self.last_vec = data["last_vec"]
