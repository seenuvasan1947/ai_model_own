"""Pair 3 — Long-Term Memory.

Sparse quantized key-value store. Reads Pair 1's full 384-dim hidden
vectors; never modifies Pair 1's weights. Solves the "continuous vectors
don't share dictionary keys" problem via vector quantization (see
architecture_design.md, Section 6): a growing codebook snaps similar
vectors to the same discrete entry so repeated/related exposures
reinforce one entry instead of creating endless near-duplicates.

Promotion into this store is triggered by (see Section 7):
  1. surprise/salience  -- loss above a threshold at time of exposure
  2. repetition          -- a new exposure close enough to an existing
                             entry reinforces it instead of duplicating
"""

import torch
import torch.nn.functional as F

from config import Pair3Config


class LongTermStore:
    def __init__(self, cfg: Pair3Config):
        self.cfg = cfg
        self.codebook = torch.zeros(0, cfg.dim)   # quantized keys
        self.labels = []                           # associated token/text snippet per entry
        self.strength = torch.zeros(0)              # reinforcement count (repetition lever)

    def _nearest(self, vec: torch.Tensor):
        if len(self.codebook) == 0:
            return None, -1.0
        sims = F.cosine_similarity(self.codebook, vec.unsqueeze(0).expand(len(self.codebook), -1))
        best = sims.argmax().item()
        return best, sims[best].item()

    @torch.no_grad()
    def maybe_promote(self, vec: torch.Tensor, label: str, loss_value: float):
        """Call once per exposure. Promotes/reinforces based on surprise or
        repetition, per architecture_design.md Section 7."""
        vec = vec.detach()
        idx, sim = self._nearest(vec)

        # Repetition trigger: close enough to something already stored ->
        # reinforce that entry instead of adding a duplicate.
        if idx is not None and sim > self.cfg.similarity_threshold:
            self.strength[idx] += 1
            return idx, "reinforced"

        # Surprise/salience trigger: high loss at this exposure -> worth
        # keeping even on a single occurrence (flashbulb-memory analogy).
        if loss_value > self.cfg.promotion_loss_threshold:
            self._add(vec, label)
            return len(self.codebook) - 1, "promoted_by_surprise"

        return None, "not_promoted"

    def _add(self, vec: torch.Tensor, label: str):
        self.codebook = torch.cat([self.codebook, vec.unsqueeze(0)], dim=0)
        self.labels.append(label)
        self.strength = torch.cat([self.strength, torch.tensor([1.0])])

        if len(self.codebook) > self.cfg.capacity:
            weakest = self.strength.argmin().item()
            keep = [i for i in range(len(self.codebook)) if i != weakest]
            self.codebook = self.codebook[keep]
            self.strength = self.strength[keep]
            del self.labels[weakest]

    def retrieve(self, vec: torch.Tensor, k: int = 1):
        if len(self.codebook) == 0:
            return []
        sims = F.cosine_similarity(self.codebook, vec.unsqueeze(0).expand(len(self.codebook), -1))
        topk = sims.topk(min(k, len(sims))).indices.tolist()
        return [(self.labels[i], sims[i].item(), self.strength[i].item()) for i in topk]

    def __len__(self):
        return len(self.codebook)

    def save(self, path: str):
        torch.save(
            {"codebook": self.codebook, "labels": self.labels, "strength": self.strength}, path
        )

    def load(self, path: str):
        data = torch.load(path, weights_only=False)  # labels is a list[str], not a tensor
        self.codebook = data["codebook"]
        self.labels = data["labels"]
        self.strength = data["strength"]
