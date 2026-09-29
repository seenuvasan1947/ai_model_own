"""Pair 3 — Long-Term Memory (v2).

Sparse quantized key-value store. Reads Pair 1's full hidden vectors;
never modifies Pair 1's weights. Solves the "continuous vectors don't
share dictionary keys" problem via vector quantization -- see
architecture_design.md, Section 6.

v2 change vs. v1 (see GAP_VS_TITANS.md, item 1): promotion is driven by a
decaying running average of surprise, not a hard instantaneous cutoff --
so a sustained pattern of moderately-surprising exposures can accumulate
enough momentum to get promoted, even if no single exposure alone crossed
the threshold. This is a heuristic, non-gradient stand-in for Titans'
momentum-based surprise term.
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
        self.running_surprise = 0.0                 # item 1: momentum on surprise, not instantaneous-only

    def _nearest(self, vec: torch.Tensor):
        if len(self.codebook) == 0:
            return None, -1.0
        sims = F.cosine_similarity(self.codebook, vec.unsqueeze(0).expand(len(self.codebook), -1))
        best = sims.argmax().item()
        return best, sims[best].item()

    @torch.no_grad()
    def maybe_promote(self, vec: torch.Tensor, label: str, loss_value: float):
        """Call once per exposure. Promotes/reinforces based on accumulated
        surprise or repetition, per architecture_design.md Section 7 and
        GAP_VS_TITANS.md item 1."""
        vec = vec.detach()
        idx, sim = self._nearest(vec)

        # Repetition trigger: close enough to something already stored ->
        # reinforce that entry instead of adding a duplicate.
        if idx is not None and sim > self.cfg.similarity_threshold:
            self.strength[idx] += 1
            # a reinforced concept is not "surprising" -- let running
            # surprise relax back down rather than staying inflated
            self.running_surprise *= self.cfg.surprise_momentum_decay
            return idx, "reinforced"

        # Momentum on surprise: blend this exposure's loss into a running
        # average instead of judging it in isolation. A moderately
        # surprising exposure that keeps recurring can cross the threshold
        # even if no single occurrence was extreme enough alone.
        self.running_surprise = (
            self.cfg.surprise_momentum_decay * self.running_surprise
            + (1 - self.cfg.surprise_momentum_decay) * loss_value
        )

        if self.running_surprise > self.cfg.promotion_loss_threshold:
            self._add(vec, label)
            self.running_surprise *= 0.5  # partial reset after acting on it
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

    def retrieve_vecs(self, vec: torch.Tensor, k: int = 1, min_sim: float = 0.3):
        """v2, item 3 (GAP_VS_TITANS.md): return the actual stored vectors
        (not just text labels) for direct injection into Pair 1 via
        forward_with_memory -- avoids the lossy vector-to-text round trip."""
        if len(self.codebook) == 0:
            return torch.zeros(0, self.cfg.dim), []
        sims = F.cosine_similarity(self.codebook, vec.unsqueeze(0).expand(len(self.codebook), -1))
        topk = sims.topk(min(k, len(sims))).indices.tolist()
        topk = [i for i in topk if sims[i].item() > min_sim]
        if not topk:
            return torch.zeros(0, self.cfg.dim), []
        vecs = self.codebook[topk]
        info = [(self.labels[i], sims[i].item()) for i in topk]
        return vecs, info

    def __len__(self):
        return len(self.codebook)

    def save(self, path: str):
        torch.save(
            {
                "codebook": self.codebook,
                "labels": self.labels,
                "strength": self.strength,
                "running_surprise": self.running_surprise,
            },
            path,
        )

    def load(self, path: str):
        data = torch.load(path, weights_only=False)  # labels is a list[str], not a tensor
        self.codebook = data["codebook"]
        self.labels = data["labels"]
        self.strength = data["strength"]
        self.running_surprise = data.get("running_surprise", 0.0)
