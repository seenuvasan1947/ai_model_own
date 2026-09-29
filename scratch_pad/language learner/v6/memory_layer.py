"""Shared implementation for Layer 2 (short-term) and Layer 3 (long-term).

Both layers use the same mechanism:
  - Sparse growing set of units. Each unit has a preferred vector.
  - When a hidden vector arrives, activate units by cosine similarity.
  - If no existing unit clears novelty_threshold, spawn a new unit.
  - Every pair of co-activated units strengthens its connection.
  - Every connection decays each step; below-floor connections prune.
  - Units inactive for too long fade and are removed.

Layer 2 and Layer 3 differ only in the rate constants (see config.py).

See V5_DESIGN.md for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class _Unit:
    preferred_vector: torch.Tensor   # (dim,) L2-normalized
    last_active_step: int
    total_activations: float = 0.0    # cumulative activation over life; useful for inspection


class MemoryLayer:
    def __init__(self, cfg, name: str):
        """cfg: a Layer2Config or Layer3Config (see config.py). They share
        the same field names; this class doesn't care which."""
        self.cfg = cfg
        self.name = name
        self.units: List[_Unit] = []
        # Sparse connection graph: (i, j) with i < j -> strength.
        # Connections are undirected; we always store with i < j to avoid
        # duplicating entries.
        self.connections: Dict[Tuple[int, int], float] = {}
        self.step_count = 0

    # ------------------------------------------------------------------
    # Activation: convert a hidden vector into a pattern of unit activations
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _activate(self, hidden_vec: torch.Tensor) -> Dict[int, float]:
        """Compute activation of every existing unit for this hidden vector.
        Returns {unit_index: activation_strength} for units above activity_floor.
        Distributed activation -- many units can activate partially at once."""
        hidden_vec = F.normalize(hidden_vec, dim=0)
        activations: Dict[int, float] = {}
        for i, unit in enumerate(self.units):
            sim = float(torch.dot(hidden_vec, unit.preferred_vector))
            if sim > self.cfg.activity_floor:
                activations[i] = sim
        return activations

    # ------------------------------------------------------------------
    # Growing: create a new unit when nothing existing matches well enough
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _grow(self, hidden_vec: torch.Tensor) -> int:
        """Create a new unit whose preferred vector is this hidden vector."""
        if len(self.units) >= self.cfg.max_units:
            # capacity hit -- evict the unit with the lowest total_activations
            # (least-used, most disposable) to free room for something new
            worst = min(range(len(self.units)),
                        key=lambda k: self.units[k].total_activations)
            self._remove_unit(worst)
        self.units.append(_Unit(
            preferred_vector=F.normalize(hidden_vec, dim=0).clone(),
            last_active_step=self.step_count,
            total_activations=1.0,
        ))
        return len(self.units) - 1

    def _remove_unit(self, idx: int):
        """Remove a unit and all its connections. Reindexes remaining units
        (list order shifts), so connection keys also need shifting."""
        del self.units[idx]
        # Rebuild connections with new indices
        new_connections: Dict[Tuple[int, int], float] = {}
        for (i, j), s in self.connections.items():
            if i == idx or j == idx:
                continue
            ni = i - 1 if i > idx else i
            nj = j - 1 if j > idx else j
            new_connections[(ni, nj)] = s
        self.connections = new_connections

    # ------------------------------------------------------------------
    # Connection updates: strengthen co-activated, decay everything, prune
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_connections(self, activations: Dict[int, float]):
        """Hebbian: co-activated units strengthen their connection."""
        active_ids = list(activations.keys())
        for a in range(len(active_ids)):
            for b in range(a + 1, len(active_ids)):
                i, j = sorted((active_ids[a], active_ids[b]))
                delta = self.cfg.strengthen_rate * activations[i] * activations[j]
                self.connections[(i, j)] = self.connections.get((i, j), 0.0) + delta

    @torch.no_grad()
    def _decay_and_prune(self):
        """Every connection decays this step; ones below floor disappear."""
        to_remove = []
        for key in self.connections:
            self.connections[key] *= (1.0 - self.cfg.decay_rate)
            if self.connections[key] < self.cfg.connection_floor:
                to_remove.append(key)
        for key in to_remove:
            del self.connections[key]

    @torch.no_grad()
    def _fade_stale_units(self):
        """Units that haven't been active for a long time fade away."""
        alive = self.step_count - self.cfg.unit_lifetime_steps
        # Iterate in reverse so removals don't shift subsequent indices mid-loop
        for idx in range(len(self.units) - 1, -1, -1):
            if self.units[idx].last_active_step < alive:
                self._remove_unit(idx)

    # ------------------------------------------------------------------
    # Public step: process one hidden vector, updating memory
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, hidden_vec: torch.Tensor,
             pre_activations: Dict[int, float] | None = None
             ) -> Dict[int, float]:
        """Process one hidden vector. Updates layer state and returns the
        activation pattern this vector produced.

        `pre_activations`: if provided, use this activation dict instead of
        computing one from `hidden_vec`. Used by Layer 3, which receives
        activations from Layer 2 rather than computing its own from Layer 1.
        In that case `hidden_vec` is still needed for potential unit growth
        (it becomes the preferred vector of any new unit created).
        """
        self.step_count += 1
        hidden_vec = hidden_vec.detach()

        # 1. Activation: which existing units respond to this input?
        if pre_activations is not None:
            activations = dict(pre_activations)
        else:
            activations = self._activate(hidden_vec)

        # 2. Growth: if nothing matched well enough, spawn a new unit
        if len(activations) == 0 or max(activations.values()) < self.cfg.novelty_threshold:
            new_idx = self._grow(hidden_vec)
            activations[new_idx] = 1.0

        # 3. Bookkeeping on each activated unit
        for idx, act in activations.items():
            self.units[idx].last_active_step = self.step_count
            self.units[idx].total_activations += act

        # 4. Connection updates (Hebbian, then decay+prune)
        self._update_connections(activations)
        self._decay_and_prune()

        # 5. Periodic unit fade check (expensive; not every step)
        if self.step_count % 100 == 0:
            self._fade_stale_units()

        return activations

    # ------------------------------------------------------------------
    # Recall: given a query hidden vector, activate units + spread through
    # existing connections. Used to shape generation.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def recall(self, hidden_vec: torch.Tensor) -> Dict[int, float]:
        """Read-only: what pattern does this query activate, including
        associative spread through existing connections? Does NOT update
        state -- pure retrieval."""
        # Step 1: direct activation from the query
        activations = self._activate(hidden_vec)

        # Step 2: associative spread -- for each activated unit, connected
        # units get partial activation proportional to connection strength.
        # Only one hop of spread (keeps this bounded and predictable).
        spread: Dict[int, float] = {}
        for i, act_i in activations.items():
            for (a, b), strength in self.connections.items():
                if a == i:
                    other = b
                elif b == i:
                    other = a
                else:
                    continue
                # spread activation, capped by min(act_i, strength)
                contribution = min(act_i, strength) * 0.5
                spread[other] = max(spread.get(other, 0.0), contribution)

        # Merge direct + spread; direct wins on overlaps
        for k, v in spread.items():
            if k not in activations:
                activations[k] = v
        return activations

    # ------------------------------------------------------------------
    # Projection back up to Layer 1's residual stream
    # ------------------------------------------------------------------
    @torch.no_grad()
    def project(self, activations: Dict[int, float]) -> torch.Tensor:
        """Combine activated units' preferred vectors, weighted by activation,
        into a single vector in Layer 1's embedding space. Empty -> zero
        vector (memory has nothing to say)."""
        if not activations:
            return torch.zeros(self.cfg.dim)
        total = torch.zeros(self.cfg.dim)
        weight_sum = 0.0
        for idx, act in activations.items():
            total = total + act * self.units[idx].preferred_vector
            weight_sum += act
        return total / max(weight_sum, 1e-8)

    # ------------------------------------------------------------------
    # Get promotion-eligible activations for feeding into a lower layer.
    # Used by Layer 2 -> Layer 3.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def promoted_activations(self, activations: Dict[int, float],
                              min_connection: float) -> Dict[int, float]:
        """Return the subset of activations whose unit is involved in at
        least one connection above `min_connection`. Only these get to
        drive the lower layer."""
        result: Dict[int, float] = {}
        for idx, act in activations.items():
            for (a, b), strength in self.connections.items():
                if (a == idx or b == idx) and strength >= min_connection:
                    result[idx] = act
                    break
        return result

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "name": self.name,
            "units": len(self.units),
            "connections": len(self.connections),
            "steps": self.step_count,
            "avg_connection_strength": (
                sum(self.connections.values()) / len(self.connections)
                if self.connections else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        if self.units:
            preferred = torch.stack([u.preferred_vector for u in self.units])
        else:
            preferred = torch.zeros(0, self.cfg.dim)
        state = {
            "preferred_vectors": preferred,
            "last_active_step": [u.last_active_step for u in self.units],
            "total_activations": [u.total_activations for u in self.units],
            "connections": self.connections,
            "step_count": self.step_count,
        }
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, weights_only=False)
        preferred = state["preferred_vectors"]
        last = state["last_active_step"]
        totals = state["total_activations"]
        self.units = [
            _Unit(preferred_vector=preferred[i],
                  last_active_step=last[i],
                  total_activations=totals[i])
            for i in range(len(last))
        ]
        self.connections = state["connections"]
        self.step_count = state["step_count"]
