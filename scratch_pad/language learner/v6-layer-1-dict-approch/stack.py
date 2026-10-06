"""v5 vertical stack: Layer 1 (frozen transformer) sits on top, Layer 2
(short-term) below it, Layer 3 (long-term) at the bottom. Hidden states
from Layer 1 flow down into L2; promoted L2 patterns flow down into L3.
During generation, activation from L2 and L3 flows back up into Layer 1's
residual stream, biasing generation.

See V5_DESIGN.md for the architecture and reasoning.
"""

import os

import torch
import torch.nn.functional as F

from config import Layer1Config, Layer2Config, Layer3Config, StackConfig
from layer1_language import Pair1Foundation as Layer1Foundation
from memory_layer import MemoryLayer


L2_STATE_PATH = "out/layer2_state.pt"
L3_STATE_PATH = "out/layer3_state.pt"


class MemoryStack:
    """Wires Layer 1 -> Layer 2 -> Layer 3 (downward) and L2+L3 -> Layer 1
    residual (upward). Layer 1 is a frozen module owned externally; the
    stack composes memory around it, never modifies it.
    """

    def __init__(self,
                 layer1: Layer1Foundation,
                 l2_cfg: Layer2Config | None = None,
                 l3_cfg: Layer3Config | None = None,
                 stack_cfg: StackConfig | None = None):
        self.layer1 = layer1
        self.l2_cfg = l2_cfg or Layer2Config()
        self.l3_cfg = l3_cfg or Layer3Config()
        self.stack_cfg = stack_cfg or StackConfig()

        self.layer2 = MemoryLayer(self.l2_cfg, "L2-short-term")
        self.layer3 = MemoryLayer(self.l3_cfg, "L3-long-term")

    # ------------------------------------------------------------------
    # Downward: process a hidden vector through both memory layers,
    # updating their state. Called during reading (continual_read.py).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def observe(self, hidden_vec: torch.Tensor):
        """Feed one Layer 1 hidden vector down through L2 (which may grow
        and update connections) and then, for the subset that's already
        well-connected in L2, into L3."""
        hidden_vec = hidden_vec.detach()

        # L2 processes the raw hidden vector
        l2_activations = self.layer2.step(hidden_vec)

        # L3 only receives activations that are already reinforced in L2
        promoted = self.layer2.promoted_activations(
            l2_activations,
            min_connection=self.l3_cfg.promotion_activation,
        )
        if promoted:
            # For L3, we still need a query vector -- use the mean of the
            # preferred vectors of the promoted L2 units, so L3's own
            # units live in the same embedding space.
            promoted_vec = torch.zeros(self.l3_cfg.dim)
            weight_sum = 0.0
            for idx, act in promoted.items():
                promoted_vec = promoted_vec + act * self.layer2.units[idx].preferred_vector
                weight_sum += act
            promoted_vec = promoted_vec / max(weight_sum, 1e-8)
            self.layer3.step(promoted_vec)

    # ------------------------------------------------------------------
    # Upward: given a query hidden vector, compute the memory contribution
    # to add to Layer 1's residual stream. Used during generation.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def recall_vector(self, hidden_vec: torch.Tensor) -> torch.Tensor:
        """Query L2 and L3 with this hidden vector, project the combined
        recall pattern back into Layer 1's embedding space, and return
        the vector to add to Layer 1's residual stream.

        Read-only: memory state is not updated by recall.
        """
        hidden_vec = hidden_vec.detach()

        l2_act = self.layer2.recall(hidden_vec)
        l2_vec = self.layer2.project(l2_act)

        # L3 gets queried with a vector shaped by L2's recall -- so what L3
        # contributes depends on which L2 patterns fired. If L2 recalled
        # nothing (empty store), L3 sees the raw hidden vector.
        l3_query = l2_vec if l2_act else hidden_vec
        l3_act = self.layer3.recall(l3_query)
        l3_vec = self.layer3.project(l3_act)

        # Combine. L2 (short-term / recent) and L3 (long-term / consolidated)
        # both contribute; L3 slightly favored since it represents deeper
        # knowledge.
        combined = 0.5 * l2_vec + 0.5 * l3_vec

        return self.stack_cfg.memory_feedback_strength * combined

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, out_dir: str = "out"):
        os.makedirs(out_dir, exist_ok=True)
        self.layer2.save(os.path.join(out_dir, "layer2_state.pt"))
        self.layer3.save(os.path.join(out_dir, "layer3_state.pt"))

    def load(self, out_dir: str = "out"):
        l2p = os.path.join(out_dir, "layer2_state.pt")
        l3p = os.path.join(out_dir, "layer3_state.pt")
        if os.path.exists(l2p):
            self.layer2.load(l2p)
        if os.path.exists(l3p):
            self.layer3.load(l3p)

    def summary(self) -> dict:
        return {
            "layer2": self.layer2.summary(),
            "layer3": self.layer3.summary(),
        }
