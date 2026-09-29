"""Phase 2: local, forward-only continual learning + usage-based decay.

This is the piece that was missing from Phase 1 (train.py, which only does
standard backprop). It implements, on top of an already-trained
BrainInspiredGPT (see model.py):

  1. Router freezing      -> routing structure stops changing (stable "anatomy",
                              same reasoning as why the brain's regional wiring
                              is settled early and then just gets tuned).
  2. Local Hebbian update  -> for each routed expert, nudge its weights using
                              only that expert's own pre/post activation
                              (no backward pass, no global error signal).
  3. Usage-based decay     -> connections/experts that go unused for a while
                              shrink toward zero, freeing "capacity" the same
                              way unused cortical territory gets reclaimed
                              (the cross-modal plasticity discussion).

This is intentionally the simplest possible version of each mechanism, meant
to be inspectable and testable, not a state-of-the-art implementation of
Forward-Forward / EWC / fast-weights. See prototype_spec.md Section 3 (M2).
"""

import torch
import torch.nn.functional as F

from model import BrainInspiredGPT


class ContinualLearner:
    def __init__(self, model: BrainInspiredGPT, lr: float = 1e-3, decay: float = 1e-4):
        self.model = model
        self.lr = lr
        self.decay = decay

        # freeze the router in every block: routing "anatomy" is now fixed,
        # only expert weights update from here on.
        for block in self.model.blocks:
            for p in block.moe.router.parameters():
                p.requires_grad = False

        # usage counters per expert, per block -- drives the decay step.
        self.usage = [
            torch.zeros(block.moe.n_experts) for block in self.model.blocks
        ]

    @torch.no_grad()
    def step(self, idx: torch.Tensor):
        """One forward-only local-update step on a single input sequence.

        No target/label needed, no backward pass. Each token's activations
        are used as their own local learning signal (Hebbian-style: units
        that were co-active get their connecting weights nudged, everything
        else is untouched).
        """
        self.model.eval()
        _, _, all_routing = self.model(idx)

        x = self.model.tok_emb(idx) + self.model.pos_emb(
            torch.arange(idx.size(1), device=idx.device).unsqueeze(0)
        )

        for block, routing, usage in zip(self.model.blocks, all_routing, self.usage):
            pre = block.ln2(x + block.attn(block.ln1(x)))  # input to the MoE layer
            flat_pre = pre.view(-1, pre.size(-1))
            flat_routing = routing.view(-1)

            for expert_id, expert in enumerate(block.moe.experts):
                token_mask = flat_routing == expert_id
                n_selected = int(token_mask.sum().item())
                if n_selected == 0:
                    continue

                usage[expert_id] += n_selected

                selected = flat_pre[token_mask]              # (n, C) "pre-synaptic" activity
                post = F.gelu(expert.fc(selected))            # "post-synaptic" activity

                # Hebbian-style local update: strengthen fc weights in proportion
                # to co-activation of input and output units (outer product),
                # averaged over the selected tokens. No error signal involved.
                hebb = post.t() @ selected / max(n_selected, 1)  # (4C, C)
                expert.fc.weight += self.lr * hebb

            x, _ = block(x)  # advance to the next block's input using the (just-updated) weights

    @torch.no_grad()
    def decay_unused(self, min_usage: float = 1.0):
        """Shrink weights of experts that saw little/no usage since the last call.

        This is the "unused neurons decay, space gets reclaimed" mechanism.
        Reset usage counters afterward so decay reflects usage since the last
        decay pass, not all-time usage.
        """
        for block, usage in zip(self.model.blocks, self.usage):
            for expert_id, expert in enumerate(block.moe.experts):
                if usage[expert_id].item() < min_usage:
                    expert.fc.weight *= (1.0 - self.decay)
                    expert.proj.weight *= (1.0 - self.decay)
            usage.zero_()
