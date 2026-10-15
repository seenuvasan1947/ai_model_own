"""Brain-like synapses for the slow weights:
- strength grows when a weight is repeatedly useful, slowly decays when unused
- strong weights resist being changed
- weak weights slowly fade toward zero
"""
import torch

from model import is_fast


def slow_params(model):
    return [(n, p) for n, p in model.named_parameters() if not is_fast(n)]


class Plasticity:
    def __init__(self, cfg):
        self.cfg = cfg
        self.strength = {}

    def snapshot(self, model):
        return {n: p.detach().clone() for n, p in slow_params(model)}

    def relative(self, st):
        return st / (st + st.mean() + 1e-12)

    @torch.no_grad()
    def after_step(self, model, before, lr_scale=1.0):
        """Call after optimizer.step() while grads are still present."""
        c = self.cfg
        for n, p in slow_params(model):
            st = self.strength.get(n)
            if st is None or st.shape != p.shape:
                st = self.strength[n] = torch.zeros_like(p, dtype=torch.float32)
            use = (p.grad * before[n]).abs().float() if p.grad is not None else torch.zeros_like(st)
            st.mul_(c.strength_keep).add_(use, alpha=1 - c.strength_keep)
            s = self.relative(st)
            # resistance: strong weights move less
            delta = (p - before[n]) * lr_scale
            p.copy_(before[n] + delta / (1 + c.resist * s))
            # fading: weak matrix weights shrink (norm gains stay)
            if p.dim() >= 2:
                p.mul_(1 - c.fade * (1 - s))

    def stats(self):
        if not self.strength:
            return {}
        allst = torch.cat([v.flatten() for v in self.strength.values()])
        rel = allst / (allst + allst.mean() + 1e-12)
        return {"strong_frac": (rel > 0.5).float().mean().item(), "mean_rel": rel.mean().item()}

    def state_dict(self):
        return {k: v.cpu() for k, v in self.strength.items()}

    def load_state_dict(self, d, device):
        self.strength = {k: v.to(device) for k, v in d.items()}
