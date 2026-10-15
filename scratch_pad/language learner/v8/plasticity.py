"""Brain-like synapses for the slow weights that are currently learning:
- strength grows when a weight is repeatedly useful, slowly decays when unused
- strong weights resist being changed
- weak weights slowly fade toward zero
Frozen weights (requires_grad=False) are skipped: they neither learn nor fade.
"""
import torch

from model import section_of


def learning_slow_params(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and not section_of(n).startswith("fast")]


class Plasticity:
    def __init__(self, cfg):
        self.cfg = cfg
        self.strength = {}

    def snapshot(self, model):
        return {n: p.detach().clone() for n, p in learning_slow_params(model)}

    @staticmethod
    def relative(st):
        return st / (st + st.mean() + 1e-12)

    @torch.no_grad()
    def after_step(self, model, before):
        c = self.cfg
        for n, p in learning_slow_params(model):
            if n not in before:
                continue
            st = self.strength.get(n)
            if st is None or st.shape != p.shape:
                st = self.strength[n] = torch.zeros_like(p, dtype=torch.float32)
            use = (p.grad * before[n]).abs().float() if p.grad is not None else torch.zeros_like(st)
            st.mul_(c.strength_keep).add_(use, alpha=1 - c.strength_keep)
            s = self.relative(st)
            p.copy_(before[n] + (p - before[n]) / (1 + c.resist * s))
            if p.dim() >= 2:
                p.mul_(1 - c.fade * (1 - s))

    def stats(self):
        if not self.strength:
            return {}
        allst = torch.cat([v.flatten() for v in self.strength.values()])
        rel = allst / (allst + allst.mean() + 1e-12)
        return {"strong_frac": round((rel > 0.5).float().mean().item(), 3), "mean_rel": round(rel.mean().item(), 3)}

    def state_dict(self):
        return {k: v.cpu() for k, v in self.strength.items()}

    def load_state_dict(self, d, device):
        self.strength = {k: v.to(device) for k, v in d.items()}
