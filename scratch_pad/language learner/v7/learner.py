"""The always-on learner: every input is learned from, then (optionally) answered."""
import os
import random

import torch

from config import Config
from curriculum import grammar_lesson
from memory import Episodes
from model import LanguageModel, is_fast
from plasticity import Plasticity

NL = 10  # newline byte separates episodes


class Learner:
    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LanguageModel(cfg).to(self.device)
        self.plast = Plasticity(cfg)
        self.episodes = Episodes(cfg)
        self.step = 0
        self.class_idx = 0
        self.losses = []
        self.use_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        self._make_opt()

    def _make_opt(self):
        slow = [p for n, p in self.model.named_parameters() if not is_fast(n)]
        fast = [p for n, p in self.model.named_parameters() if is_fast(n)]
        self.opt = torch.optim.AdamW([
            {"params": slow, "lr": self.cfg.lr, "base_lr": self.cfg.lr},
            {"params": fast, "lr": self.cfg.fast_lr, "base_lr": self.cfg.fast_lr},
        ], weight_decay=0.0, betas=(0.9, 0.95))

    # ---------------- data ----------------
    def _rows(self, texts):
        L = self.cfg.seq_len + 1
        rows = []
        for t in texts:
            data = b"\n" + t.encode("utf-8", errors="replace") + b"\n"
            for i in range(0, max(1, len(data) - 1), L - 1):
                rows.append(data[i:i + L])
        x = torch.zeros(len(rows), L - 1, dtype=torch.long)
        y = torch.full((len(rows), L - 1), -1, dtype=torch.long)
        for r, d in enumerate(rows):
            d = torch.tensor(list(d), dtype=torch.long)
            x[r, :len(d) - 1] = d[:-1]
            y[r, :len(d) - 1] = d[1:]
        return x.to(self.device), y.to(self.device)

    # ---------------- learning ----------------
    def learn(self, texts, lr_scale=1.0, remember=True, replay=True, train_fast=True):
        """One learning step on new texts (+ revision of old episodes)."""
        texts = [t for t in texts if t.strip()]
        if not texts:
            return None
        batch = list(texts)
        if replay and len(self.episodes):
            k = max(1, int(len(texts) * self.cfg.replay_ratio / (1 - self.cfg.replay_ratio)))
            batch += self.episodes.sample(k)
        if replay and self.cfg.grammar_ratio > 0:
            # grammar anchor: English is the base, so every step also practises grammar
            k = max(1, int(len(texts) * self.cfg.grammar_ratio / (1 - self.cfg.grammar_ratio)))
            batch += [grammar_lesson() for _ in range(k)]
        loss = self._update(batch, lr_scale, train_fast)
        if remember:
            for t in texts:
                self.episodes.add(t, self.step)
        self.step += 1
        self.losses.append(loss)
        if self.step % self.cfg.sleep_every == 0:
            self.sleep()
        self.maybe_grow()
        return loss

    def _update(self, batch, lr_scale, train_fast):
        self.model.train()
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * lr_scale
        if not train_fast:
            self.opt.param_groups[1]["lr"] = 0.0
        x, y = self._rows(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            _, loss = self.model(x, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        before = self.plast.snapshot(self.model)
        self.opt.step()
        self.plast.after_step(self.model, before)
        with torch.no_grad():
            for b in self.model.blocks:
                b.fast_B.mul_(1 - self.cfg.fast_decay)
        return loss.item()

    def sleep(self):
        """Consolidate repeated episodes into slow memory; forget one-off details; shrink fast memory."""
        consolidated = 0
        for _ in range(self.cfg.sleep_steps):
            batch = self.episodes.sample(self.cfg.batch_size, min_count=self.cfg.min_repeats)
            if not batch:
                break
            self._update(batch, lr_scale=0.5, train_fast=False)
            consolidated += len(batch)
        with torch.no_grad():
            for b in self.model.blocks:
                b.fast_B.mul_(self.cfg.sleep_fast_keep)
        forgotten = self.episodes.prune(self.step)
        return {"consolidated": consolidated, "forgotten": forgotten, "episodes": len(self.episodes)}

    def maybe_grow(self):
        w = self.cfg.grow_window
        if self.cfg.n_layer >= self.cfg.max_layer or len(self.losses) < 2 * w or self.step % w:
            return False
        prev = sum(self.losses[-2 * w:-w]) / w
        cur = sum(self.losses[-w:]) / w
        if (prev - cur) / max(prev, 1e-8) >= self.cfg.grow_min_gain:
            return False
        self.model.grow_depth()
        self._make_opt()
        self.losses.clear()
        print(f"  [grow] learning stalled (loss {cur:.3f}) -> now {self.cfg.n_layer} layers, "
              f"{self.model.n_params() / 1e6:.2f}M params")
        return True

    # ---------------- answering ----------------
    @torch.no_grad()
    def generate(self, prompt, max_new=80, temperature=0.8):
        self.model.eval()
        ids = torch.tensor([[NL] + list(prompt.encode("utf-8", errors="replace"))], device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.model.generate(ids, max_new=max_new, temperature=temperature)
        return bytes(out[0, ids.shape[1]:].tolist()).decode("utf-8", errors="replace")

    def answer(self, prompt, learn=True, temperature=0.8):
        """Same process at 'test time': learn from the input first, then reply."""
        if learn:
            self.learn([prompt], lr_scale=self.cfg.chat_lr_scale)
        return self.generate(prompt, temperature=temperature)

    def quiz(self, quiz_fn, n):
        correct = 0
        for _ in range(n):
            p, a = quiz_fn()
            got = self.generate(p, max_new=len(a) + 4, temperature=0)
            correct += got.strip() == a
        return correct / n

    # ---------------- checkpoints ----------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "cfg": self.cfg.to_dict(),
            "model": self.model.state_dict(),
            "strength": self.plast.state_dict(),
            "episodes": self.episodes.state_dict(),
            "step": self.step,
            "class_idx": self.class_idx,
        }, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self = cls(Config(**ck["cfg"]), device)
        self.model.load_state_dict(ck["model"])
        self.plast.load_state_dict(ck["strength"], self.device)
        self.episodes.load_state_dict(ck["episodes"])
        self.step, self.class_idx = ck["step"], ck["class_idx"]
        return self


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
