"""Always-on learner with two sections:
  kind="en"   -> trains the English section only (en-train, dictionary lessons)
  kind="know" -> trains the knowledge section (feed, chat); English section frozen by default
"""
import os
import random

import torch

from config import Config
from curriculum import grammar_lesson
from dictionary import Dictionary, WordTracker
from memory import Episodes
from model import LanguageModel, section_of
from plasticity import Plasticity

NL = 10


class Learner:
    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LanguageModel(cfg).to(self.device)
        self.plast = Plasticity(cfg)
        self.episodes = Episodes(cfg)
        self.words = WordTracker(cfg)
        self.dictionary = Dictionary(cfg.dict_path)
        self.step = 0
        self.class_idx = 0
        self.losses = {"en": [], "know": []}
        self.last_gap = 0.0
        self.use_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        self._make_opt()

    # ---------------- optimizer / modes ----------------
    def _make_opt(self):
        groups = {"lang": [], "know": [], "fast_lang": [], "fast_know": []}
        for n, p in self.model.named_parameters():
            groups[section_of(n)].append(p)
        self.opt = torch.optim.AdamW([{"params": ps, "name": k} for k, ps in groups.items()],
                                     weight_decay=0.0, betas=(0.9, 0.95))
        self.opt_born = self.step     # warmup restarts after every growth

    def _set_mode(self, kind, lr_scale):
        c = self.cfg
        warm = min(1.0, (self.step - self.opt_born + 1) / c.warmup)
        if kind == "en":
            scales = {"lang": 1.0, "know": 0.0, "fast_lang": 1.0, "fast_know": 0.0}
        else:
            scales = {"lang": c.lang_lr_normal, "know": 1.0, "fast_lang": c.lang_lr_normal, "fast_know": 1.0}
        for g in self.opt.param_groups:
            base = c.fast_lr if g["name"].startswith("fast") else c.lr
            g["lr"] = base * scales[g["name"]] * lr_scale * warm
            for p in g["params"]:
                p.requires_grad_(scales[g["name"]] > 0)

    def set_dictionary(self, path):
        self.cfg.dict_path = path
        self.dictionary = Dictionary(path)

    # ---------------- data ----------------
    def _rows(self, texts):
        L = self.cfg.max_ctx
        rows = []
        for t in texts:
            data = b"\n" + t.encode("utf-8", errors="replace") + b"\n"
            for i in range(0, max(1, len(data) - 1), L):
                rows.append(data[i:i + L + 1])
        T = max(len(r) for r in rows) - 1
        x = torch.zeros(len(rows), T, dtype=torch.long)
        y = torch.full((len(rows), T), -1, dtype=torch.long)
        for r, d in enumerate(rows):
            d = torch.tensor(list(d), dtype=torch.long)
            x[r, :len(d) - 1] = d[:-1]
            y[r, :len(d) - 1] = d[1:]
        return x.to(self.device), y.to(self.device)

    def _update(self, batch, kind, lr_scale=1.0, train_fast=True, anchor=()):
        """anchor: English lessons. In knowledge mode the full model must give the same output on
        them as the English section alone (teacher), so knowledge layers can't bend the English."""
        self.model.train()
        self._set_mode(kind, lr_scale)
        if not train_fast:
            for g in self.opt.param_groups:
                if g["name"].startswith("fast"):
                    g["lr"] = 0.0
        # keep each step under the byte budget (long documents -> fewer rows)
        longest = max(len(t) for t in list(batch) + list(anchor)) + 2
        batch = batch[:max(1, self.cfg.batch_tokens // longest)]
        x, y = self._rows(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            _, loss = self.model(x, y, english_only=(kind == "en"))   # English section learns standalone
            lesson_loss = loss.item()                                  # what feed reports / stops on
            if anchor:
                ax, ay = self._rows(list(anchor))
                with torch.no_grad():
                    teacher, _ = self.model(ax, english_only=True)
                student, a_loss = self.model(ax, ay)
                keep = ay.reshape(-1) >= 0
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(student.reshape(-1, 256)[keep].float(), -1),
                    torch.log_softmax(teacher.reshape(-1, 256)[keep].float(), -1),
                    log_target=True, reduction="batchmean")
                loss = loss + a_loss + self.cfg.anchor_kl * kl
                self.last_gap = kl.item()      # how far English output drifted from the English section
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.cfg.grad_clip)
        before = self.plast.snapshot(self.model)
        self.opt.step()
        self.plast.after_step(self.model, before)
        with torch.no_grad():
            for b in self.model.blocks():
                if b.fast_B.requires_grad:      # only the section that is learning
                    b.fast_B.mul_(1 - self.cfg.fast_decay)
        return lesson_loss

    # ---------------- learning ----------------
    def learn(self, texts, kind="know", lr_scale=1.0, remember=True, replay=True):
        texts = [t for t in texts if t.strip()]
        if not texts:
            return None
        batch = list(texts)
        n = len(texts)
        anchor = []
        if replay:
            k = max(1, int(n * self.cfg.replay_ratio / (1 - self.cfg.replay_ratio)))
            batch += self.episodes.sample(k, kind)
            if kind == "know" and self.cfg.grammar_ratio > 0:
                # English anchor: knowledge layers sit on top of the English section, so on English
                # lessons they must keep the English section's output unchanged
                k = max(1, int(n * self.cfg.grammar_ratio / (1 - self.cfg.grammar_ratio)))
                anchor = self.episodes.sample(k // 2, "en", min_count=self.cfg.min_repeats)
                anchor += [grammar_lesson() for _ in range(k - len(anchor))]
        loss = self._update(batch, kind, lr_scale, anchor=anchor)
        self.losses[kind].append(loss)
        self._dictionary_step(texts)
        if remember:
            for t in texts:
                self.episodes.add(t, self.step, kind)
        self.step += 1
        if self.step % self.cfg.sleep_every == 0:
            self.sleep()
        self.maybe_grow(kind)
        return loss

    def _dictionary_step(self, texts):
        """New words found -> look them up and learn their meaning in the English section."""
        if not len(self.dictionary):
            for t in texts:
                self.words.observe(t)
            return
        new = []
        for t in texts:
            new += self.words.unknown(t, self.dictionary)
            self.words.observe(t)
        new = list(dict.fromkeys(new))
        if new:
            lessons = self.words.lookup_lessons(new, self.dictionary)
            # revise old English lessons at the same time, so new words don't push out old ones
            k = max(4, int(len(lessons) * self.cfg.replay_ratio / (1 - self.cfg.replay_ratio)) * 3)
            self._update(lessons + self.episodes.sample(k, "en", min_count=self.cfg.min_repeats), "en")
            for les in lessons:
                self.episodes.add(les, self.step, "en")

    def sleep(self):
        consolidated = 0
        for kind in ("en", "know"):
            for _ in range(self.cfg.sleep_steps // 2):
                batch = self.episodes.sample(self.cfg.batch_size, kind, min_count=self.cfg.min_repeats)
                if not batch:
                    break
                self._update(batch, kind, lr_scale=0.5, train_fast=False)
                consolidated += len(batch)
        with torch.no_grad():
            for b in self.model.blocks():
                b.fast_B.mul_(self.cfg.sleep_fast_keep)
        return {"consolidated": consolidated, "forgotten": self.episodes.prune(self.step),
                "episodes": len(self.episodes)}

    def maybe_grow(self, kind):
        c, w, losses = self.cfg, self.cfg.grow_window, self.losses[kind]
        if len(losses) < 2 * w or len(losses) % w:
            return False
        prev, cur = sum(losses[-2 * w:-w]) / w, sum(losses[-w:]) / w
        if (prev - cur) / max(prev, 1e-8) >= c.grow_min_gain:
            return False
        m = self.model
        if kind == "en":
            if len(c.lang_layers) >= c.max_lang_layer or c.n_layer >= c.max_total_layer:
                return False
            m.grow_lang()
            what = f"English section -> {len(c.lang_layers)} layers"
        else:
            layer = next((i for i, (_, n) in enumerate(c.know_layers) if n < c.max_experts), None)
            if layer is not None:
                m.split_expert(layer)
                what = f"knowledge layer {layer} -> {c.know_layers[layer][1]} experts"
            elif c.n_layer < c.max_total_layer:
                m.grow_know()
                what = f"knowledge section -> {len(c.know_layers)} layers"
            else:
                return False
        self._make_opt()
        losses.clear()
        print(f"  [grow] {kind} learning stalled (loss {cur:.3f}): {what}, "
              f"total {c.n_layer} layers, {m.n_params() / 1e6:.2f}M params")
        return True

    # ---------------- answering ----------------
    def context_for(self, prompt):
        """Put dictionary meanings of still-unknown words in front of the prompt."""
        if not len(self.dictionary):
            return ""
        new = self.words.unknown(prompt, self.dictionary)
        return "".join(f"{w} means {self.dictionary.entries[w]}\n" for w in new)

    @torch.no_grad()
    def generate(self, prompt, max_new=200, temperature=0.8, use_dict=True):
        self.model.eval()
        text = (self.context_for(prompt) if use_dict else "") + prompt
        ids = torch.tensor([[NL] + list(text.encode("utf-8", errors="replace"))], device=self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.model.generate(ids, max_new=max_new, temperature=temperature)
        return bytes(out).decode("utf-8", errors="replace")

    def answer(self, prompt, learn=True, temperature=0.8):
        if learn:
            self.learn([prompt], kind="know", lr_scale=self.cfg.chat_lr_scale)
        return self.generate(prompt, temperature=temperature)

    def quiz(self, quiz_fn, n):
        correct = 0
        for _ in range(n):
            p, a = quiz_fn()
            got = self.generate(p, max_new=len(a) + 4, temperature=0, use_dict=False)
            correct += got.strip() == a
        return correct / n

    # ---------------- checkpoints ----------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"cfg": self.cfg.to_dict(), "model": self.model.state_dict(),
                    "strength": self.plast.state_dict(), "episodes": self.episodes.state_dict(),
                    "words": self.words.state_dict(), "step": self.step, "class_idx": self.class_idx}, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self = cls(Config(**ck["cfg"]), device)
        self.model.load_state_dict(ck["model"])
        self.plast.load_state_dict(ck["strength"], self.device)
        self.episodes.load_state_dict(ck["episodes"])
        self.words.load_state_dict(ck["words"])
        self.step, self.class_idx = ck["step"], ck["class_idx"]
        self.opt_born = self.step - self.cfg.warmup     # resumed model: no warmup needed
        return self


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
