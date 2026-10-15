"""v9 always-on learner: English is the master, knowledge is the slave.

kind="en"   (en-train, dictionary lessons, English practice)
    -> only the master (English) learns, standalone. Once the slave holds knowledge, the
       basics below the doorway learn slowly and the doorway check keeps old understanding steady.
kind="know" (feed, chat: any subject)
    -> only the slave (knowledge) learns, reading through the master. Then:
       - sentences made only of known words -> gentle English practice (master gets stronger)
       - unknown words -> dictionary lookup -> master learns the meaning
       - new / unusual text never trains English blindly
sleep -> revise English; re-read old knowledge notes with the current English (re-file);
         refresh the doorway reference; forget unimportant one-off notes.
"""
import copy
import os
import random

import torch

from config import Config
from curriculum import grammar_lesson
from dictionary import Dictionary, StrongWords, WordTracker
from memory import Episodes
from model import LanguageModel, section_of
from plasticity import Plasticity

NL = 10
MASTER = ("lang_low", "lang", "fast_lang")
SLAVE = ("know", "fast_know")


class Learner:
    def __init__(self, cfg: Config, device=None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LanguageModel(cfg).to(self.device)
        self.plast = Plasticity(cfg)
        self.episodes = Episodes(cfg)
        self.words = WordTracker(cfg)
        self.dictionary = Dictionary(cfg.dict_path)
        self.strong = StrongWords(cfg.strong_words_path)
        self.door_ref = None          # snapshot of the master below the doorway (old understanding)
        self.step = 0
        self.class_idx = 0
        self.losses = {"en": [], "know": []}
        self.last_gate = 0.0          # how much knowledge leaks into plain English
        self.last_drift = 0.0         # how far the doorway moved from the reference
        self.use_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        self._make_opt()

    # ---------------- setup ----------------
    def _make_opt(self):
        groups = {k: [] for k in MASTER + SLAVE}
        for n, p in self.model.named_parameters():
            groups[section_of(n, self.cfg.know_tap)].append(p)
        self.opt = torch.optim.AdamW([{"params": ps, "name": k} for k, ps in groups.items()],
                                     weight_decay=0.0, betas=(0.9, 0.95))
        self.opt_born = self.step     # warmup restarts after every growth

    def set_dictionary(self, path):
        self.cfg.dict_path = path
        self.dictionary = Dictionary(path)

    def set_strong_words(self, path):
        self.cfg.strong_words_path = path
        self.strong = StrongWords(path)

    @property
    def has_knowledge(self):
        return self.door_ref is not None

    def _set_mode(self, kind, lr_scale):
        c = self.cfg
        warm = min(1.0, (self.step - self.opt_born + 1) / c.warmup)
        if kind == "en":
            scales = {"lang_low": c.low_lr_scale if self.has_knowledge else 1.0, "lang": 1.0, "fast_lang": 1.0,
                      "know": 0.0, "fast_know": 0.0}
        else:
            scales = {"lang_low": 0.0, "lang": 0.0, "fast_lang": 0.0, "know": 1.0, "fast_know": 1.0}
        for g in self.opt.param_groups:
            base = c.fast_lr if g["name"].startswith("fast") else c.lr
            g["lr"] = base * scales[g["name"]] * lr_scale * warm
            for p in g["params"]:
                p.requires_grad_(scales[g["name"]] > 0)

    # ---------------- doorway reference ----------------
    def _refresh_doorway(self):
        """Take a frozen copy of the master below the doorway = 'how shelves are labelled now'."""
        emb = copy.deepcopy(self.model.emb)
        blocks = copy.deepcopy(self.model.lang[:self.cfg.know_tap])
        for p in list(emb.parameters()) + list(blocks.parameters()):
            p.requires_grad_(False)
        self.door_ref = (emb, blocks)
        self.plast.protect_basics = True

    def _doorway_drift(self, x, y):
        h = self.model.doorway(x).float()
        with torch.no_grad():
            h_ref = self.model.doorway(x, lower=self.door_ref).float()
        keep = y >= 0
        return ((h - h_ref) ** 2).mean(-1)[keep].mean()

    @torch.no_grad()
    def measure_drift(self, n=32):
        refs = self.episodes.sample(n, "en") or self.episodes.sample(n, "know")
        if not self.has_knowledge or not refs:
            return 0.0
        self.model.eval()
        x, y = self._rows(refs)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            return self._doorway_drift(x, y).item()

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

    # ---------------- one update ----------------
    def _update(self, batch, kind, lr_scale=1.0, train_fast=True, anchor=(), door_refs=()):
        """anchor   (know): English lessons; the knowledge gate is pushed to 0 on them.
        door_refs (en)  : old English text; the master's doorway must stay close to the reference."""
        self.model.train()
        self._set_mode(kind, lr_scale)
        if not train_fast:
            for g in self.opt.param_groups:
                if g["name"].startswith("fast"):
                    g["lr"] = 0.0
        longest = max(len(t) for t in list(batch) + list(anchor) + list(door_refs)) + 2
        batch = batch[:max(1, self.cfg.batch_tokens // longest)]
        x, y = self._rows(batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            _, loss = self.model(x, y, english_only=(kind == "en"))
            lesson_loss = loss.item()
            if anchor:
                ax, ay = self._rows(list(anchor))
                _, a_loss = self.model(ax, ay)
                gate = self.model.last_gate.squeeze(-1).float()[ay >= 0].abs().mean()
                loss = loss + a_loss + self.cfg.gate_penalty * gate
                self.last_gate = gate.item()
            if door_refs and self.has_knowledge:
                rx, ry = self._rows(list(door_refs))
                drift = self._doorway_drift(rx, ry)
                loss = loss + self.cfg.doorway_weight * drift
                self.last_drift = drift.item()
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

    def _revision(self, n, kind, min_count=1):
        k = max(1, int(n * self.cfg.replay_ratio / (1 - self.cfg.replay_ratio)))
        return self.episodes.sample(k, kind, min_count)

    def _english_step(self, lessons, lr_scale, replay=True):
        """Master learns: lessons + English revision, with the doorway check on old English."""
        revision = self._revision(len(lessons), "en") if replay else []
        refs = revision or lessons
        return self._update(lessons + revision, "en", lr_scale, door_refs=refs)

    # ---------------- learning ----------------
    def learn(self, texts, kind="know", lr_scale=1.0, remember=True, replay=True, importance=None):
        texts = [t for t in texts if t.strip()]
        if not texts:
            return None
        if kind == "en":
            loss = self._english_step(texts, lr_scale, replay)
        else:
            if not self.has_knowledge:
                self._refresh_doorway()          # the slave starts filing knowledge: label the shelves
            practice = [s for t in texts for s in self.words.known_sentences(t)]
            batch = list(texts) + (self._revision(len(texts), "know") if replay else [])
            anchor = []
            if replay:
                k = max(1, int(len(texts) * self.cfg.english_anchor_ratio / (1 - self.cfg.english_anchor_ratio)))
                anchor = self.episodes.sample(k // 2, "en", min_count=self.cfg.min_repeats)
                anchor += [grammar_lesson() for _ in range(k - len(anchor))]
            loss = self._update(batch, "know", lr_scale, anchor=anchor)
            if practice:                          # reading a subject also strengthens English
                self._english_step(practice[:self.cfg.batch_size], self.cfg.practice_lr_scale * lr_scale)
        self.losses[kind].append(loss)
        self._dictionary_step(texts)
        if remember:
            for t in texts:
                imp = importance if importance is not None else self.strong.importance(t, self.cfg)
                self.episodes.add(t, self.step, kind, imp)
        self.step += 1
        if self.step % self.cfg.sleep_every == 0:
            self.sleep()
        self.maybe_grow(kind)
        return loss

    def _dictionary_step(self, texts):
        """New words -> look them up -> the master learns their meaning (gently, with revision)."""
        new = []
        for t in texts:
            if len(self.dictionary):
                new += self.words.unknown(t, self.dictionary)
            self.words.observe(t)
        new = list(dict.fromkeys(new))
        if new:
            lessons = self.words.lookup_lessons(new, self.dictionary)
            revision = self.episodes.sample(max(4, 3 * len(lessons)), "en", min_count=self.cfg.min_repeats)
            self._update(lessons + revision, "en", self.cfg.dict_lr_scale, door_refs=revision or lessons)
            for les in lessons:
                self.episodes.add(les, self.step, "en")

    def sleep(self):
        """Revise English; re-read old knowledge notes with the current English; re-label the shelves."""
        consolidated = 0
        for _ in range(self.cfg.sleep_steps // 2):
            batch = self.episodes.sample(self.cfg.batch_size, "en", min_count=self.cfg.min_repeats)
            if not batch:
                break
            self._update(batch, "en", lr_scale=0.5, train_fast=False, door_refs=batch)
            consolidated += len(batch)
        drift = self.measure_drift()
        reread = self.cfg.sleep_steps // 2 + (self.cfg.refile_steps if drift > self.cfg.refile_drift else 0)
        for _ in range(reread):
            batch = self.episodes.sample(self.cfg.batch_size, "know", min_count=self.cfg.min_repeats)
            if not batch:
                break
            self._update(batch, "know", lr_scale=0.5, train_fast=False)
            consolidated += len(batch)
        if self.has_knowledge:
            self._refresh_doorway()               # knowledge is re-filed under today's English
        with torch.no_grad():
            for b in self.model.blocks():
                b.fast_B.mul_(self.cfg.sleep_fast_keep)
        return {"consolidated": consolidated, "doorway_drift": round(drift, 4), "re_read_steps": reread,
                "forgotten": self.episodes.prune(self.step), "episodes": len(self.episodes)}

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
            m.grow_lang()                         # added above the doorway: shelves unchanged
            what = f"English -> {len(c.lang_layers)} layers"
        else:
            layer = next((i for i, (_, n) in enumerate(c.know_layers) if n < c.max_experts), None)
            if layer is not None:
                m.split_expert(layer)
                what = f"knowledge layer {layer} -> {c.know_layers[layer][1]} experts"
            elif c.n_layer < c.max_total_layer:
                m.grow_know()
                what = f"knowledge -> {len(c.know_layers)} layers"
            else:
                return False
        self._make_opt()
        losses.clear()
        print(f"  [grow] {kind} learning stalled (loss {cur:.3f}): {what}, "
              f"total {c.n_layer} layers, {m.n_params() / 1e6:.2f}M params")
        return True

    # ---------------- answering ----------------
    def context_for(self, prompt):
        """Dictionary meanings of still-unknown words, placed in front of the prompt."""
        if not len(self.dictionary):
            return ""
        return "".join(f"{w} means {self.dictionary.entries[w]}\n" for w in self.words.unknown(prompt, self.dictionary))

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
        """Exact answer, or any of the forms listed as 'a | b' (same meaning, other words)."""
        correct = 0
        for _ in range(n):
            p, a = quiz_fn()
            forms = [f.strip() for f in a.split("|")]
            got = self.generate(p, max_new=max(len(f) for f in forms) + 4, temperature=0, use_dict=False)
            correct += got.strip() in forms
        return correct / n

    # ---------------- checkpoints ----------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        door = None
        if self.door_ref is not None:
            door = {"emb": self.door_ref[0].state_dict(), "blocks": self.door_ref[1].state_dict()}
        torch.save({"cfg": self.cfg.to_dict(), "model": self.model.state_dict(),
                    "strength": self.plast.state_dict(), "episodes": self.episodes.state_dict(),
                    "words": self.words.state_dict(), "door_ref": door,
                    "step": self.step, "class_idx": self.class_idx}, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self = cls(Config(**ck["cfg"]), device)
        self.model.load_state_dict(ck["model"])
        self.plast.load_state_dict(ck["strength"], self.device)
        self.episodes.load_state_dict(ck["episodes"])
        self.words.load_state_dict(ck["words"])
        if ck.get("door_ref"):
            self._refresh_doorway()                     # right shapes, then the saved values
            self.door_ref[0].load_state_dict(ck["door_ref"]["emb"])
            self.door_ref[1].load_state_dict(ck["door_ref"]["blocks"])
        self.step, self.class_idx = ck["step"], ck["class_idx"]
        self.opt_born = self.step - self.cfg.warmup
        return self


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
