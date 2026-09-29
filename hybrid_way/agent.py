"""HybridAgent: fast episodic index + slow neural memory (our own design).

  teach   -> written to the episodic index at once (one shot, no training)
             and to a small working buffer
  sleep   -> CONSOLIDATION: replay unconsolidated index entries into the neural
             memory (slow, gradient-based, with rehearsal); optionally prune them
  ask     -> 1) episodic index (nearest by meaning)  2) neural memory  3) refuse
  wrong   -> mistake memory + relearn (index at once, neural at next sleep)

The neural memory is the same mini-Titans module used in titan_way.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import importlib.util

for p in (ROOT, os.path.join(ROOT, "titan_way"), HERE):
    sys.path.insert(0, p)
from brainlm.encoders import make_encoder
from neural_memory import NeuralMemory
from episodic import EpisodicIndex

# titan_way/agent.py has the same file name as this one, so load it under another name
_spec = importlib.util.spec_from_file_location("titan_agent", os.path.join(ROOT, "titan_way", "agent.py"))
_ta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ta)
split_sentences, grounded_in = _ta.split_sentences, _ta.grounded_in
SENTENCE_WORDS, NO_ANSWER = _ta.SENTENCE_WORDS, _ta.NO_ANSWER

DIM, MAX_UPDATES, REPLAY, TARGET = 128, 60, 16, 0.05
CONFLICT_COS = 0.9
WORKING_SIZE = 8


class HybridAgent:
    def __init__(self, state="hybrid_state", encoder="minilm", decoder=None, seed=0,
                 thr_idx=0.35, thr_neu=0.55, auto_consolidate=0, prune=False):
        self.state, self.encoder_name, self.decoder = state, encoder, decoder
        self.enc = make_encoder(encoder)
        self.idx = EpisodicIndex(self.enc.dim)
        self.mem = NeuralMemory(self.enc.dim, dim=DIM, depth=1, seed=seed)
        self.texts = []                 # answer wordings the neural memory can name
        self.working = []               # layer 1: last few (q, a) seen
        self.mistakes = []
        self.updates = 0
        self.thr_idx, self.thr_neu = thr_idx, thr_neu
        self.auto_consolidate, self.prune = auto_consolidate, prune
        self.use_index = self.use_neural = True
        self.last = None
        self._cache = {}
        self.loaded = self._load()

    def E(self, t):
        if t not in self._cache:
            self._cache[t] = torch.tensor(self.enc.encode(t))[None]
        return self._cache[t]

    def _k(self, t):
        return self.E(t)[0].numpy()

    # ---- persistence ---------------------------------------------------
    def _load(self):
        mp, tp = os.path.join(self.state, "memory.pt"), os.path.join(self.state, "meta.json")
        if not (os.path.exists(mp) and os.path.exists(tp) and self.idx.load(self.state)):
            return False
        d = json.load(open(tp))
        if d["encoder"] != self.encoder_name:
            raise SystemExit(f"saved with encoder '{d['encoder']}'; use --encoder {d['encoder']} or a new --state")
        self.mem.load(mp)
        self.texts, self.mistakes, self.updates = d["texts"], d["mistakes"], d["updates"]
        return True

    def save(self):
        os.makedirs(self.state, exist_ok=True)
        self.idx.save(self.state)
        self.mem.save(os.path.join(self.state, "memory.pt"))
        json.dump({"encoder": self.encoder_name, "texts": self.texts, "mistakes": self.mistakes,
                   "updates": self.updates}, open(os.path.join(self.state, "meta.json"), "w"), indent=1)

    # ---- teach: one shot into the episodic index ------------------------
    def teach(self, q, a, important=False, save=True):
        info = {"conflict": None}
        if q != a:                                   # real Q->A pair: check contradictions
            for i, it in list(enumerate(self.idx.items))[::-1]:
                if it["q"] != it["a"] and it["a"] != a and (
                        it["q"] == q or float(self.idx.keys[i] @ self._k(q)) > CONFLICT_COS):
                    info["conflict"] = it["a"]
                    self.idx.remove(i)
        i = self.idx.write(self._k(q), q, a, important)
        if a not in self.texts:
            self.texts.append(a)
        self.working = (self.working + [(q, a)])[-WORKING_SIZE:]
        info["index"] = i
        info["pending"] = sum(1 for it in self.idx.items if not it["consolidated"])
        if self.auto_consolidate and info["pending"] >= self.auto_consolidate:
            info["consolidated"] = self.consolidate()
        if save:
            self.save()
        return info

    def learn_text(self, text, important=False, progress=None):
        out = []
        sents = split_sentences(text)
        for n, s in enumerate(sents, 1):
            r = self.teach(s, s, important, save=False)   # save once at the end, not per sentence
            r["sentence"] = s
            out.append(r)
            if progress:
                progress(n, len(sents), r)
        self.save()
        return out

    # ---- consolidate: index -> neural memory (slow) ---------------------
    def _replay(self, exclude, n=REPLAY):
        old = [it for it in self.idx.items if it["consolidated"] and it is not exclude]
        if not old:
            return []
        w = torch.tensor([it["sal"] for it in old])
        return [old[j] for j in torch.multinomial(w, min(n, len(old)), replacement=False).tolist()]

    def consolidate(self):
        done = n_up = 0
        for it in [x for x in self.idx.items if not x["consolidated"]]:
            x, y = self.E(it["q"]), self.E(it["a"])
            n = 0
            while n < MAX_UPDATES and (n < 3 or self.mem.surprise_pair(x, y).item() > TARGET):
                rep = self._replay(it)
                self.mem.update(torch.cat([x] + [self.E(r["q"]) for r in rep]),
                                torch.cat([y] + [self.E(r["a"]) for r in rep]))
                n += 1
            it["consolidated"] = True
            done += 1; n_up += n; self.updates += n
        return {"consolidated": done, "updates": n_up}

    def sleep(self, replay_batches=10):
        """Consolidate new entries, replay old by salience, fade, optionally prune."""
        r = self.consolidate()
        for _ in range(replay_batches):
            rep = self._replay(None, REPLAY * 2)
            if rep:
                self.mem.update(torch.cat([self.E(x["q"]) for x in rep]),
                                torch.cat([self.E(x["a"]) for x in rep]))
                self.updates += 1
        pruned = []
        for it in list(self.idx.items):
            it["sal"] *= 0.95
        if self.prune:
            for i in range(len(self.idx.items) - 1, -1, -1):
                it = self.idx.items[i]
                if it["consolidated"] and it["sal"] < 1.0 and not it["imp"] \
                        and float(self.mem.surprise_pair(self.E(it["q"]), self.E(it["a"]))) < TARGET * 2:
                    pruned.append(it["q"]); self.idx.remove(i)
        self.save()
        return {**r, "replayed": replay_batches, "pruned": pruned}

    # ---- recall ----------------------------------------------------------
    def ask_index(self, q):
        top = self.idx.search(self._k(q), 1)
        if not top:
            return None, 0.0
        i, c = top[0]
        return self.idx.items[i]["a"], c

    def ask_neural(self, q):
        if not self.texts:
            return None, 0.0
        vec = F.normalize(self.mem.retrieve(self.E(q)), dim=-1)
        cand = F.normalize(torch.cat([self.mem.target(self.E(t)) for t in self.texts]), dim=-1)
        s, j = (vec @ cand.T)[0].max(0)
        return self.texts[j.item()], s.item()

    def answer(self, q):
        """index first, then neural memory, else refuse. Returns dict with the path used."""
        tried = []
        text = path = None
        score = 0.0
        if self.use_index:
            t, s = self.ask_index(q); tried.append(("index", s))
            if t is not None and s >= self.thr_idx:
                text, path, score = t, "index", s
        if text is None and self.use_neural:
            t, s = self.ask_neural(q); tried.append(("neural", s))
            if t is not None and s >= self.thr_neu:
                text, path, score = t, "neural", s
        if text is None:
            self.last = {"q": q, "text": NO_ANSWER, "memory": []}
            return {"text": NO_ANSWER, "path": "refused", "score": max([s for _, s in tried] or [0.0]), "tried": tried}
        out, fallback = text, False
        if self.decoder and len(text.split()) < SENTENCE_WORDS:
            g = self.decoder.generate(q, text)
            if grounded_in(q, [text], g):
                out = g
            else:
                fallback = True
        self.last = {"q": q, "text": out, "memory": [text]}
        return {"text": out, "path": path, "score": score, "tried": tried, "fallback": fallback}

    # ---- feedback --------------------------------------------------------
    def wrong(self, right):
        if not self.last:
            raise RuntimeError("ask something first")
        q = self.last["q"]
        for t in self.last["memory"]:
            if t != right:
                self.mistakes.append({"q": q, "wrong": t})
        info = self.teach(q, right, important=True)
        for it in self.idx.items:
            if it["q"] == q:
                it["consolidated"] = False      # neural memory relearns it at next sleep
        self.save()
        return info

    def good(self):
        if self.last:
            for it in self.idx.items:
                if it["a"] in self.last["memory"]:
                    it["sal"] += 0.5
            self.save()

    def stats(self):
        return {"episodic_entries": len(self.idx),
                "consolidated": sum(1 for i in self.idx.items if i["consolidated"]),
                "pending": sum(1 for i in self.idx.items if not i["consolidated"]),
                "neural_params": sum(w.numel() for w in self.mem.M), "updates": self.updates,
                "mistakes": len(self.mistakes), "working_buffer": len(self.working),
                "encoder": self.encoder_name}
