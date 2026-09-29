"""TitanAgent: neural memory + decoder + salience + mistake memory + imagination.

PLAN_FUTURE items implemented here:
  1 generation      - decoder writes a sentence from question + recalled memory
  2 imagination     - candidate answers are scored BEFORE answering (support from
                      memory, self-consistency check, mistake penalty); best wins
  3 feedback        - /wrong, /good, contradiction detection, self-check
  4 salience        - importance tags, repetition, decay, replay by salience, sleep()
State (all saved after every change): memory.pt, texts.json
"""
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from brainlm.encoders import make_encoder
from neural_memory import NeuralMemory

DIM = 128
MIN_SIM = 0.55
MAX_UPDATES, REPLAY = 60, 16
TARGET = {False: 0.05, True: 0.02}   # surprise to reach; important facts learned harder
CONFLICT_COS = 0.85                  # questions this similar are "the same question"
NO_ANSWER = "I don't know."
# words of question/answer templates; every OTHER word of the question must be copied
STOP = set("what is the of who which city tell me about a an define does do and created creator "
           "capital boiling point how why where when are was".split())


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
SENTENCE_WORDS = 6   # a stored answer this long is a whole sentence: return it as-is


def split_sentences(text):
    """Paragraph or file text -> sentences of at least 3 words (order kept, no repeats)."""
    seen, out = set(), []
    for s in (x.strip() for x in _SENT.split(text)):
        if len(s.split()) >= 3 and s not in seen:
            seen.add(s); out.append(s)
    return out


def grounded_in(q, memory_texts, text):
    """Self-check against invented content: the stored answer AND the entity
    words from the question must appear in the generated sentence."""
    low = text.lower()
    if not all(t.lower().rstrip(".") in low for t in memory_texts):
        return False
    words = [w.strip("?.,!").lower() for w in q.split()]
    return all(w in low for w in words if w and w not in STOP)


class TitanAgent:
    def __init__(self, state="titan_state", decoder=None, seed=0, encoder="hash"):
        self.state = state
        self.enc = make_encoder(encoder)
        self.encoder_name = encoder
        self.mem = NeuralMemory(self.enc.dim, dim=DIM, depth=1, seed=seed)
        self.texts = []       # answer wordings (weights hold no words)
        self.pairs = []       # rehearsal buffer: {q, a, count, imp, sal}
        self.mistakes = []    # {q, wrong}: answers the user rejected
        self.updates = 0
        self.last = None      # {q, text, memory}
        self.decoder = decoder
        self._cache = {}
        self._load()

    # ---- helpers -------------------------------------------------------
    def E(self, t):
        if t not in self._cache:
            self._cache[t] = torch.tensor(self.enc.encode(t))[None]
        return self._cache[t]

    def _wflat(self):
        return torch.cat([w.flatten() for w in self.mem.M])

    # ---- persistence ---------------------------------------------------
    def _load(self):
        mp, tp = (os.path.join(self.state, f) for f in ("memory.pt", "texts.json"))
        self.loaded = os.path.exists(mp) and os.path.exists(tp)
        if self.loaded:
            self.mem.load(mp)
            d = json.load(open(tp))
            if d.get("encoder", "hash") != self.encoder_name:
                raise SystemExit(f"this memory was saved with the '{d.get('encoder', 'hash')}' encoder; "
                                 f"run with --encoder {d.get('encoder', 'hash')} or use a new --state folder")
            self.texts, self.updates = d["texts"], d["updates"]
            self.pairs, self.mistakes = d.get("pairs", []), d.get("mistakes", [])

    def save(self):
        os.makedirs(self.state, exist_ok=True)
        self.mem.save(os.path.join(self.state, "memory.pt"))
        json.dump({"encoder": self.encoder_name, "texts": self.texts, "updates": self.updates, "pairs": self.pairs,
                   "mistakes": self.mistakes},
                  open(os.path.join(self.state, "texts.json"), "w"), indent=1)

    # ---- learn ---------------------------------------------------------
    def _replay_batch(self, exclude=None, n=REPLAY):
        old = [p for p in self.pairs if p is not exclude]
        if not old:
            return []
        w = torch.tensor([p["sal"] for p in old])
        idx = torch.multinomial(w, min(n, len(old)), replacement=False).tolist()
        return [old[i] for i in idx]

    def teach(self, q, a, important=False, min_updates=5):
        """Learn q -> a. Returns a dict describing what happened."""
        info = {"conflict": None}
        # contradiction check (feedback signal: the new fact disagrees with an old one)
        for p in list(self.pairs):
            if q == a or p["q"] == p["a"]:   # taught sentences (sentence -> itself) are never "contradictions"
                continue
            same_q = p["q"] == q or (self.E(p["q"]) @ self.E(q).T).item() > CONFLICT_COS
            if same_q and p["a"] != a:
                info["conflict"] = p["a"]
                self.pairs.remove(p)
        pair = next((p for p in self.pairs if p["q"] == q and p["a"] == a), None)
        if pair:                                  # repetition strengthens it
            pair["count"] += 1
            pair["sal"] += 0.5
            pair["imp"] = pair["imp"] or important
        else:
            pair = {"q": q, "a": a, "count": 1, "imp": important, "sal": 1.0}
            self.pairs.append(pair)
        if important:
            pair["sal"] += 2.0
        if a not in self.texts:
            self.texts.append(a)
        x, y = self.E(q), self.E(a)
        info["surprise_before"] = self.mem.surprise_pair(x, y).item()
        w0 = self._wflat()
        n = 0
        while n < MAX_UPDATES and (n < min_updates or
                                   self.mem.surprise_pair(x, y).item() > TARGET[pair["imp"]]):
            rep = self._replay_batch(pair)
            bx = torch.cat([x] + [self.E(p["q"]) for p in rep])
            by = torch.cat([y] + [self.E(p["a"]) for p in rep])
            self.mem.update(bx, by)
            self.updates += 1
            n += 1
        info.update(surprise_after=self.mem.surprise_pair(x, y).item(), updates=n,
                    moved=(self._wflat() - w0).norm().item(), sal=pair["sal"])
        self.save()
        return info

    # ---- recall --------------------------------------------------------
    def recall(self, q):
        if not self.texts:
            return [], None
        vec = F.normalize(self.mem.retrieve(self.E(q)), dim=-1)
        cand = F.normalize(torch.cat([self.mem.target(self.E(t)) for t in self.texts]), dim=-1)
        sims = (vec @ cand.T)[0]
        top = sims.topk(min(4, len(self.texts)))
        return [(self.texts[i], s.item()) for s, i in zip(top.values, top.indices)], vec

    def _penalty(self, q, memory_texts):
        """Imagined outcome: has this answer been rejected for a similar question?"""
        for m in self.mistakes:
            if m["wrong"] in memory_texts and (self.E(m["q"]) @ self.E(q).T).item() > 0.8:
                return 1.0
        return 0.0

    def learn_text(self, text, important=False, progress=None):
        """Read any-length text: split into sentences and learn each one
        (sentence -> itself). Returns list of per-sentence info dicts."""
        results = []
        sents = split_sentences(text)
        for i, s in enumerate(sents, 1):
            r = self.teach(s, s, important=important)
            r["sentence"] = s
            results.append(r)
            if progress:
                progress(i, len(sents), r)
        return results

    def _write(self, q, mem_texts, temp=0.0):
        m = " ; ".join(mem_texts)
        if self.decoder is None or (len(mem_texts) == 1 and len(m.split()) >= SENTENCE_WORDS):
            return m
        return self.decoder.generate(q, m, temp=temp)

    def answer(self, q):
        """recall -> imagine candidates -> score -> pick -> answer.
        Returns dict(text, candidates=[...], ranked)."""
        ranked, _ = self.recall(q)
        if not ranked or ranked[0][1] < MIN_SIM:
            self.last = {"q": q, "text": NO_ANSWER, "memory": []}
            return {"text": NO_ANSWER, "candidates": [], "ranked": ranked}
        # candidate memory sets (what to base an answer on)
        sets = [[ranked[i]] for i in range(min(3, len(ranked))) if ranked[i][1] >= MIN_SIM * 0.8]
        if " and " in q and len(ranked) > 1 and ranked[1][1] >= MIN_SIM * 0.8:
            sets.append([ranked[0], ranked[1]])
        cands = []
        for s in sets:
            texts = [t for t, _ in s]
            tries = [0.0] + ([0.7, 0.7] if s is sets[0] else [])
            for temp in tries:
                text = self._write(q, texts, temp)
                support = sum(sim for _, sim in s) / len(s)
                used_decoder = self.decoder and not (
                    len(texts) == 1 and len(" ; ".join(texts).split()) >= SENTENCE_WORDS)
                grounded = grounded_in(q, texts, text) if used_decoder else True
                pen = self._penalty(q, texts)
                score = 0.6 * support + 0.4 * (1.0 if grounded else 0.0) - pen
                cands.append({"text": text, "memory": texts, "support": support,
                              "grounded": grounded, "penalty": pen, "score": score, "temp": temp})
        best = max(cands, key=lambda c: c["score"])
        text, fallback = best["text"], False
        if self.decoder and not best["grounded"]:   # decoder garbled the memory: say the stored text as is
            text, fallback = " ; ".join(best["memory"]), True
        self.last = {"q": q, "text": text, "memory": best["memory"]}
        return {"text": text, "candidates": cands, "ranked": ranked, "fallback": fallback}

    # ---- learning from feedback ---------------------------------------
    def wrong(self, right):
        """Last answer was wrong: remember the mistake, relearn the right answer."""
        if not self.last:
            raise RuntimeError("ask something first")
        q = self.last["q"]
        for t in self.last["memory"]:
            if t != right:
                self.mistakes.append({"q": q, "wrong": t})
        info = self.teach(q, right, important=True)
        return info

    def good(self):
        if not self.last:
            return
        for p in self.pairs:
            if p["a"] in self.last["memory"]:
                p["sal"] += 0.5
        self.save()

    # ---- sleep: consolidate by salience, forget the unimportant -------
    def sleep(self, updates=30):
        if not self.pairs:
            return {"replayed": 0, "dropped": 0}
        for _ in range(updates):
            rep = self._replay_batch(None, REPLAY * 2)
            self.mem.update(torch.cat([self.E(p["q"]) for p in rep]),
                            torch.cat([self.E(p["a"]) for p in rep]))
            self.updates += 1
        dropped = []
        for p in list(self.pairs):
            p["sal"] *= 0.9                       # fades unless it was reinforced
            if p["sal"] < 0.3 and not p["imp"]:
                self.pairs.remove(p); dropped.append(p["q"])
        self.save()
        return {"replayed": updates, "dropped": dropped}

    def stats(self):
        return {"answers": len(self.texts), "rehearsal_pairs": len(self.pairs),
                "mistakes": len(self.mistakes), "updates": self.updates,
                "params": sum(w.numel() for w in self.mem.M),
                "decoder": self.decoder.backbone if self.decoder else None}
