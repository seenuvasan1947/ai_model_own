"""Layer 5 - persistent long-term store (key vectors + text gists + strength).

Retrieval score = cosine similarity * strength weight. Corrections (memories
of mistakes) get a boost and decay slower, so they are recalled first.
Everything persists to disk (keys.npy + meta.json).
"""
import json
import os

import numpy as np


class LongTermStore:
    def __init__(self, dim, capacity=5000):
        self.dim = dim
        self.capacity = capacity
        self.keys = np.zeros((0, dim), dtype=np.float32)
        self.meta = []  # dicts: text, kind, strength, hits, wrong

    def __len__(self):
        return len(self.meta)

    def _sims(self, q):
        return self.keys @ q if len(self.meta) else np.zeros(0, dtype=np.float32)

    def _weight(self, m):
        w = min(1.5, 0.6 + 0.2 * np.log1p(m["strength"]))
        return w * (1.5 if m["kind"] == "correction" else 1.0)

    def write(self, key, text, kind="fact", strength=1.0, dup_sim=0.92):
        """Write a memory. A near-identical memory of the same kind is
        reinforced (repetition) instead of duplicated. Returns its index."""
        if len(self):
            sims = self._sims(key)
            for i in np.argsort(-sims)[:3]:
                if sims[i] >= dup_sim and self.meta[i]["kind"] == kind:
                    self.meta[i]["strength"] += strength
                    if kind == "correction":
                        self.meta[i]["text"] = text  # newest correction wins
                    return int(i)
        self.keys = np.vstack([self.keys, key[None, :]])
        self.meta.append({"text": text, "kind": kind, "strength": float(strength),
                          "hits": 0, "wrong": 0})
        if len(self) > self.capacity:
            self._drop(int(np.argmin([m["strength"] for m in self.meta])))
        return len(self) - 1

    def retrieve(self, query, k=3):
        """Returns list of (index, score, similarity)."""
        if not len(self):
            return []
        sims = self._sims(query)
        scores = np.array([s * self._weight(m) for s, m in zip(sims, self.meta)])
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i]), float(sims[i])) for i in order]

    def reinforce(self, i, amount=1.0):
        self.meta[i]["strength"] += amount
        self.meta[i]["hits"] += 1

    def demote(self, i, factor=0.5):
        """Mistake: a stored gist turned out wrong -> weaken it."""
        self.meta[i]["strength"] *= factor
        self.meta[i]["wrong"] += 1

    def _drop(self, i):
        self.keys = np.delete(self.keys, i, axis=0)
        del self.meta[i]

    def decay(self, fact_rate=0.97, correction_rate=0.995, floor=0.2):
        """Sleep-like pass: everything fades unless reinforced; weak items go."""
        for m in self.meta:
            m["strength"] *= correction_rate if m["kind"] == "correction" else fact_rate
        for i in reversed(range(len(self.meta))):
            if self.meta[i]["strength"] < floor:
                self._drop(i)

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "keys.npy"), self.keys)
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=1)

    def load(self, folder):
        kp, mp = os.path.join(folder, "keys.npy"), os.path.join(folder, "meta.json")
        if os.path.exists(kp) and os.path.exists(mp):
            self.keys = np.load(kp)
            with open(mp) as f:
                self.meta = json.load(f)
            return True
        return False
