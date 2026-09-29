"""Fast episodic index (the "hippocampus"): one-shot write, exact recall by meaning.

Each entry = the embedding of the question (or the sentence itself) plus its
text. Writing is instant: no training. Persists as keys.npy + index.json.
"""
import json
import os

import numpy as np


class EpisodicIndex:
    def __init__(self, dim):
        self.dim = dim
        self.keys = np.zeros((0, dim), dtype=np.float32)
        self.items = []   # {q, a, count, imp, sal, consolidated}

    def __len__(self):
        return len(self.items)

    def write(self, key, q, a, important=False):
        """One-shot write. Same (q, a) again = repetition, not a new entry."""
        for i, it in enumerate(self.items):
            if it["q"] == q and it["a"] == a:
                it["count"] += 1
                it["sal"] += 0.5
                it["imp"] = it["imp"] or important
                return i
        self.keys = np.vstack([self.keys, key[None, :]])
        self.items.append({"q": q, "a": a, "count": 1, "imp": important,
                           "sal": 1.0 + (2.0 if important else 0.0), "consolidated": False})
        return len(self.items) - 1

    def search(self, key, k=3):
        """[(index, cosine)] best first."""
        if not len(self):
            return []
        sims = self.keys @ key
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def remove(self, i):
        self.keys = np.delete(self.keys, i, axis=0)
        del self.items[i]

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "keys.npy"), self.keys)
        json.dump(self.items, open(os.path.join(folder, "index.json"), "w"), indent=1)

    def load(self, folder):
        kp, ip = os.path.join(folder, "keys.npy"), os.path.join(folder, "index.json")
        if os.path.exists(kp) and os.path.exists(ip):
            self.keys = np.load(kp)
            self.items = json.load(open(ip))
            return True
        return False
