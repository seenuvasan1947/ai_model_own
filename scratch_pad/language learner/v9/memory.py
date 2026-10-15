"""Episode buffer: the model's notes. Remembers what was read (as text, so it can be re-read
with better English later), how often, which section it belongs to, and how important it is."""
import random


class Episodes:
    def __init__(self, cfg):
        self.cfg = cfg
        self.items = {}   # text -> [count, last_step, kind, importance]   kind: "en" or "know"

    def add(self, text, step, kind, importance=1.0):
        e = self.items.get(text)
        if e:
            e[0] += 1
            e[1] = step
            e[3] = max(e[3], importance)
        else:
            self.items[text] = [1, step, kind, importance]
        if len(self.items) > self.cfg.max_episodes:
            ranked = sorted(self.items.items(), key=lambda kv: (kv[1][0] * kv[1][3], kv[1][1]))
            for t, _ in ranked[: len(ranked) // 10]:
                del self.items[t]

    def sample(self, n, kind, min_count=1):
        pool = [(t, c * imp) for t, (c, _, k, imp) in self.items.items() if k == kind and c * imp >= min_count]
        if not pool or n <= 0:
            return []
        texts, weights = zip(*pool)
        return random.choices(texts, weights=[w ** 0.5 for w in weights], k=n)

    def prune(self, step):
        """Forget notes that were seen rarely, are not important, and have not come back for long."""
        old = [t for t, (c, last, _, imp) in self.items.items()
               if c * imp < self.cfg.min_repeats and step - last > self.cfg.forget_after * imp]
        for t in old:
            del self.items[t]
        return len(old)

    def count(self, kind):
        return sum(1 for v in self.items.values() if v[2] == kind)

    def __len__(self):
        return len(self.items)

    def state_dict(self):
        return self.items

    def load_state_dict(self, d):
        self.items = d
