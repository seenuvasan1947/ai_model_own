"""Episode buffer (the 'hippocampus'): remembers what was read, how often, and which section it belongs to."""
import random


class Episodes:
    def __init__(self, cfg):
        self.cfg = cfg
        self.items = {}   # text -> [count, last_step, kind]   kind: "en" or "know"

    def add(self, text, step, kind):
        e = self.items.get(text)
        if e:
            e[0] += 1
            e[1] = step
        else:
            self.items[text] = [1, step, kind]
        if len(self.items) > self.cfg.max_episodes:
            ranked = sorted(self.items.items(), key=lambda kv: (kv[1][0], kv[1][1]))
            for t, _ in ranked[: len(ranked) // 10]:
                del self.items[t]

    def sample(self, n, kind, min_count=1):
        pool = [(t, c) for t, (c, _, k) in self.items.items() if k == kind and c >= min_count]
        if not pool or n <= 0:
            return []
        texts, counts = zip(*pool)
        return random.choices(texts, weights=[c ** 0.5 for c in counts], k=n)

    def prune(self, step):
        old = [t for t, (c, last, _) in self.items.items()
               if c < self.cfg.min_repeats and step - last > self.cfg.forget_after]
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
