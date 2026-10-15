"""Episode buffer (the 'hippocampus'): remembers what was read and how often."""
import random


class Episodes:
    def __init__(self, cfg):
        self.cfg = cfg
        self.items = {}   # text -> [count, last_step]

    def add(self, text, step):
        e = self.items.get(text)
        if e:
            e[0] += 1
            e[1] = step
        else:
            self.items[text] = [1, step]
        if len(self.items) > self.cfg.max_episodes:
            self._evict()

    def _evict(self):
        # drop the least repeated, oldest tenth
        ranked = sorted(self.items.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        for t, _ in ranked[: len(ranked) // 10]:
            del self.items[t]

    def sample(self, n, min_count=1):
        pool = [(t, c) for t, (c, _) in self.items.items() if c >= min_count]
        if not pool or n <= 0:
            return []
        texts, weights = zip(*pool)
        return random.choices(texts, weights=[c ** 0.5 for c in weights], k=n)

    def prune(self, step):
        """Forget episodes seen once that have not come back for a long time."""
        old = [t for t, (c, last) in self.items.items()
               if c < self.cfg.min_repeats and step - last > self.cfg.forget_after]
        for t in old:
            del self.items[t]
        return len(old)

    def __len__(self):
        return len(self.items)

    def state_dict(self):
        return self.items

    def load_state_dict(self, d):
        self.items = d
