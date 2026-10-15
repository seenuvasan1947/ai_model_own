"""Dictionary the model refers to only for words it does not know yet.

File format, one word per line:   word: definition
"""
import re
from collections import Counter

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def words_in(text):
    return [w.lower().strip("'-") for w in WORD.findall(text)]


class Dictionary:
    def __init__(self, path=""):
        self.path = path
        self.entries = {}
        if path:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if ":" in line and not line.lstrip().startswith("#"):
                        w, d = line.split(":", 1)
                        w, d = w.strip().lower(), " ".join(d.split())
                        if w and d and w not in self.entries:
                            self.entries[w] = d

    def __len__(self):
        return len(self.entries)


class WordTracker:
    """Counts how often each word was seen and how often it was looked up."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.seen = Counter()
        self.refs = Counter()

    def unknown(self, text, dictionary):
        """Words in text that are still new and have a dictionary entry (each returned once)."""
        out = []
        for w in dict.fromkeys(words_in(text)):
            if (self.seen[w] < self.cfg.word_known and self.refs[w] < self.cfg.dict_max_refs
                    and w in dictionary.entries):
                out.append(w)
        return out

    def observe(self, text):
        self.seen.update(words_in(text))

    def lookup_lessons(self, words, dictionary):
        lessons = []
        for w in words:
            self.refs[w] += 1
            lessons.append(f"{w} means {dictionary.entries[w]}")
        return lessons

    def state_dict(self):
        return {"seen": dict(self.seen), "refs": dict(self.refs)}

    def load_state_dict(self, d):
        self.seen, self.refs = Counter(d["seen"]), Counter(d["refs"])
