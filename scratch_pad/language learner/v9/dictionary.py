"""Words the model knows, the dictionary it refers to for new words, and strong words.

Dictionary file, one word per line:      word: definition
Strong-words file, one word per line:    word        (strong / destructive / emotional words)
"""
import re
from collections import Counter

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def words_in(text):
    return [w.lower().strip("'-") for w in WORD.findall(text)]


def sentences(text):
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


class Dictionary:
    """Looked up only for words the model does not know yet; never read as a whole."""

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


class StrongWords:
    """A simple stand-in for emotion: texts with strong words are kept longer and revised more."""

    def __init__(self, path=""):
        self.words = set()
        if path:
            with open(path, encoding="utf-8", errors="replace") as f:
                self.words = {l.strip().lower() for l in f if l.strip() and not l.startswith("#")}

    def importance(self, text, cfg):
        hits = sum(1 for w in words_in(text) if w in self.words)
        return min(cfg.max_importance, 1.0 + cfg.strong_weight * hits)


class WordTracker:
    """Counts how often each word was seen and how often it was looked up."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.seen = Counter()
        self.refs = Counter()

    def is_known(self, w):
        return self.seen[w] >= self.cfg.word_known

    def unknown(self, text, dictionary):
        """Still-new words in text that the dictionary can explain (each returned once)."""
        return [w for w in dict.fromkeys(words_in(text))
                if not self.is_known(w) and self.refs[w] < self.cfg.dict_max_refs and w in dictionary.entries]

    def known_sentences(self, text):
        """Sentences made only of known words: safe English practice while reading any subject."""
        return [s for s in sentences(text) if words_in(s) and all(self.is_known(w) for w in words_in(s))]

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
