"""A note keeper: the external memory that helps recollection.

People take notes because memory fades. Here a note is a sentence (or a line you typed) stored word for word, indexed by
the *content words* it contains with Hebbian word->note links:

* **Association**: every content word of a note gets a link to it; reading or recalling it again strengthens the links
  ("fire together, wire together"); the links of common words are discounted by their spread (inverse document frequency).
* **Recall by cue**: a question activates the notes that share its content words (plus dictionary synonyms at half
  weight); the score is the fraction of the cue's weight a note covers, scaled by how well the note is still remembered.
* **Learning from use**: when a note turns out to answer a question, the *question's* words are linked to it, so the next
  paraphrase of the question finds it more easily.
* **Forgetting, slowly**: notes fade far more slowly than facts (they are written down) but they do fade: every note has
  a half-life that grows with every recall, and a note that is never used is eventually dropped.

* **Selective**: a note keeper that wrote everything down would be a transcript. ``importance`` scores each sentence and
  ``select`` keeps only the points worth keeping: words never met before, words that keep coming back in the text (the
  topic), words you have *asked about* before (attention), names, numbers, definitions, corrections and anything you flag
  ("remember ...", "important ..."). Repeats of what is already noted are not noted again (they strengthen the old note);
  in a long text only a small share of the sentences is kept; and a long sentence is kept as a short *gist* plus its key
  terms, not word for word. More important notes fade more slowly.

Notes are stored in the order they were taken, so the oldest can be reviewed or forgotten explicitly.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
STOP = {"the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from", "and", "or", "but", "so", "is", "are",
        "was", "were", "be", "been", "am", "do", "does", "did", "has", "have", "had", "it", "he", "she", "they", "i", "you",
        "we", "him", "her", "them", "his", "their", "its", "my", "your", "our", "this", "that", "these", "those", "there",
        "what", "who", "whom", "where", "when", "why", "how", "which", "not", "no", "can", "could", "will", "would",
        "should", "may", "might", "must", "shall", "as", "if", "then", "than", "too", "very", "just", "also", "about",
        "into", "out", "up", "down", "over", "some", "any", "all", "s", "t"}
FRAME = {"color", "colour", "sound", "kind", "thing", "place", "name", "way", "number", "one", "time", "day", "mean",
         "tell", "say", "know", "remember", "note", "much", "many"}       # words that name the question, not its subject


CUES = {"remember", "important", "note", "never", "always", "must", "forget", "urgent", "key"}


@dataclass
class Note:
    id: int
    text: str
    words: List[str]
    born: float
    last: float
    stab: float
    source: str = "read"
    recalls: int = 0
    strength: float = 1.0
    keys: List[str] = field(default_factory=list)      # the important terms of the note
    importance: float = 1.0


class NoteKeeper:
    def __init__(self, lexicon=None, base_stab: float = 1200.0) -> None:
        self.lex = lexicon
        self.notes: Dict[int, Note] = {}
        self.links: Dict[str, Dict[int, float]] = {}      # content word -> {note id: association weight}
        self.base_stab = base_stab
        self.clock = 0.0
        self._next = 1
        self._lemma: Dict[str, str] = {}
        self.seen: Dict[str, int] = {}                     # how often each content word has been read (novelty)
        self.attention: Dict[str, float] = {}              # what has been asked about (what to attend to)
        self.threshold = 0.9                               # importance needed to be noted
        self.share = 0.4                                   # in a long text (more than 12 new sentences) only this share is kept

    # ------------------------------------------------------------------ words
    def lemma(self, w: str) -> str:
        r = self._lemma.get(w)
        if r is None:
            r = w
            if self.lex is not None:
                try:
                    a = self.lex.analyze(w)[0]
                    r = a.lemma if a.known else w
                except Exception:
                    r = w
            self._lemma[w] = r
        return r

    def content(self, text: str) -> List[str]:
        out: List[str] = []
        for w in WORD.findall(text.lower()):
            if w in STOP or len(w) < 2:
                continue
            l = self.lemma(w)
            if l not in STOP and l not in out:
                out.append(l)
        return out

    def idf(self, word: str) -> float:
        n = max(1, len(self.notes))
        df = len(self.links.get(word, ()))
        return math.log(1.0 + (n + 1.0) / (df + 1.0))

    # ------------------------------------------------------------------ writing
    def find(self, text: str) -> Optional[Note]:
        key = " ".join(sorted(self.content(text)))
        for n in self.notes.values():
            if " ".join(sorted(n.words)) == key:
                return n
        return None

    def add(self, text: str, source: str = "read", importance: float = 1.0, keys: Optional[List[str]] = None) -> Optional[Note]:
        text = " ".join(text.split())
        words = self.content(text)
        if not words:
            return None
        old = self.find(text)
        if old is not None:                                   # the same note again strengthens it
            self._strengthen(old)
            if source in ("told", "note"):
                old.source = source
            return old
        if keys:
            words = words + [k for k in keys if k not in words]      # key terms are searchable even if the gist dropped them
        keys = [w for w in (keys if keys is not None else words) if w in words]
        life = 0.5 + min(1.5, max(0.0, importance) / 2.0)     # important notes fade more slowly
        n = Note(self._next, text, words, self.clock, self.clock,
                 self.base_stab * life * (2.0 if source in ("told", "note") else 1.0), source, keys=keys, importance=importance)
        self._next += 1
        self.notes[n.id] = n
        for w in words:
            self.links.setdefault(w, {})[n.id] = 1.0 if w in keys else 0.4
        return n

    # ------------------------------------------------------------------ what is worth noting
    def importance(self, words: List[str], doc_count: Dict[str, int], n_props: int, told_cue: bool = False,
                   names: int = 0, numbers: bool = False, definition: bool = False, negation: bool = False) -> Tuple[float, List[str]]:
        """How much a sentence deserves a place in the notes, and which of its words are the key terms.
        Per word: novelty (1 for a word never read, falling as it is read again) times topicality (words that recur in this
        text), plus attention (words asked about before)."""
        wt: Dict[str, float] = {}
        for w in words:
            nov = 1.0 / (1.0 + self.seen.get(w, 0))
            c = doc_count.get(w, 1)
            top = 1.0 + 0.4 * min(3, c - 1) if c <= 6 else 0.6      # a recurring character is the topic; a phrase repeated everywhere is boilerplate
            wt[w] = nov * top + 1.2 * min(2.0, self.attention.get(w, 0.0))
        ranked = sorted(words, key=lambda w: -wt[w])
        best = [wt[w] for w in ranked[:3]]
        score = sum(best) / max(1, len(best)) if best else 0.0
        score += 0.5 * (n_props > 0) - 0.4 * (n_props == 0)
        score += 2.0 * told_cue + 0.2 * min(2, names) + 0.2 * numbers + 0.3 * definition + 0.2 * negation
        return score, ranked[:5]

    def select(self, scored: List[Tuple[float, int]], always: Set[int]) -> List[int]:
        """Indices to keep: everything flagged, else the best-scoring ones above the threshold, a limited share in a long text."""
        ok = [(s, i) for s, i in scored if s >= self.threshold and i not in always]
        ok.sort(key=lambda x: -x[0])
        n = len(scored)
        cap = len(ok) if n <= 12 else max(5, math.ceil(self.share * n))
        keep = sorted({i for _, i in ok[:cap]} | set(always))
        return keep

    def read_words(self, words: List[str]) -> None:
        for w in set(words):
            self.seen[w] = self.seen.get(w, 0) + 1

    def attend(self, words: List[str]) -> None:
        """Asking about something marks it as worth attention: later sentences with these words are more likely to be noted."""
        for w in words:
            self.attention[w] = min(3.0, self.attention.get(w, 0.0) + 0.5)

    def _strengthen(self, n: Note, amount: float = 0.3) -> None:
        n.last = self.clock
        n.recalls += 1
        n.strength += amount
        n.stab = min(self.base_stab * 25, n.stab * 1.5)
        for w in n.words:
            self.links[w][n.id] = self.links[w].get(n.id, 0.0) + 0.2

    def forget(self, note_id: int) -> bool:
        n = self.notes.pop(note_id, None)
        if n is None:
            return False
        for w in n.words:
            self.links.get(w, {}).pop(note_id, None)
        return True

    # ------------------------------------------------------------------ time
    def retention(self, n: Note) -> float:
        return 0.5 ** (max(0.0, self.clock - n.last) / n.stab)

    def advance(self, units: float, drop_below: float = 0.02) -> int:
        """Time passes; notes that were never used and have faded away are dropped. Returns how many."""
        self.clock += units
        for w in list(self.attention):                        # attention fades too
            self.attention[w] *= 0.5 ** (units / (self.base_stab / 2))
            if self.attention[w] < 0.05:
                del self.attention[w]
        dead = [i for i, n in self.notes.items() if self.retention(n) < drop_below]
        for i in dead:
            self.forget(i)
        return len(dead)

    # ------------------------------------------------------------------ reading
    def anchors(self, text: str) -> List[str]:
        """The words naming what a question is about (its subject), without the words naming the question itself."""
        return [w for w in self.content(text) if w not in FRAME]

    def search(self, query: str, k: int = 3, min_cover: float = 0.6, require_anchors: bool = True) -> List[Tuple[Note, float]]:
        cue = self.content(query)
        if not cue:
            return []
        anchors = set(self.anchors(query))
        weights: Dict[str, float] = {}
        for w in cue:
            weights[w] = max(weights.get(w, 0.0), 1.0)
            for syn in self._synonyms(w):
                weights[syn] = max(weights.get(syn, 0.0), 0.5)
        total = sum(self.idf(w) * v for w, v in weights.items() if v >= 1.0) or 1.0
        act: Dict[int, float] = {}
        for w, v in weights.items():
            for nid, a in self.links.get(w, {}).items():
                act[nid] = act.get(nid, 0.0) + self.idf(w) * v * min(2.0, a)
        out: List[Tuple[Note, float]] = []
        for nid, a in act.items():
            n = self.notes[nid]
            if require_anchors and anchors and not anchors <= set(n.words):
                if not all(self._has(n, w) for w in anchors):
                    continue
            cover = min(1.0, a / total)
            score = cover * (0.6 + 0.4 * self.retention(n)) * min(1.0, 0.7 + 0.3 * n.strength)
            if cover >= min_cover * (1.0 if not anchors else 0.5):
                out.append((n, score))
        out.sort(key=lambda x: (-x[1], -x[0].born))
        return out[:k]

    def _has(self, n: Note, w: str) -> bool:
        return w in n.words or any(s in n.words for s in self._synonyms(w))

    def _synonyms(self, w: str) -> List[str]:
        if self.lex is None:
            return []
        try:
            senses = self.lex.senses(w, "n") + self.lex.senses(w, "v")
        except Exception:
            return []
        out: List[str] = []
        for s in senses[:1]:
            out += [x for x in s.synonyms[:3] if x.isalpha() and x != w]
        return out

    def recalled(self, note: Note, cue: str) -> None:
        """A note answered a question: strengthen it, and link the question's own words to it (learning from use)."""
        self._strengthen(note)
        self.attend(note.keys)
        for w in self.content(cue):
            self.links.setdefault(w, {})
            self.links[w][note.id] = self.links[w].get(note.id, 0.0) + 0.3
            if w not in note.words and w not in FRAME:
                pass                                # kept as a link only; the note's own words are unchanged

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return dict(clock=self.clock, next=self._next,
                    seen=self.seen, attention=self.attention,
                    notes=[dict(id=n.id, text=n.text, words=n.words, born=n.born, last=n.last, stab=n.stab, source=n.source,
                                recalls=n.recalls, strength=n.strength, keys=n.keys, importance=n.importance)
                           for n in self.notes.values()],
                    links={w: {str(i): a for i, a in d.items()} for w, d in self.links.items()})

    def from_dict(self, d: dict) -> None:
        self.clock, self._next = d.get("clock", 0.0), d.get("next", 1)
        self.notes = {x["id"]: Note(x["id"], x["text"], x["words"], x["born"], x["last"], x["stab"], x["source"],
                                    x["recalls"], x["strength"], x.get("keys", x["words"]), x.get("importance", 1.0))
                      for x in d.get("notes", [])}
        self.seen, self.attention = d.get("seen", {}), d.get("attention", {})
        self.links = {w: {int(i): a for i, a in dd.items()} for w, dd in d.get("links", {}).items()}

    def __len__(self) -> int:
        return len(self.notes)
