"""BrainAgent: working buffer -> surprise gate -> consolidation -> long-term
store, with mandatory recall before answering and error-driven memory updates.

Layers (see PLAN_CURRENT.md):
  0 encoder | 1 working buffer | 2 surprise gate | 4 consolidation |
  5 long-term store | recall + error-driven update (steps 2 and 3)
Layers 3 (fast memory) and 6 (router) live in od/full_model.py's toy test.
"""
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .encoders import GRUEncoder, HashEncoder
from .memory import LongTermStore

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Answer:
    text: str | None
    score: float = 0.0
    sources: list = field(default_factory=list)  # [(index, score, text, kind)]


class BrainAgent:
    def __init__(self, state_dir="brain_state", encoder="hash", surprise_gate=0.15,
                 min_score=0.2, capacity=5000):
        self.state_dir = state_dir
        if encoder == "gru":
            self.encoder = GRUEncoder()
            p = os.path.join(state_dir, "encoder.pt")
            if os.path.exists(p):
                self.encoder.load(p)
        else:
            self.encoder = HashEncoder()
        self.store = LongTermStore(self.encoder.dim, capacity)
        self.buffer = []  # working memory: (key, text, surprise)
        self.surprise_gate = surprise_gate
        self.min_score = min_score
        self.last = None  # (question, Answer) for /ok and /wrong
        self.steps = 0
        self._load()

    # ---- persistence -------------------------------------------------
    def _load(self):
        if self.store.load(self.state_dir):
            with open(os.path.join(self.state_dir, "agent.json")) as f:
                self.steps = json.load(f).get("steps", 0)

    def save(self):
        os.makedirs(self.state_dir, exist_ok=True)
        self.store.save(self.state_dir)
        self.encoder.save(os.path.join(self.state_dir, "encoder.pt"))
        with open(os.path.join(self.state_dir, "agent.json"), "w") as f:
            json.dump({"steps": self.steps, "encoder": self.encoder.name}, f)

    # ---- learning (layers 1, 2, 4) ----------------------------------
    def surprise(self, key):
        """1 - similarity to the closest memory. Empty store = fully novel."""
        if not len(self.store):
            return 1.0
        return float(1.0 - self.store._sims(key).max())

    def learn(self, text):
        """Read any-length text. Returns (stored, reinforced) counts."""
        for sent in (s.strip() for s in _SENT.split(text)):
            if len(sent.split()) < 3:
                continue
            key = self.encoder.encode(sent)
            self.buffer.append((key, sent, self.surprise(key)))
        return self.consolidate()

    def consolidate(self):
        """Promote buffered items: novel ones are stored, familiar ones just
        reinforce the memory they resemble. Then everything decays a little."""
        stored = reinforced = 0
        for key, sent, surprise in self.buffer:
            if surprise >= self.surprise_gate:
                self.store.write(key, sent, "fact")
                stored += 1
            else:
                top = self.store.retrieve(key, 1)
                if top:
                    self.store.reinforce(top[0][0])
                    reinforced += 1
            self.steps += 1
        self.buffer = []
        self.store.decay()
        return stored, reinforced

    # ---- recall (mandatory before every answer) ---------------------
    def ask(self, question, k=3):
        q = self.encoder.encode(question)
        hits = self.store.retrieve(q, k)
        sources = [(i, s, self.store.meta[i]["text"], self.store.meta[i]["kind"])
                   for i, s, _ in hits]
        if not hits or hits[0][1] < self.min_score:
            ans = Answer(None, hits[0][1] if hits else 0.0, sources)
        else:
            ans = Answer(sources[0][2], hits[0][1], sources)
        self.last = (question, ans)
        return ans

    # ---- learning from mistakes (error-driven update) ---------------
    def confirm(self, question=None):
        """The answer was right: strengthen the memory that produced it."""
        q, ans = self._last(question)
        if ans.sources:
            self.store.reinforce(ans.sources[0][0])

    def correct(self, right_answer, question=None):
        """The answer was wrong (explicit signal). error = 1 - sim(given, right).
        Weaken the wrong memory and store the right answer keyed by the
        question as a 'correction'; bigger error = stronger memory.
        Returns the error."""
        q, ans = self._last(question)
        right_key = self.encoder.encode(right_answer)
        given = ans.text
        error = 1.0 if given is None else float(1.0 - self.encoder.encode(given) @ right_key)
        if error < 0.02:  # user "corrected" to what we already said
            self.confirm(q)
            return error
        if given is not None and ans.sources:
            self.store.demote(ans.sources[0][0], 0.3)
        # bigger error -> stronger memory of the mistake
        self.store.write(self.encoder.encode(q), right_answer, "correction",
                         strength=3.0 + 5.0 * error)
        self.store.write(right_key, right_answer, "fact", strength=1.0)
        return error

    def _last(self, question):
        if question is not None:
            return question, self.ask(question)
        if self.last is None:
            raise RuntimeError("ask a question first")
        return self.last

    def stats(self):
        kinds = {}
        for m in self.store.meta:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        return {"memories": len(self.store), "by_kind": kinds, "steps": self.steps,
                "encoder": self.encoder.name, "state_dir": self.state_dir}
