"""Semantic memory: where what was *read* is stored, so that it can be retrieved exactly.

The language layer (``model.py``) is excellent at grammar and fluency but stores facts only diffusely, in word
statistics; that is why it needs several repetitions and can blur two facts together. This memory stores each
proposition (subject, relation, value) as an explicit edge of a graph:

* **Hebbian strengthening**: reading the same proposition again strengthens the edge (``strength += 1``).
* **Lateral inhibition**: values that cannot both be true of a subject (two colours, two sounds, two homes, two
  leg counts, antonyms) compete; new evidence weakens the rival, told (authoritative) evidence weakens it more.
  This is how a correction replaces an old belief.
* **Recency**: among equally strong values the newer one wins.
* **Spreading activation through "is a"**: if nothing is stored about a concept, activation climbs to its
  ancestors (stored "is a" edges plus the dictionary's hypernyms), so a new word inherits what is known about
  its kind, with a confidence discount per step.
* **Typed retrieval**: "what colour" only accepts colour words, "where" places, "what does X say" sounds,
  decided by the dictionary's word kinds.

Everything can be inspected (`explain`) and saved as JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .comprehension import Prop, Query, evaluate_arithmetic, number_word
from .entities import EntityStore, is_instance, lemma_of
from .lexicon import Lexicon

SPATIAL = ("in", "on", "at", "near", "under", "inside", "behind", "beside", "above", "below")
PLACE_KINDS = {"place", "building"}


@dataclass
class Fact:
    obj: str
    qty: str = ""
    neg: bool = False
    strength: float = 0.0
    count: int = 0
    t: int = 0
    told: bool = False
    source: str = "read"
    obj_num: str = "sg"
    rel: str = ""

    def score(self, now: int, half_life: float = 400.0) -> float:
        age = max(0, now - self.t)
        return self.strength * (0.8 + 0.2 * 0.5 ** (age / half_life)) * (1.15 if self.told else 1.0)


@dataclass
class Answer:
    items: List[Fact] = field(default_factory=list)
    confidence: float = 0.0
    inferred: bool = False
    via: str = ""                    # the ancestor an inherited answer came from
    verdict: str = ""                # yes | no | unknown (yes/no questions)
    text_hint: str = ""              # for definitions / arithmetic
    chain: List[str] = field(default_factory=list)


class SemanticMemory:
    def __init__(self, lexicon: Lexicon) -> None:
        self.lex = lexicon
        self.slots: Dict[Tuple[str, str], Dict[Tuple[str, str, bool], Fact]] = {}
        self.reverse: Dict[Tuple[str, str], Set[str]] = {}
        self.defs: Dict[str, str] = {}               # definitions learned or told
        self.subj_num: Dict[str, str] = {}           # remembered grammatical number of a subject
        self.proper: Set[str] = set()
        self.ents = EntityStore()                    # the instances ("ball#1") that facts can be about
        self.t = 0
        self.inhibit_read = 0.25
        self.inhibit_told = 0.8

    # ------------------------------------------------------------------ writing
    def _color(self, w: str) -> bool:
        return self.lex.isa_any_pos(w, "color") or any(s.kind == "color" for p in ("a", "n") for s in self.lex.senses(w, p))

    def _conflict(self, rel: str, a: Fact, b_obj: str, b_qty: str) -> bool:
        """Can fact `a` and a new value (b_obj, b_qty) not both be true of the same subject?"""
        if a.obj == b_obj and a.qty == b_qty:
            return False
        if rel == "attr":
            if self._color(a.obj) and self._color(b_obj):
                return True
            return b_obj in self.lex.antonyms(a.obj) or a.obj in self.lex.antonyms(b_obj)
        if rel in ("say", "count", "equals"):
            return True
        if rel == "have":
            return a.obj == b_obj and a.qty != b_qty
        if rel.startswith("live_") or rel in SPATIAL:
            return True
        return False

    def add(self, prop: Prop, told: bool = False, source: str = "read") -> Fact:
        self.t += 1
        rel = "in" if prop.rel in ("be_in",) else prop.rel
        slot_key = (prop.subj, self._slot_rel(rel))
        group = self.slots.setdefault(slot_key, {})
        key = (prop.obj, prop.qty, prop.neg)
        f = group.get(key)
        if f is None:
            f = Fact(obj=prop.obj, qty=prop.qty, neg=prop.neg, obj_num=prop.obj_num, rel=rel)
            group[key] = f
        f.strength += 1.0
        f.count += 1
        f.t = self.t
        f.told = f.told or told
        f.source = source if f.source == "read" else f.source
        f.rel = rel
        self.reverse.setdefault((rel, prop.obj), set()).add(prop.subj)
        if is_instance(prop.obj):
            self.reverse.setdefault((rel, lemma_of(prop.obj)), set()).add(prop.subj)
        self.subj_num.setdefault(prop.subj, prop.subj_num)
        if prop.subj_proper:
            self.proper.add(prop.subj)
        inh = self.inhibit_told if told else self.inhibit_read
        for k, g in group.items():
            if k == key:
                continue
            if g.neg == prop.neg and self._conflict(rel, g, prop.obj, prop.qty):
                g.strength *= (1.0 - inh)
            elif g.obj == prop.obj and g.qty == prop.qty and g.neg != prop.neg:
                g.strength *= (1.0 - inh)         # an assertion and its denial compete
        return f

    @staticmethod
    def _slot_rel(rel: str) -> str:
        return "@loc" if rel in SPATIAL else rel

    # ------------------------------------------------------------------ taxonomy
    def parents(self, x: str) -> List[Tuple[str, str]]:
        """Direct 'is a' parents of x: stored first, then the dictionary's nearest hypernym."""
        out: List[Tuple[str, str]] = []
        for f in self.slots.get((x, "isa"), {}).values():
            if not f.neg and f.strength > 0.3:
                out.append((f.obj, "told"))
        if is_instance(x):
            out.append((lemma_of(x), "kind"))
            return out
        for p in ("n",):
            for s in self.lex.senses(x, p)[:2]:
                for h in s.hypernyms[:1]:
                    out.append((h, "dictionary"))
        return out

    def ancestors(self, x: str, depth: int = 5) -> List[Tuple[str, int]]:
        seen: Dict[str, int] = {}
        frontier = [(x, 0)]
        while frontier:
            cur, d = frontier.pop(0)
            if d >= depth:
                continue
            for par, _ in self.parents(cur):
                if par not in seen and par != x:
                    seen[par] = d + 1
                    frontier.append((par, d + 1))
        # the dictionary lists every ancestor, even where a chain link is missing
        for s in self.lex.senses(x, "n")[:2]:
            for a in s.ancestors:
                seen.setdefault(a, 3)
        return sorted(seen.items(), key=lambda kv: kv[1])

    def is_a(self, x: str, y: str) -> Optional[bool]:
        """True if x is a kind of y; False if both are known and it is not; None if the memory cannot tell."""
        if x == y:
            return True
        if is_instance(x):
            if any(f.obj == y and not f.neg and f.strength > 0.3 for f in self.slots.get((x, "isa"), {}).values()):
                return True
            for f in self.slots.get((x, "isa"), {}).values():
                if f.neg and f.obj == y and f.strength > 0.3:
                    return False
            return self.is_a(lemma_of(x), y)
        if y in {a for a, _ in self.ancestors(x)}:
            return True
        for f in self.slots.get((x, "isa"), {}).values():
            if f.neg and f.obj == y and f.strength > 0.3:
                return False
        known_x = bool(self.lex.senses(x, "n")) or (x, "isa") in self.slots
        known_y = bool(self.lex.senses(y, "n")) or any(r == "isa" and o == y for (r, o) in self.reverse)
        if known_x and known_y and self.lex.best_sense(x, "n") and self.lex.best_sense(y, "n"):
            return False
        return None

    # ------------------------------------------------------------------ reading values out
    def inherits(self, x: str) -> List[Tuple[str, int]]:
        """Where to look when nothing is stored about x: an instance first asks its own kind (no discount), then
        the kind's ancestors; a plain concept goes up its 'is a' chain."""
        if not is_instance(x):
            return self.ancestors(x)
        base = lemma_of(x)
        out = [(base, 0)]
        for a, d in self.ancestors(base):
            out.append((a, d))
        for a, d in self.ancestors(x):
            if a != base and all(a != o for o, _ in out):
                out.append((a, d))
        return out

    def _type_ok(self, obj: str, answer_type: str) -> bool:
        if not answer_type:
            return True
        if answer_type == "color":
            return self._color(obj)
        if answer_type == "sound":
            return self.lex.isa_any_pos(obj, "sound") or self.lex.kind(obj) == "sound"
        if answer_type == "place":
            k = self.lex.kind(obj)
            return k in PLACE_KINDS or self.lex.isa_any_pos(obj, "place") or k == "plant"
        if answer_type == "person_or_animal":
            return True
        return True

    def _collect(self, subj: str, rels: List[str], answer_type: str) -> List[Fact]:
        out: List[Fact] = []
        for (s, r), grp in self.slots.items():
            if s == subj and r in rels:
                for f in grp.values():
                    if not f.neg and f.strength > 0.15 and self._type_ok(f.obj, answer_type):
                        out.append(f)
        return out

    def _rels_for(self, q: Query) -> List[str]:
        rel = q.rel
        if rel == "@loc":
            return ["@loc"]
        if rel.endswith("_@"):
            stem = rel[:-1]
            return sorted({r for (s, r) in self.slots if s == q.subj and r.startswith(stem)}) + ["@loc"]
        return [rel]

    def query(self, q: Query) -> Answer:
        now = self.t
        if q.kind == "arithmetic":
            v = evaluate_arithmetic(q.expr)
            return Answer(text_hint=str(v) if v is not None else "", confidence=1.0 if v is not None else 0.0,
                          via="arithmetic rules")
        if q.kind == "define":
            return self._define(q.subj)
        if q.kind == "opposite":
            ants = []
            for f in self._collect(q.subj, ["antonym"], ""):
                ants.append(f)
            if not ants:
                for a in self.lex.antonyms(q.subj):
                    ants.append(Fact(obj=a, strength=1.0, source="dictionary", rel="antonym"))
            for (s, r), grp in self.slots.items():       # symmetric
                if r == "antonym":
                    for f in grp.values():
                        if f.obj == q.subj and not f.neg:
                            ants.append(Fact(obj=s, strength=f.strength, t=f.t, told=f.told, rel="antonym"))
            ants.sort(key=lambda f: -f.score(now))
            return Answer(items=ants[:1], confidence=1.0 if ants else 0.0, via="dictionary" if ants and ants[0].source == "dictionary" else "memory")
        if q.kind == "count":
            fs = self._collect(q.subj, ["count"], "")
            return self._rank(fs, now)
        if q.kind == "yesno":
            return self._yesno(q)
        if q.want == "subj":
            subs: Dict[str, float] = {}
            for s in self.reverse.get((q.rel, q.obj), set()):
                for f in self.slots.get((s, q.rel), {}).values():
                    if lemma_of(f.obj) == q.obj and not f.neg and f.strength > 0.15:
                        subs[s] = max(subs.get(s, 0.0), f.score(now))
            items = [Fact(obj=s, strength=v, rel=q.rel) for s, v in sorted(subs.items(), key=lambda kv: -kv[1])]
            return Answer(items=items[:3], confidence=1.0 if items else 0.0)
        if q.want == "qty":
            fs = [f for f in self._collect(q.subj, [q.rel], "") if lemma_of(f.obj) == q.obj]
            made = [f for f in fs if is_instance(f.obj)]
            if made:                                  # several particular things: count them
                n = len({f.obj for f in made}) + sum(1 for f in fs if not is_instance(f.obj) and f.qty == "one")
                return Answer(items=[Fact(obj=q.obj, qty=number_word(n), strength=1.0, rel=q.rel,
                                          t=max(f.t for f in made))], confidence=1.0)
            return self._rank(fs, now)
        rels = self._rels_for(q)
        fs = self._collect(q.subj, rels, q.answer_type)
        ans = self._rank(fs, now)
        if ans.items:
            return ans
        # nothing stored: inherit from ancestors (stored "is a" + dictionary hypernyms)
        for anc, depth in self.inherits(q.subj):
            fs = self._collect(anc, rels, q.answer_type)
            if fs:
                a = self._rank(fs, now)
                a.inferred, a.via = True, anc
                a.confidence *= 0.6 ** depth
                return a
        return Answer()

    def _rank(self, fs: List[Fact], now: int) -> Answer:
        if not fs:
            return Answer()
        fs = sorted(fs, key=lambda f: -f.score(now))
        top = fs[0].score(now)
        rivals = sum(f.score(now) for f in fs[1:] if f.rel == fs[0].rel and f.obj != fs[0].obj
                     and self._conflict(f.rel, fs[0], f.obj, f.qty))
        conf = top / (top + rivals) if top + rivals > 0 else 0.0
        return Answer(items=fs, confidence=conf)

    def _define(self, x: str) -> Answer:
        gloss = self.defs.get(x)
        par = [a for a, _ in self.parents(x)]
        for s in self.lex.senses(x, "n")[:1] + self.lex.senses(x, "v")[:1] + self.lex.senses(x, "a")[:1]:
            gloss = gloss or s.gloss
        if not par and not gloss:
            return Answer()
        return Answer(text_hint=gloss or "", chain=par, confidence=1.0 if par or gloss else 0.0,
                      via="memory" if x in self.defs or (x, "isa") in self.slots else "dictionary")

    def _yesno(self, q: Query) -> Answer:
        s, rel, o = q.subj, q.rel, q.obj
        if rel == "isa":
            v = self.is_a(s, o)
            return Answer(verdict={True: "yes", False: "no", None: "unknown"}[v], confidence=1.0 if v is not None else 0.0,
                          chain=[a for a, _ in self.inherits(s)][:4])
        grp = self._collect(s, [self._slot_rel(rel)], "")
        pos = [f for f in grp if lemma_of(f.obj) == lemma_of(o) and (not q.qty or f.qty == q.qty)]
        rivals = [f for f in grp if f not in pos and self._conflict(rel, f, o, q.qty)]
        if pos and rivals and max(f.score(self.t) for f in rivals) > max(f.score(self.t) for f in pos):
            return Answer(verdict="no", confidence=0.9, items=[max(rivals, key=lambda f: f.score(self.t))])
        if pos:
            neg_f = [f for fs in self.slots.get((s, self._slot_rel(rel)), {}).values() for f in [fs] if fs.neg and fs.obj == o]
            if neg_f and max(f.score(self.t) for f in neg_f) > max(f.score(self.t) for f in pos):
                return Answer(verdict="no", confidence=1.0)
            return Answer(verdict="yes", confidence=1.0, items=pos)
        for f in self.slots.get((s, self._slot_rel(rel)), {}).values():
            if f.neg and f.obj == o and f.strength > 0.3:
                return Answer(verdict="no", confidence=1.0)
        # a stored, incompatible value means "no" (the sky is red, so it is not blue)
        for f in grp:
            if self._conflict(rel, f, o, q.qty):
                return Answer(verdict="no", confidence=0.8, items=[f])
        if rel == "attr":
            for anc, depth in self.inherits(s):
                for f in self._collect(anc, ["attr"], ""):
                    if f.obj == o:
                        return Answer(verdict="yes", confidence=0.6 ** depth, inferred=True, via=anc)
        if rel not in ("attr", "isa"):
            for anc, depth in self.inherits(s):
                for f in self._collect(anc, [self._slot_rel(rel)], ""):
                    if lemma_of(f.obj) == lemma_of(o):
                        return Answer(verdict="yes", confidence=0.6 ** depth, inferred=True, via=anc)
        return Answer(verdict="unknown")

    # ------------------------------------------------------------------ inspection / persistence
    def facts_about(self, subj: str) -> List[Tuple[str, Fact]]:
        out = []
        for (s, r), grp in self.slots.items():
            if s == subj:
                for f in grp.values():
                    out.append((r, f))
        return sorted(out, key=lambda x: -x[1].score(self.t))

    def size(self) -> int:
        return sum(len(g) for g in self.slots.values())

    def to_dict(self) -> dict:
        return dict(t=self.t, entities=self.ents.to_dict(), defs=self.defs, subj_num=self.subj_num, proper=sorted(self.proper),
                    facts=[dict(subj=s, rel=r, **{k: getattr(f, k) for k in
                                                  ("obj", "qty", "neg", "strength", "count", "t", "told", "source", "obj_num")})
                           for (s, r), grp in self.slots.items() for f in grp.values()])

    def from_dict(self, d: dict) -> None:
        self.t, self.defs, self.subj_num = d["t"], d["defs"], d["subj_num"]
        self.proper = set(d.get("proper", []))
        self.ents.from_dict(d.get("entities", {}))
        self.slots.clear()
        self.reverse.clear()
        for x in d["facts"]:
            f = Fact(obj=x["obj"], qty=x["qty"], neg=x["neg"], strength=x["strength"], count=x["count"], t=x["t"],
                     told=x["told"], source=x["source"], obj_num=x["obj_num"], rel=x["rel"])
            self.slots.setdefault((x["subj"], x["rel"]), {})[(f.obj, f.qty, f.neg)] = f
            self.reverse.setdefault((x["rel"], f.obj), set()).add(x["subj"])
            if is_instance(f.obj):
                self.reverse.setdefault((x["rel"], lemma_of(f.obj)), set()).add(x["subj"])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            self.from_dict(json.load(fh))
