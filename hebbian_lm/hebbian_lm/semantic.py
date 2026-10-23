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
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .comprehension import Prop, Query, evaluate_arithmetic, number_word
from .entities import EntityStore, is_instance, lemma_of
from .lexicon import Lexicon
from .roles import TIME_UNITS, same_verb

NON_ACTIONS = {"attr", "isa", "count", "antonym", "equals", "can", "have", "@loc"}
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
    subj: str = ""                   # whose fact this is (kept so events can be found without knowing the subject)
    adj: Dict[str, str] = field(default_factory=dict)       # roles: where / when / with / about / duration ...
    stab: float = 0.0                # half-life of the trace in time units: grows each time it is re-read or recalled
    reh: int = 0                     # times it was strengthened again or successfully recalled
    tense: str = "present"

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
        self.the_names: Set[str] = set()             # names that go with "the": the Titanic, the British, the Great Fire of London
        self.ents = EntityStore()                    # the instances ("ball#1") that facts can be about
        self.t = 0
        self.rng = random.Random(0)                  # traces differ a little in how durable they are
        self.clock = 0.0                             # time units that have passed (see ``advance``); events alone do not age memory
        self.base_stab = 120.0                       # half-life of a fresh fact, in time units; told facts get twice
        self.stab_growth = 1.6                       # each re-reading multiplies the half-life (spacing / consolidation)
        self.recall_boost = 0.4                      # strength added when a fact is successfully used to answer
        self.prune_below = 0.03
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
        if rel.startswith("live_") or rel in SPATIAL or rel.endswith(("_in", "_at", "_on")):
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
            f = Fact(obj=prop.obj, qty=prop.qty, neg=prop.neg, obj_num=prop.obj_num, rel=rel,
                     stab=self.base_stab * (2.0 if told else 1.0) * math.exp(self.rng.gauss(0.0, 0.45)))
            group[key] = f
        else:
            f.reh += 1
            f.stab = min(self.base_stab * 25, max(f.stab, self.base_stab) * self.stab_growth)
        f.tense = prop.tense
        f.subj = prop.subj
        if prop.adj:
            f.adj.update(prop.adj)
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
        if prop.obj_proper:
            self.proper.add(prop.obj)
        if prop.subj_the:
            self.the_names.add(prop.subj)
        if prop.obj_the:
            self.the_names.add(prop.obj)
        inh = self.inhibit_told if told else self.inhibit_read
        for k, g in group.items():
            if k == key:
                continue
            if g.neg == prop.neg and self._conflict(rel, g, prop.obj, prop.qty):
                g.strength *= (1.0 - inh)
            elif g.obj == prop.obj and g.qty == prop.qty and g.neg != prop.neg:
                g.strength *= (1.0 - inh)         # an assertion and its denial compete
        return f

    # ------------------------------------------------------------------ time: forgetting and rehearsal
    def advance(self, units: float) -> int:
        """Time passes. Every trace loses strength with its own half-life (a trace that was re-read or recalled often has
        a long one); traces that fall below ``prune_below`` are forgotten for good. Returns how many were forgotten."""
        if units <= 0:
            return 0
        self.clock += units
        gone = 0
        for key in list(self.slots):
            grp = self.slots[key]
            for k in list(grp):
                f = grp[k]
                f.strength *= 0.5 ** (units / (f.stab or self.base_stab))
                if f.strength < self.prune_below:
                    del grp[k]
                    gone += 1
            if not grp:
                del self.slots[key]
        if gone:
            self.reverse.clear()
            for (subj, rel), grp in self.slots.items():
                for f in grp.values():
                    self.reverse.setdefault((rel, f.obj), set()).add(subj)
                    if is_instance(f.obj):
                        self.reverse.setdefault((rel, lemma_of(f.obj)), set()).add(subj)
        return gone

    def rehearse(self, facts: List[Fact]) -> None:
        """Using a fact to answer a question strengthens it (the testing effect) and slows its forgetting."""
        for f in facts:
            if f.source == "dictionary" or not f.stab:
                continue
            f.strength += self.recall_boost
            f.stab = min(self.base_stab * 25, f.stab * 1.25)
            f.reh += 1
            f.t = self.t

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
        if q.kind == "event":
            return self._event(q)
        if q.kind == "happened":
            return self._happened(q)
        if q.kind == "define":
            return self._define(q.subj)
        if q.kind == "actions":
            fs = [f for r, f in self.facts_about(q.subj) if r not in NON_ACTIONS and not f.neg and f.strength > 0.15]
            best: Dict[str, Fact] = {}                          # one fact per verb: the object if there is one, else a place
            for f in fs:
                v = f.rel.split("_")[0]
                cur = best.get(v)
                if cur is None or (("_" in cur.rel) and ("_" not in f.rel)):
                    best[v] = f
            return Answer(items=list(best.values())[:4], confidence=1.0 if best else 0.0)
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
            tenses: Dict[str, str] = {}
            for s in self.reverse.get((q.rel, q.obj), set()):
                for f in self.slots.get((s, q.rel), {}).values():
                    if lemma_of(f.obj) == q.obj and not f.neg and f.strength > 0.15:
                        subs[s] = max(subs.get(s, 0.0), f.score(now))
                        tenses[s] = f.tense
            items = [Fact(obj=s, strength=v, rel=q.rel, tense=tenses.get(s, "present"))
                     for s, v in sorted(subs.items(), key=lambda kv: -kv[1])]
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
        if not fs and q.answer_type == "sound":                  # "what did they say ?" may have a non-sound answer
            fs = self._collect(q.subj, rels, "")
        if not fs and not q.rel.endswith("_@") and q.rel not in ("@loc", "attr", "have", "can"):
            preps = sorted({r for (s, r) in self.slots if s == q.subj and r.startswith(q.rel + "_")})
            fs = self._collect(q.subj, preps, "") if preps else []        # "what did he jump ?" -> jumped on the pail
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

    # ------------------------------------------------------------------ events
    @staticmethod
    def same_name(a: str, b: str) -> bool:
        """'mutiny' ~ 'sepoy mutiny', 'buster' ~ 'buster bear', 'soldiers' ~ 'soldier'."""
        a, b = lemma_of(a), lemma_of(b)
        if a == b:
            return True
        ta, tb = set(a.split()), set(b.split())
        return bool(ta and tb) and (ta <= tb or tb <= ta)

    _ROLE_KEYS = {"n": ("n",), "where": ("where", "to", "from"), "when": ("when",), "why": ("about", "for", "because"), "with": ("with", "by"),
                  "about": ("about",), "duration": ("duration",), "by": ("by",), "from": ("from",), "for": ("for",)}

    def _event(self, q: Query) -> Answer:
        keys = self._ROLE_KEYS.get(q.want, (q.want,))
        found = []
        for (s, r), grp in self.slots.items():
            if q.want == "who_inf":
                if not same_verb(q.rel, r):
                    continue
                for f in grp.values():
                    if not f.neg and f.strength > 0.15 and f.adj.get("inf", "").split(" ")[0] == q.obj:
                        found.append((True, f.score(self.t), f))
                continue
            if not self.same_name(q.subj, s):
                continue
            exact = r.split("_")[0] == q.rel
            if q.rel == "attr":
                if r != "attr":
                    continue
            elif not same_verb(q.rel, r):
                continue
            for f in grp.values():
                if q.want == "duration" and not f.adj.get("duration") and lemma_of(f.obj) in TIME_UNITS and f.qty:
                    found.append((exact, f.score(self.t), f))                 # "the flight lasted twelve seconds"
                    continue
                if f.neg or f.strength <= 0.15 or not any(f.adj.get(k) for k in keys):
                    continue
                if q.rel == "attr" and f.obj not in (q.alts or (q.obj,)):
                    continue
                if q.obj and q.rel != "attr" and q.obj.split()[0] not in lemma_of(f.obj).split():
                    continue
                found.append((exact, f.score(self.t), f))
        found.sort(key=lambda x: (-x[0], -x[1]))
        items = [f for _, _, f in found]
        return Answer(items=items[:3], confidence=1.0 if items else 0.0)

    def _happened(self, q: Query) -> Answer:
        """'what happened in 1858 ?' / 'what happened after the mutiny ?': the events whose time matches."""
        prep, _, rest = q.obj.partition(" ")
        want = set(rest.split()) - {"the", "a", "an"}
        items = []
        for (s, r), grp in self.slots.items():
            if r in ("attr", "isa", "count", "antonym", "equals", "can", "have"):
                continue
            for f in grp.values():
                w = f.adj.get("when", "")
                if f.neg or f.strength <= 0.15 or not w:
                    continue
                wp, _, wr = w.partition(" ")
                wt = set(wr.split()) - {"the", "a", "an"}
                if prep in ("after", "before", "during", "until", "since"):
                    if wp == prep and any(self.same_name(x, y) for x in want for y in wt):
                        items.append(f)
                elif want and want <= wt | set(wr.split()):
                    items.append(f)
        items.sort(key=lambda f: f.t)
        return Answer(items=items[:4], confidence=1.0 if items else 0.0)

    def _rank(self, fs: List[Fact], now: int) -> Answer:
        if not fs:
            return Answer()
        fs = sorted(fs, key=lambda f: -f.score(now))
        top = fs[0].score(now)
        rivals = sum(f.score(now) for f in fs[1:] if f.rel == fs[0].rel and f.obj != fs[0].obj
                     and self._conflict(f.rel, fs[0], f.obj, f.qty))
        conf = top / (top + rivals) if top + rivals > 0 else 0.0
        conf *= 0.5 + 0.5 * min(1.0, fs[0].strength)          # a faded trace is a less certain one
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
        return dict(t=self.t, clock=self.clock, entities=self.ents.to_dict(), defs=self.defs, subj_num=self.subj_num, proper=sorted(self.proper), the_names=sorted(self.the_names),
                    facts=[dict(subj=s, rel=r, **{k: getattr(f, k) for k in
                                                  ("obj", "qty", "neg", "strength", "count", "t", "told", "source", "obj_num", "stab", "reh", "tense", "adj")})
                           for (s, r), grp in self.slots.items() for f in grp.values()])

    def from_dict(self, d: dict) -> None:
        self.t, self.defs, self.subj_num = d["t"], d["defs"], d["subj_num"]
        self.proper = set(d.get("proper", []))
        self.clock = d.get("clock", 0.0)
        self.the_names = set(d.get("the_names", []))
        self.ents.from_dict(d.get("entities", {}))
        self.slots.clear()
        self.reverse.clear()
        for x in d["facts"]:
            f = Fact(obj=x["obj"], qty=x["qty"], neg=x["neg"], strength=x["strength"], count=x["count"], t=x["t"],
                     told=x["told"], source=x["source"], obj_num=x["obj_num"], rel=x["rel"],
                     stab=x.get("stab", self.base_stab), reh=x.get("reh", 0), tense=x.get("tense", "present"),
                     subj=x["subj"], adj=x.get("adj", {}))
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
