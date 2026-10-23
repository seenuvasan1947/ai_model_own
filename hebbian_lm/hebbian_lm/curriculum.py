"""A small, *meaningful* child-English language with an exact grammar.

Why synthetic: the grammar is finite and fully enumerated, so we can say with certainty
whether any sentence the model produces is grammatical, whether it was in the training
set, and whether it is a new combination. Words are typed (animals eat foods, people
read books ...), so the sentences make sense rather than being random slot-filling.

The language covers: singular/plural nouns with agreeing determiners (a/an, this/these),
pronouns, subject-verb agreement, present / past / progressive tense, regular and
irregular verbs, negation (do-support), yes/no questions, modals, adjectives (before and
after the noun), prepositions, numbers ("there are three dogs") and basic world facts for
question answering.

Some (subject, verb, object) *combinations* are held out of training entirely; the model
can only get those right by generalising through learned word categories.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

# --------------------------------------------------------------------------- lexicon
ANIMALS = [("dog", "dogs"), ("cat", "cats"), ("bird", "birds"), ("cow", "cows"), ("horse", "horses"),
           ("duck", "ducks"), ("pig", "pigs"), ("frog", "frogs"), ("rabbit", "rabbits"),
           ("lion", "lions"), ("bear", "bears"), ("monkey", "monkeys"), ("mouse", "mice"),
           ("elephant", "elephants"), ("owl", "owls")]
PEOPLE = [("boy", "boys"), ("girl", "girls"), ("man", "men"), ("woman", "women"), ("child", "children"),
          ("teacher", "teachers"), ("baby", "babies"), ("friend", "friends"), ("doctor", "doctors"),
          ("farmer", "farmers")]
FOODS = [("apple", "apples"), ("banana", "bananas"), ("carrot", "carrots"), ("cake", "cakes"),
         ("cookie", "cookies"), ("egg", "eggs"), ("peach", "peaches"), ("pear", "pears"),
         ("sandwich", "sandwiches")]
OBJECTS = [("ball", "balls"), ("book", "books"), ("car", "cars"), ("table", "tables"),
           ("chair", "chairs"), ("door", "doors"), ("box", "boxes"), ("cup", "cups"), ("hat", "hats"),
           ("bag", "bags"), ("toy", "toys"), ("pen", "pens"), ("bed", "beds")]
DRINKS = ["milk", "water", "juice", "tea"]
PLACES = ["park", "garden", "farm", "school", "house", "room", "shop"]
FURNITURE = ["table", "chair", "bed", "box"]

ADJ = {
    "animal": ["big", "small", "little", "old", "young", "fast", "slow", "happy", "sad", "hungry",
               "tired", "brown", "black", "white"],
    "person": ["big", "small", "old", "young", "tall", "short", "happy", "sad", "hungry", "tired",
               "kind", "good"],
    "food": ["big", "small", "red", "green", "yellow", "good", "hot", "cold", "sweet"],
    "object": ["big", "small", "red", "blue", "green", "yellow", "pink", "black", "white", "new",
               "old", "long", "short"],
}

NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

# (base, 3sg, past, -ing, subject kinds, allowed subject heads or None)
INTRANS = [
    ("run", "runs", "ran", "running", {"animal", "person"}, None),
    ("sleep", "sleeps", "slept", "sleeping", {"animal", "person"}, None),
    ("jump", "jumps", "jumped", "jumping", {"animal", "person"}, None),
    ("walk", "walks", "walked", "walking", {"animal", "person"}, None),
    ("play", "plays", "played", "playing", {"animal", "person"}, None),
    ("sit", "sits", "sat", "sitting", {"animal", "person"}, None),
    ("fly", "flies", "flew", "flying", {"animal"}, {"bird", "owl", "duck"}),
    ("swim", "swims", "swam", "swimming", {"animal"}, {"duck", "frog"}),
    ("hop", "hops", "hopped", "hopping", {"animal"}, {"frog", "rabbit", "bird"}),
    ("talk", "talks", "talked", "talking", {"person"}, None),
    ("laugh", "laughs", "laughed", "laughing", {"person"}, None),
    ("smile", "smiles", "smiled", "smiling", {"person"}, None),
    ("work", "works", "worked", "working", {"person"}, None),
]
# (base, 3sg, past, subject kinds, object spec (kinds/lemmas), subject-det style, object-det style)
TRANS = [
    ("eat", "eats", "ate", {"animal", "person"}, {"food"}, "full", "np"),
    ("drink", "drinks", "drank", {"animal", "person"}, {"@drink"}, "full", "mass"),
    ("read", "reads", "read", {"person"}, {"book"}, "full", "np"),
    ("see", "sees", "saw", {"animal", "person"}, {"animal", "person"}, "the", "the"),
    ("like", "likes", "liked", {"animal", "person"}, {"animal", "person"}, "the", "the"),
    ("love", "loves", "loved", {"animal", "person"}, {"animal", "person"}, "the", "the"),
    ("open", "opens", "opened", {"person"}, {"door", "box", "bag", "book"}, "full", "np"),
    ("carry", "carries", "carried", {"person"}, {"box", "bag", "cup", "toy", "ball", "book"}, "full", "np"),
    ("throw", "throws", "threw", {"person"}, {"ball", "hat", "toy"}, "full", "np"),
    ("kick", "kicks", "kicked", {"person"}, {"ball"}, "full", "np"),
    ("help", "helps", "helped", {"person"}, {"person"}, "the", "the"),
    ("feed", "feeds", "fed", {"person"}, {"animal"}, "the", "the"),
]
MOTION_IN = {"run": ["park", "garden", "farm"], "walk": ["park", "garden", "farm"],
             "play": ["park", "garden", "room", "school"], "jump": ["park", "garden", "room"],
             "sleep": ["house", "room"]}
MOTION_TO = {"run": ["park", "school", "shop", "farm"], "walk": ["park", "school", "shop", "garden", "farm", "house"]}
SIT_ON = ["chair", "table", "bed", "box"]

# pronouns: text, subject kinds it can stand for, agreement
PRONOUNS = [("he", {"person"}, "3sg"), ("she", {"person"}, "3sg"), ("it", {"animal", "object"}, "3sg"),
            ("they", {"animal", "person", "object"}, "pl"), ("i", {"person"}, "1sg"),
            ("you", {"person"}, "2"), ("we", {"person"}, "pl")]


@dataclass(frozen=True)
class NP:
    text: str
    kind: str
    head: str
    agr: str          # '3sg' | '1sg' | '2' | 'pl'
    pron: bool = False

    @property
    def be(self) -> str:
        return "is" if self.agr == "3sg" else ("am" if self.agr == "1sg" else "are")

    @property
    def be_past(self) -> str:
        return "was" if self.agr in ("3sg", "1sg") else "were"

    @property
    def do(self) -> str:
        return "does" if self.agr == "3sg" else "do"

    @property
    def sg(self) -> bool:
        return self.agr == "3sg"


def _art(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


class Noun:
    def __init__(self, sg: str, pl: str, kind: str):
        self.sg, self.pl, self.kind = sg, pl, kind


NOUNS: List[Noun] = ([Noun(s, p, "animal") for s, p in ANIMALS] + [Noun(s, p, "person") for s, p in PEOPLE]
                     + [Noun(s, p, "food") for s, p in FOODS] + [Noun(s, p, "object") for s, p in OBJECTS])
BY_LEMMA = {n.sg: n for n in NOUNS}


def _match(noun: Noun, spec: Set[str]) -> bool:
    return noun.kind in spec or noun.sg in spec


def subject_nps(kinds: Set[str], heads: Optional[Set[str]] = None, style: str = "full") -> List[NP]:
    out: List[NP] = []
    for n in NOUNS:
        if n.kind not in kinds or (heads is not None and n.sg not in heads):
            continue
        sg_dets = ["the", "a", "my"] if style == "full" else ["the"]
        pl_dets = ["the", "my", ""] if style == "full" else ["the"]
        for d in sg_dets:
            det = _art(n.sg) if d == "a" else d
            out.append(NP(f"{det} {n.sg}", n.kind, n.sg, "3sg"))
        for d in pl_dets:
            out.append(NP(f"{d} {n.pl}".strip(), n.kind, n.sg, "pl"))
    for text, pk, agr in PRONOUNS:
        if (kinds & pk) and (heads is None or text in ("it", "they")):
            out.append(NP(text, sorted(kinds & pk)[0], text, agr, pron=True))
    return out


def object_nps(spec: Set[str], style: str) -> List[Tuple[str, str]]:
    """(text, head) object phrases."""
    out: List[Tuple[str, str]] = []
    if style == "mass":
        return [(m, m) for m in DRINKS]
    for n in NOUNS:
        if not _match(n, spec):
            continue
        if style == "the":
            out += [(f"the {n.sg}", n.sg), (f"the {n.pl}", n.sg)]
        else:
            out += [(f"the {n.sg}", n.sg), (f"{_art(n.sg)} {n.sg}", n.sg),
                    (f"the {n.pl}", n.sg), (n.pl, n.sg)]
    return out


def _s(*parts: str) -> str:
    return " ".join(p for p in parts if p)


def _heldout(key: tuple, pct: int) -> bool:
    return zlib.crc32(repr(key).encode()) % 100 < pct


# --------------------------------------------------------------------------- facts
COLOR_TRAIN = [("sky", "blue"), ("grass", "green"), ("sun", "yellow"), ("snow", "white"),
               ("apple", "red"), ("coal", "black"), ("milk", "white"), ("tomato", "red"),
               ("sea", "blue"), ("night", "black"), ("pig", "pink"), ("cloud", "white"),
               ("tree", "brown"), ("frog", "green"), ("lemon", "yellow"), ("rose", "red")]
COLOR_HELD = [("cherry", "red"), ("corn", "yellow"), ("leaf", "green"), ("blood", "red"),
              ("sand", "brown"), ("chocolate", "brown"), ("ocean", "blue"), ("strawberry", "red")]
SOUND_TRAIN = [("dog", "woof"), ("cat", "meow"), ("cow", "moo"), ("duck", "quack"), ("pig", "oink"),
               ("lion", "roar"), ("horse", "neigh"), ("frog", "ribbit"), ("owl", "hoot"),
               ("bird", "tweet"), ("mouse", "squeak")]
SOUND_HELD = [("bear", "growl"), ("elephant", "trumpet")]
LEGS_TRAIN = [("dog", "four"), ("cat", "four"), ("cow", "four"), ("horse", "four"), ("pig", "four"),
              ("lion", "four"), ("bird", "two"), ("duck", "two"), ("owl", "two"), ("frog", "four"),
              ("man", "two"), ("woman", "two")]
LEGS_HELD = [("spider", "eight"), ("ant", "six")]
HOME_TRAIN = [("cow", "farm"), ("bird", "nest"), ("frog", "pond"), ("duck", "pond"), ("bear", "forest"),
              ("lion", "jungle"), ("horse", "farm"), ("pig", "farm"), ("owl", "tree")]
HOME_HELD = [("monkey", "jungle"), ("elephant", "jungle")]
OPP_TRAIN = [("big", "small"), ("hot", "cold"), ("up", "down"), ("fast", "slow"), ("happy", "sad"),
             ("day", "night"), ("tall", "short"), ("good", "bad"), ("open", "closed"), ("clean", "dirty")]
OPP_HELD = [("wet", "dry"), ("hard", "soft"), ("full", "empty")]


@dataclass(frozen=True)
class Fact:
    rel: str
    subj: str
    value: str
    statement: str
    question: str
    answer: str


def _facts(rel: str, pairs: Sequence[Tuple[str, str]]) -> List[Fact]:
    """Facts as (statement, question, answer). The answer restates the subject in a full
    sentence, so a plainly stated fact is already the answer to its question."""
    out: List[Fact] = []
    for s, v in pairs:
        if rel == "color":
            st = f"the {s} is {v} ."
            out.append(Fact(rel, s, v, st, f"what color is the {s} ?", st))
        elif rel == "sound":
            st = f"the {s} says {v} ."
            out.append(Fact(rel, s, v, st, f"what does the {s} say ?", st))
        elif rel == "legs":
            st = f"the {s} has {v} legs ."
            out.append(Fact(rel, s, v, st, f"how many legs does the {s} have ?", st))
        elif rel == "home":
            st = f"the {s} lives in the {v} ."
            out.append(Fact(rel, s, v, st, f"where does the {s} live ?", st))
        elif rel == "opp":
            for a, b in ((s, v), (v, s)):
                st = f"the opposite of {a} is {b} ."
                out.append(Fact(rel, a, b, st, f"what is the opposite of {a} ?", st))
    return out


def _sum_facts() -> Tuple[List[Fact], List[Fact]]:
    train, held = [], []
    for a in range(1, 10):
        for b in range(a, 10):
            if a + b > 10:
                continue
            na, nb, nc = NUMBERS[a - 1], NUMBERS[b - 1], NUMBERS[a + b - 1]
            st = f"{na} plus {nb} is {nc} ."
            f = Fact("sum", f"{na} plus {nb}", nc, st, f"what is {na} plus {nb} ?", st)
            (held if _heldout((a, b), 20) else train).append(f)
    return train, held


# --------------------------------------------------------------------------- language
class Language:
    """Enumerates the whole grammar once; provides train/held-out splits and tests."""

    TEMPLATE_WEIGHTS = {
        "intrans_pres": 1.0, "intrans_past": 0.7, "intrans_prog": 0.6, "intrans_neg": 0.5,
        "intrans_q": 0.5, "modal": 0.4, "trans_pres": 1.2, "trans_past": 0.8, "trans_neg": 0.5,
        "trans_q": 0.5, "wh": 0.3, "copula": 1.0, "copula_past": 0.4, "copula_neg": 0.3,
        "copula_q": 0.4, "attributive": 0.7, "locative": 0.6, "demonstrative": 0.4, "there": 0.5,
        "counting": 0.1,
    }

    def __init__(self, holdout_pct: int = 12) -> None:
        self.holdout_pct = holdout_pct
        self.templates: Dict[str, List[Tuple[str, tuple]]] = {}  # name -> [(sentence, sub-keys)]
        self._build()
        self.train_pool: Dict[str, List[str]] = {}
        self.heldout: List[str] = []
        self.heldout_by_template: Dict[str, List[str]] = {}
        for name, items in self.templates.items():
            tr, ho = [], []
            for s, keys in items:
                (ho if any(_heldout(k, holdout_pct) for k in keys) else tr).append(s)
            self.train_pool[name] = tr
            self.heldout_by_template[name] = ho
            self.heldout += ho
        self.grammar: Set[str] = {s for items in self.templates.values() for s, _ in items}
        # facts
        self.facts_train: List[Fact] = []
        self.facts_held: List[Fact] = []
        for rel, tr, ho in (("color", COLOR_TRAIN, COLOR_HELD), ("sound", SOUND_TRAIN, SOUND_HELD),
                            ("legs", LEGS_TRAIN, LEGS_HELD), ("home", HOME_TRAIN, HOME_HELD),
                            ("opp", OPP_TRAIN, OPP_HELD)):
            self.facts_train += _facts(rel, tr)
            self.facts_held += _facts(rel, ho)
        st, sh = _sum_facts()
        self.facts_train += st
        self.facts_held += sh
        # keep the two worlds apart: sentences that are taught facts are never "held out" grammar, and
        # held-out facts never appear in the grammar's training pool
        taught_sents = {f.statement for f in self.facts_train}
        held_sents = {f.statement for f in self.facts_held}
        self.heldout = [s for s in self.heldout if s not in taught_sents]
        for name in self.train_pool:
            self.heldout_by_template[name] = [s for s in self.heldout_by_template[name] if s not in taught_sents]
            self.train_pool[name] = [s for s in self.train_pool[name] if s not in held_sents]
        self.accepted: Set[str] = set(self.grammar)
        for f in self.facts_train + self.facts_held:
            self.accepted.add(f.statement.strip())
            self.accepted.add(f.answer.strip())
            self.accepted.add(f.question.strip())
            self.accepted.add("question : " + f.question)
            self.accepted.add("answer : " + f.answer)
        self.trained: Set[str] = set()   # filled by sample_docs

    # ---------------------------------------------------------------- templates
    def _add(self, name: str, text: str, *keys: tuple) -> None:
        self.templates.setdefault(name, []).append((text, keys))

    def _build(self) -> None:
        # ---- intransitive verbs
        for base, s3, past, ing, kinds, heads in INTRANS:
            for np_ in subject_nps(kinds, heads):
                v = s3 if np_.agr == "3sg" else base
                key = (np_.head, base)
                self._add("intrans_pres", _s(np_.text, v, "."), key)
                self._add("intrans_past", _s(np_.text, past, "."), key)
                self._add("intrans_prog", _s(np_.text, np_.be, ing, "."), key)
                self._add("intrans_neg", _s(np_.text, np_.do, "not", base, "."), key)
                self._add("intrans_neg", _s(np_.text, "did not", base, "."), key)
                self._add("intrans_q", _s(np_.do, np_.text, base, "?"), key)
                self._add("intrans_q", _s("did", np_.text, base, "?"), key)
                self._add("modal", _s(np_.text, "can", base, "."), key)
                self._add("modal", _s("can", np_.text, base, "?"), key)
        # ---- transitive verbs
        for base, s3, past, kinds, ospec, sstyle, ostyle in TRANS:
            spec = {o for o in ospec}
            if "@drink" in spec:
                spec = set()
            subs = subject_nps(kinds, None, sstyle)
            objs = object_nps(spec, ostyle)
            for np_ in subs:
                v = s3 if np_.agr == "3sg" else base
                for otext, ohead in objs:
                    k1, k2 = (np_.head, base), (base, otext)   # hold out subject-verb pairs and verb+object-phrase pairs
                    self._add("trans_pres", _s(np_.text, v, otext, "."), k1, k2)
                    self._add("trans_past", _s(np_.text, past, otext, "."), k1, k2)
                    self._add("trans_neg", _s(np_.text, np_.do, "not", base, otext, "."), k1, k2)
                    self._add("trans_q", _s(np_.do, np_.text, base, otext, "?"), k1, k2)
        # ---- wh questions about eating / drinking / reading
        for np_ in subject_nps({"animal", "person"}, None, "the"):
            if np_.pron and np_.text in ("i", "you", "we"):
                continue
            self._add("wh", _s("what", np_.do, np_.text, "eat", "?"), (np_.head, "eat", "what"))
            self._add("wh", _s("where", np_.do, np_.text, "sleep", "?"), (np_.head, "sleep", "where"))
        # ---- copula + adjectives
        for n in NOUNS:
            for np_ in self._noun_subjects(n):
                for adj in ADJ[n.kind]:
                    key = (np_.head, adj)
                    self._add("copula", _s(np_.text, np_.be, adj, "."), key)
                    self._add("copula_past", _s(np_.text, np_.be_past, adj, "."), key)
                    self._add("copula_neg", _s(np_.text, np_.be, "not", adj, "."), key)
                    self._add("copula_q", _s(np_.be, np_.text, adj, "?"), key)
        for text, pk, agr in PRONOUNS:
            np_ = NP(text, sorted(pk)[0], text, agr, pron=True)
            kinds = pk
            adjs = sorted({a for k in kinds if k in ADJ for a in ADJ[k]}) if text != "it" else ADJ["animal"]
            for adj in adjs:
                key = (text, adj)
                self._add("copula", _s(text, np_.be, adj, "."), key)
                self._add("copula_past", _s(text, np_.be_past, adj, "."), key)
                self._add("copula_neg", _s(text, np_.be, "not", adj, "."), key)
                self._add("copula_q", _s(np_.be, text, adj, "?"), key)
        # ---- attributive adjectives (adjective before the noun)
        for n in NOUNS:
            if n.kind == "food":
                continue
            for adj in ADJ[n.kind]:
                for det in ("the", _art(adj), "my"):
                    for base, s3, past, ing, kinds, heads in INTRANS[:6]:
                        if n.kind in kinds:
                            self._add("attributive", _s(det, adj, n.sg, s3, "."), (n.sg, adj), (n.sg, base))
                for det in ("the", "my", ""):
                    for base, s3, past, ing, kinds, heads in INTRANS[:6]:
                        if n.kind in kinds:
                            self._add("attributive", _s(det, adj, n.pl, base, "."), (n.sg, adj), (n.sg, base))
        # ---- locatives
        for np_ in subject_nps({"animal", "person"}, None, "the"):
            for base, places in MOTION_IN.items():
                s3 = [v for v in INTRANS if v[0] == base][0][1]
                v = s3 if np_.agr == "3sg" else base
                for p in places:
                    self._add("locative", _s(np_.text, v, "in the", p, "."), (np_.head, base), (base, "in", p))
            for base, places in MOTION_TO.items():
                s3 = [x for x in INTRANS if x[0] == base][0][1]
                v = s3 if np_.agr == "3sg" else base
                for p in places:
                    self._add("locative", _s(np_.text, v, "to the", p, "."), (np_.head, base), (base, "to", p))
            v = "sits" if np_.agr == "3sg" else "sit"
            for f in SIT_ON:
                self._add("locative", _s(np_.text, v, "on the", f, "."), (np_.head, "sit"), ("sit", "on", f))
            v = "sleeps" if np_.agr == "3sg" else "sleep"
            for f in ("bed", "chair"):
                self._add("locative", _s(np_.text, v, "on the", f, "."), (np_.head, "sleep"), ("sleep", "on", f))
        for n in NOUNS:
            if n.kind == "object":
                for np_ in self._noun_subjects(n, style="the"):
                    for prep, loc in (("on", "table"), ("on", "chair"), ("on", "bed"), ("in", "room"),
                                      ("in", "house"), ("near", "door"), ("near", "table"), ("under", "table"),
                                      ("under", "chair"), ("under", "bed")):
                        if loc != n.sg:
                            self._add("locative", _s(np_.text, np_.be, prep, "the", loc, "."), (np_.head, prep, loc))
            elif n.kind in ("animal", "person"):
                for np_ in self._noun_subjects(n, style="the"):
                    for prep, loc in (("in", "house"), ("in", "garden"), ("in", "park"), ("in", "room"),
                                      ("near", "door"), ("on", "chair"), ("under", "table"), ("on", "bed"),
                                      ("at", "school"), ("at", "farm")):
                        self._add("locative", _s(np_.text, np_.be, prep, "the", loc, "."), (np_.head, prep, loc))
        # ---- demonstratives and counting
        for n in NOUNS:
            self._add("demonstrative", _s("this is", _art(n.sg), n.sg, "."), (n.sg, "this"))
            self._add("demonstrative", _s("that is", _art(n.sg), n.sg, "."), (n.sg, "that"))
            self._add("demonstrative", _s("these are", n.pl, "."), (n.sg, "these"))
            self._add("demonstrative", _s("those are", n.pl, "."), (n.sg, "those"))
            self._add("there", _s("there is one", n.sg, "."), (n.sg, "one"))
            for i, num in enumerate(NUMBERS[1:], start=2):
                self._add("there", _s("there are", num, n.pl, "."), (n.sg, num))
        for i in range(2, 11):
            self._add("counting", _s(*NUMBERS[:i], "."), ("count", i))

    def _noun_subjects(self, n: Noun, style: str = "full") -> List[NP]:
        out = []
        sg_dets = ["the", "a", "my"] if style == "full" else ["the"]
        pl_dets = ["the", "my", ""] if style == "full" else ["the"]
        for d in sg_dets:
            out.append(NP(f"{_art(n.sg) if d == 'a' else d} {n.sg}", n.kind, n.sg, "3sg"))
        for d in pl_dets:
            out.append(NP(f"{d} {n.pl}".strip(), n.kind, n.sg, "pl"))
        return out

    # ---------------------------------------------------------------- sampling
    def sample_docs(self, n_sentences: int, seed: int = 0, fact_reps: int = 6,
                    include_facts: bool = True) -> List[str]:
        """Training documents: grammar sentences (one per doc) plus fact statements and
        question/answer pairs. Held-out grammar combinations and held-out facts never appear."""
        rng = random.Random(seed)
        names = [n for n in self.TEMPLATE_WEIGHTS if self.train_pool.get(n)]
        weights = [self.TEMPLATE_WEIGHTS[n] for n in names]
        docs: List[str] = []
        for _ in range(n_sentences):
            name = rng.choices(names, weights)[0]
            s = rng.choice(self.train_pool[name])
            self.trained.add(s)
            docs.append(s)
        if include_facts:
            for f in self.facts_train:
                for r in range(fact_reps):
                    if r % 2 == 0:
                        docs.append(f"question : {f.question}\nanswer : {f.answer}")
                    else:
                        docs.append(f.statement)
                self.trained.add(f.statement)
        rng.shuffle(docs)
        return docs

    def stage_docs(self, stage: int, n: int, seed: int = 0) -> List[str]:
        """Curriculum stages (simple -> richer). Stage 1: short present-tense sentences.
        Stage 2: + past/progressive/adjectives/prepositions. Stage 3: everything + facts."""
        groups = {
            1: ["intrans_pres", "copula", "demonstrative", "counting"],
            2: ["intrans_pres", "intrans_past", "intrans_prog", "trans_pres", "copula", "copula_past",
                "attributive", "locative", "demonstrative", "there", "counting"],
            3: list(self.TEMPLATE_WEIGHTS),
        }[stage]
        rng = random.Random(seed + stage)
        names = [g for g in groups if self.train_pool.get(g)]
        weights = [self.TEMPLATE_WEIGHTS[g] for g in names]
        docs = []
        for _ in range(n):
            name = rng.choices(names, weights)[0]
            s = rng.choice(self.train_pool[name])
            self.trained.add(s)
            docs.append(s)
        if stage == 3:
            for f in self.facts_train:
                for r in range(6):
                    docs.append(f"question : {f.question}\nanswer : {f.answer}" if r % 2 == 0 else f.statement)
                self.trained.add(f.statement)
        rng.shuffle(docs)
        return docs

    # ---------------------------------------------------------------- gold labels (for category analysis)
    def gold_tags(self) -> Dict[str, str]:
        tags: Dict[str, str] = {}
        for n in NOUNS:
            tags[n.sg] = f"N_{n.kind}_sg"
            tags[n.pl] = f"N_{n.kind}_pl"
        for m in DRINKS:
            tags[m] = "N_drink"
        for p in PLACES + ["nest", "pond", "forest", "jungle", "tree"]:
            tags.setdefault(p, "N_place")
        for d in ("the", "my"):
            tags[d] = "DET"
        tags["a"] = tags["an"] = "DET_sg"
        for d in ("this", "that"):
            tags[d] = "DEM_sg"
        for d in ("these", "those"):
            tags[d] = "DEM_pl"
        for base, s3, past, ing, *_ in INTRANS:
            tags[base], tags[s3], tags[past], tags[ing] = "V_base", "V_3sg", "V_past", "V_ing"
        for base, s3, past, *_ in TRANS:
            tags.setdefault(base, "V_base")
            tags.setdefault(s3, "V_3sg")
            tags.setdefault(past, "V_past")
        for kind, adjs in ADJ.items():
            for a in adjs:
                tags.setdefault(a, "ADJ")
        for w in ("is", "are", "am", "was", "were"):
            tags[w] = "BE"
        for w in ("do", "does", "did", "can"):
            tags[w] = "AUX"
        for w in ("in", "on", "near", "under", "at", "to"):
            tags[w] = "PREP"
        for t, _, _ in PRONOUNS:
            tags[t] = "PRON"
        for n in NUMBERS:
            tags[n] = "NUM"
        for w in ("what", "where", "how", "many"):
            tags[w] = "WH"
        for w in (".", "?", ":", "<s>"):
            tags[w] = "PUNCT"
        tags["not"] = "NEG"
        return tags

    # ---------------------------------------------------------------- minimal pairs
    def minimal_pairs(self, n: int, seed: int = 0, from_heldout: bool = True) -> List[Tuple[str, str, str]]:
        """(good, bad, kind). Good sentences are grammatical; bad ones differ by one error
        and are verified to be outside the language."""
        rng = random.Random(seed)
        src = self.heldout if from_heldout else sorted(self.trained)
        agr = {}
        for base, s3, *_ in INTRANS + [(t[0], t[1]) for t in TRANS]:
            agr[base], agr[s3] = s3, base
        agr.update({"is": "are", "are": "is", "does": "do", "do": "does", "was": "were", "were": "was"})
        sg2pl = {x.sg: x.pl for x in NOUNS}
        pl2sg = {x.pl: x.sg for x in NOUNS}
        foods = {x.sg for x in NOUNS if x.kind == "food"} | {x.pl for x in NOUNS if x.kind == "food"}
        objs = [x for x in NOUNS if x.kind == "object"]
        animate_pl = {x.pl for x in NOUNS if x.kind in ("animal", "person")}
        animate = animate_pl | {x.sg for x in NOUNS if x.kind in ("animal", "person")}
        intr = {f for v in INTRANS for f in v[:3]}
        pairs: List[Tuple[str, str, str]] = []
        tries = 0
        while len(pairs) < n and tries < n * 50:
            tries += 1
            good = rng.choice(src)
            toks = good.split()
            kind = rng.choice(["agreement", "number", "selection", "subject", "order"])
            bad = None
            if kind == "agreement":
                idxs = [i for i, t in enumerate(toks) if t in agr and t not in ("read",)]
                if idxs:
                    i = rng.choice(idxs)
                    # only the finite verb (first agreeing token) is a clean error
                    i = idxs[0]
                    bad = " ".join(toks[:i] + [agr[toks[i]]] + toks[i + 1:])
            elif kind == "number":
                idxs = [i for i in range(1, len(toks)) if toks[i - 1] in ("a", "an", "this", "that") and toks[i] in sg2pl]
                idxs2 = [i for i in range(1, len(toks)) if toks[i - 1] in ("these", "those") and toks[i] in pl2sg]
                if idxs:
                    i = idxs[0]
                    bad = " ".join(toks[:i] + [sg2pl[toks[i]]] + toks[i + 1:])
                elif idxs2:
                    i = idxs2[0]
                    bad = " ".join(toks[:i] + [pl2sg[toks[i]]] + toks[i + 1:])
            elif kind == "selection":
                idxs = [i for i, t in enumerate(toks) if t in foods and i > 0 and toks[0] != "what"]
                if idxs and any(v in toks for v in ("eat", "eats", "ate")):
                    i = idxs[0]
                    o = rng.choice(objs)
                    rep = o.pl if toks[i] in {x.pl for x in NOUNS} and toks[i] not in {x.sg for x in NOUNS} else o.sg
                    bad = " ".join(toks[:i] + [rep] + toks[i + 1:])
            elif kind == "subject":
                idxs = [i for i, t in enumerate(toks[:3]) if t in animate]
                if idxs and any(t in intr for t in toks) and toks[-1] == "." and toks[0] not in ("there",):
                    i = idxs[0]
                    o = rng.choice(objs)
                    rep = o.pl if toks[i] in animate_pl else o.sg
                    bad = " ".join(toks[:i] + [rep] + toks[i + 1:])
            else:
                if len(toks) >= 4:
                    j = rng.randrange(0, len(toks) - 2)
                    t2 = toks[:]
                    t2[j], t2[j + 1] = t2[j + 1], t2[j]
                    bad = " ".join(t2)
            if bad and bad != good and bad not in self.accepted:
                pairs.append((good, bad, kind))
        return pairs

    def stats(self) -> dict:
        return dict(grammar_sentences=len(self.grammar), heldout=len(self.heldout),
                    templates={k: len(v) for k, v in self.templates.items()},
                    facts_train=len(self.facts_train), facts_held=len(self.facts_held))
