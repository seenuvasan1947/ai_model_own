"""Comprehension engine: turns a sentence into propositions, and a question into a query.

This is the grammar rule set of the system. It uses the dictionary (``lexicon.py``) for parts of speech and word
kinds, a small Viterbi tagger with hand-written transition preferences to resolve ambiguous words in context, a
chunker for noun phrases and verb groups, and pattern rules for questions. It understands the plain declarative
and question sentences of the child-English world (and a good deal of everyday English around them):

    "the sky is blue"            -> (sky, attr, blue)
    "a zebra is an animal"       -> (zebra, isa, animal)
    "the dogs eat bones"         -> (dog, eat, bone)
    "the cow lives in the farm"  -> (cow, live_in, farm)
    "the bird has two legs"      -> (bird, have, leg, qty=two)
    "there are three cats"       -> (cat, count, qty=three)
    "the opposite of big is small" -> (big, antonym, small)
    "he runs"  (after "the boy runs")  -> pronoun resolved to the boy

Not handled (by design, it is a rule system, not a parser of arbitrary English): relative clauses, passives,
subordination, quantifier scope, metaphor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from .entities import EntityStore, is_instance, lemma_of
from .numbers import WORD_VALUE as NUMWORDS, parse_number
from .lexicon import (AUX, BE_FORMS, FUNCTION_POS, HAVE_FORMS, NUMBER_WORDS, PL_DET, PRONOUNS, SG_DET, Analysis, Lexicon)

CASED_TOKEN = re.compile(r"[A-Za-z]+(?:'[a-z]+)?|[0-9]+|[.,?!:;]|[\"\u201c\u201d]|[+*/=\u00d7\u00f7]|(?<![A-Za-z])-(?![A-Za-z])")
_SYMBOL = {"+": ["plus"], "-": ["minus"], "*": ["times"], "\u00d7": ["times"], "/": ["divided", "by"], "\u00f7": ["divided", "by"],
           "=": ["equals"]}
END = {".", "?", "!"}
MALE = {"boy", "man", "king", "father", "brother", "he", "tom", "ben", "sam", "max", "jack"}
FEMALE = {"girl", "woman", "queen", "mother", "sister", "she", "anna", "lily", "mia", "emma", "zoe"}


# --------------------------------------------------------------------------- data classes
@dataclass
class NP:
    head: str                       # lemma ("dog"), "" if unresolved
    number: str = "sg"              # sg | pl
    det: str = ""
    adjs: List[str] = field(default_factory=list)
    qty: str = ""                   # number word / quantifier
    pron: str = ""                  # surface pronoun, if the NP was one
    proper: bool = False
    kind: str = ""
    of: str = ""                    # "the opposite of X": X
    text: str = ""
    owner: str = ""                 # possessor key: "tom" in "tom's ball" (a name or an entity id)
    ref: str = ""                   # the entity id this phrase refers to ("ball#2"), once bound
    indef: bool = False             # "a ball": introduces a new instance when the clause is about a particular thing

    @property
    def key(self) -> str:
        """What the memory stores facts under: the instance if there is one, otherwise the lemma."""
        return self.ref or self.head

    def copy(self, **kw) -> "NP":
        return replace(self, **kw)


@dataclass
class Prop:
    subj: str
    rel: str
    obj: str = ""
    qty: str = ""
    neg: bool = False
    tense: str = "present"
    subj_num: str = "sg"
    obj_num: str = "sg"
    subj_proper: bool = False

    def key(self) -> Tuple[str, str, str, str, bool]:
        return (self.subj, self.rel, self.obj, self.qty, self.neg)


@dataclass
class Query:
    kind: str                       # fact | define | opposite | yesno | arithmetic | count | list
    subj: str = ""
    rel: str = ""
    obj: str = ""
    want: str = "obj"               # obj | subj | qty | yesno
    answer_type: str = ""           # color | sound | place | ... (typing of the answer)
    qty: str = ""
    neg: bool = False
    num: str = "sg"
    obj_num: str = "sg"
    subj_proper: bool = False
    expr: Tuple[int, str, int] = (0, "", 0)
    raw: str = ""


@dataclass
class Tok:
    text: str
    cap: bool
    tag: str = ""
    lemma: str = ""
    feats: Dict[str, str] = field(default_factory=dict)
    known: bool = True
    analysis: Optional[Analysis] = None


# --------------------------------------------------------------------------- tagger
FUNCTION_NAMES = set(FUNCTION_POS)
PARTICLES = {"down", "up", "out", "off", "away", "back", "over", "around", "along", "forward", "together", "aside", "about"}
INTENSIFIERS = {"very", "so", "too", "quite", "rather", "really", "just", "almost", "most", "more", "much", "not", "always", "never"}
LINKING = {"feel", "look", "seem", "become", "grow", "get", "turn", "sound", "smell", "taste", "remain", "stay", "appear"}
TAGS = ["POSS", "DET", "ADJ", "N", "PN", "V", "AUX", "PREP", "PRON", "CONJ", "WH", "NEG", "NUM", "QUANT", "ADV", "PUNCT"]
_POS_TAG = {"n": "N", "v": "V", "a": "ADJ", "r": "ADV", "det": "DET", "pron": "PRON", "aux": "AUX", "prep": "PREP",
            "conj": "CONJ", "wh": "WH", "neg": "NEG", "num": "NUM", "quant": "QUANT"}

_T = {
    "START": {"DET": 0, "PRON": 0, "N": -0.1, "PN": 0, "WH": 0, "AUX": -0.1, "NUM": -0.3, "ADJ": -0.6, "V": -2,
              "ADV": -1, "PREP": -1, "QUANT": -0.5},
    "DET": {"ADJ": 0, "N": 0, "NUM": -0.4, "ADV": -1.2, "QUANT": -1},
    "ADJ": {"N": 0, "ADJ": -0.3, "CONJ": -0.4, "PUNCT": -0.2, "PREP": -0.5, "ADV": -1, "NUM": -1},
    "N": {"POSS": -0.1, "V": 0, "AUX": 0, "PREP": 0, "N": -1.0, "CONJ": 0, "PUNCT": 0, "ADV": -0.4, "NEG": -1, "DET": -0.7,
          "PRON": -1, "ADJ": -0.8, "NUM": -0.8},
    "PN": {"POSS": 0, "V": 0, "AUX": 0, "PREP": 0, "N": -0.6, "CONJ": 0, "PUNCT": 0, "ADV": -0.4, "NEG": -1, "DET": -0.7, "PN": -0.3},
    "PRON": {"V": 0, "AUX": 0, "ADV": -0.3, "NEG": -1, "PUNCT": -0.3, "PREP": -0.5, "CONJ": -0.3, "DET": -1.0},
    "V": {"DET": 0, "N": -0.3, "PREP": 0, "ADV": -0.2, "PRON": -0.2, "ADJ": -0.6, "NUM": -0.3, "PUNCT": 0,
          "CONJ": -0.3, "NEG": -0.5, "V": -3, "PN": -0.2, "QUANT": -0.5},
    "AUX": {"V": 0, "ADJ": 0, "DET": 0, "NEG": 0, "N": -0.5, "PRON": -0.2, "NUM": -0.3, "PREP": -0.2, "ADV": -0.2,
            "QUANT": -0.2, "AUX": -0.4, "PN": -0.2, "WH": -2},
    "PREP": {"V": -2.0, "AUX": -2.5, "DET": 0, "N": 0, "PN": 0, "NUM": -0.3, "ADJ": -0.2, "PRON": -0.2, "QUANT": -0.3},
    "NEG": {"V": 0, "ADJ": 0, "DET": -0.3, "N": -0.5, "AUX": -0.5, "ADV": -0.3, "PREP": -0.5},
    "CONJ": {"DET": -0.1, "N": -0.1, "PN": -0.1, "PRON": -0.1, "V": -0.3, "ADJ": -0.2, "ADV": -0.3, "NUM": -0.3, "AUX": -0.3},
    "WH": {"AUX": 0, "V": -0.3, "N": -0.2, "ADJ": -0.2, "QUANT": 0, "NUM": -0.2, "DET": -0.3, "PRON": -0.3},
    "NUM": {"N": 0, "ADJ": -0.2, "NUM": -0.2, "QUANT": -0.3, "PUNCT": -0.3, "V": -1, "PREP": -0.3, "AUX": -0.4, "CONJ": -0.3},
    "QUANT": {"N": 0, "ADJ": -0.2, "PREP": -0.3, "NUM": -0.3, "AUX": -0.3, "V": -0.5},
    "ADV": {"V": 0, "ADJ": 0, "ADV": -0.2, "PUNCT": 0, "DET": -0.3, "PREP": -0.3, "AUX": -0.3, "N": -0.8, "PRON": -0.3, "NEG": -0.3},
    "POSS": {"N": 0, "ADJ": 0, "NUM": -0.3, "PN": -0.5, "DET": -2.0},
    "PUNCT": {},
}


# after these verbs "to" is a preposition ("went to school"); after others it marks an infinitive ("wanted to be")
_TO_PREP_VERBS = {"go", "come", "walk", "run", "return", "move", "travel", "belong", "listen", "talk", "look", "turn", "get",
                  "fly", "swim", "climb", "jump", "hurry", "give", "send", "bring", "take", "say", "speak", "write", "point",
                  "reply", "refer", "lead", "stand", "sit", "creep", "rush", "step", "drive", "ride", "sail", "head", "show"}


def _trans(prev: str, cur: str, infinitive: bool = False) -> float:
    if cur == "PUNCT":
        return -0.1
    t = _T.get(prev, {}).get(cur, -3.0)
    if infinitive:                              # "to" + verb: "a word to say", "wanted to be"
        t += 2.0 if cur in ("V", "AUX") else (-0.5 if cur == "N" else 0.0)
    return t


_NT = {"can't": ["can", "not"], "won't": ["will", "not"], "shan't": ["shall", "not"], "ain't": ["is", "not"]}
_CLITIC = {"'ll": "will", "'re": "are", "'ve": "have", "'m": "am", "'d": "would"}
_S_IS_AFTER = {"he", "she", "it", "that", "there", "what", "who", "where", "here", "how", "one"}


def tokenize_cased(text: str) -> List[Tok]:
    """Words and punctuation with capitalisation kept; contractions are expanded (didn't -> did not, he'll ->
    he will, it's -> it is); a possessive 's stays as its own token (tom's ball -> tom 's ball)."""
    raw = CASED_TOKEN.findall(text)
    out: List[Tok] = []
    for k, w in enumerate(raw):
        low = w.lower()
        cap = w[:1].isupper()
        if low in _SYMBOL:
            out += [Tok(x, False) for x in _SYMBOL[low]]
        elif low in _NT:
            out += [Tok(x, cap and i == 0) for i, x in enumerate(_NT[low])]
        elif low.endswith("n't") and len(low) > 3:
            out += [Tok(low[:-3], cap), Tok("not", False)]
        elif "'" in low and low.split("'")[0] and "'" + low.split("'", 1)[1] in _CLITIC:
            head, tail = low.split("'", 1)
            out += [Tok(head, cap), Tok(_CLITIC["'" + tail], False)]
        elif low.endswith("'s") and len(low) > 2:
            head = low[:-2]
            out.append(Tok(head, cap))
            out.append(Tok("is" if head in _S_IS_AFTER else "'s", False))
        else:
            out.append(Tok(low, cap))
    return out


def _after_be(toks: List[Tok], i: int) -> bool:
    j = i - 1
    while j >= 0 and toks[j].text in {"just", "still", "always", "never", "now", "quickly", "slowly", "not", "also", "soon"}:
        j -= 1
    return j >= 0 and toks[j].text in BE_FORMS


def tag_sentence(toks: List[Tok], lex: Lexicon) -> None:
    """Viterbi over each token's candidate readings; fills tag/lemma/feats on the tokens."""
    cands: List[List[Tuple[str, Analysis, float]]] = []
    for i, t in enumerate(toks):
        if t.text in END or t.text in {",", ":", ";", '"'}:
            cands.append([("PUNCT", Analysis(t.text, t.text, "punct", {}, [], True), 0.0)])
            continue
        if t.text == "'s":
            cands.append([("POSS", Analysis("'s", "'s", "poss", {}, [], True), 0.0)])
            continue
        if t.text.isdigit():
            cands.append([("NUM", Analysis(t.text, t.text, "num", {"value": t.text}, [], True), 0.0)])
            continue
        analyses = lex.analyze(t.text)
        opts: List[Tuple[str, Analysis, float]] = []
        for rank, a in enumerate(analyses):
            tag = _POS_TAG.get(a.pos, "N")
            if a.pos == "n" and (t.text in lex.proper or (t.cap and i > 0)):
                tag = "PN"
            cost = min(1.5, 0.35 * (analyses[0].score - a.score)) if rank else 0.0
            if not a.known:
                cost = 0.9
            if rank and analyses[0].pos in ("det", "pron") and analyses[0].score >= 10:
                cost = max(cost, 3.0)            # "his", "her", "that" are determiners/pronouns, not rare nouns
            if tag == "V" and a.feats.get("form") == "ing" and _after_be(toks, i):
                cost -= 1.5                      # "was climbing": after a form of be, -ing is a verb
            opts.append((tag, a, cost))
        if analyses and not analyses[0].known:
            # unknown word: let the grammar decide whether it is a noun, adjective or verb
            a = analyses[0]
            opts = [("N", a, 0.9), ("ADJ", Analysis(t.text, t.text, "a", {}, [], False), 1.2),
                    ("V", Analysis(t.text, a.lemma if a.pos == "v" else t.text, "v",
                                   a.feats if a.pos == "v" else {"form": "base"}, [], False), 1.3)]
            if t.cap and i > 0:
                opts = [("PN", a, 0.2)] + opts
            if a.pos == "v":
                opts = [("V", a, 0.9), ("N", Analysis(t.text, t.text, "n", {"number": "sg"}, [], False), 1.0)]
        # a capitalised word in mid-sentence is a name, not a verb or adjective; so is the first word of "Little Joe"
        starts_name = (i == 0 and t.cap and i + 1 < len(toks) and toks[i + 1].cap and toks[i + 1].text.isalpha()
                       and t.text not in FUNCTION_NAMES)
        if (t.cap and i > 0 or starts_name) and not t.text.isdigit() and t.text not in FUNCTION_NAMES:
            first = analyses[0] if analyses else None
            nm = Analysis(t.text, t.text, "n", {"number": "sg"}, first.senses if first and first.pos == "n" else [], True)
            opts = [("PN", nm, 0.0)]
        if t.text == "that" and i > 0:                  # "a crash that made ...", "knew that ...": a linking word
            opts = opts + [("CONJ", Analysis("that", "that", "conj", {}, [], True), 0.4)]
        cands.append(opts)
    n = len(toks)
    inf = [False] * n
    for i in range(1, n):
        if toks[i - 1].text == "to":
            before = lex.analyze(toks[i - 2].text)[0] if i >= 2 and toks[i - 2].text.isalpha() else None
            inf[i] = not (before is not None and before.pos == "v" and before.lemma in _TO_PREP_VERBS)
    best: List[Dict[int, Tuple[float, int]]] = []
    for i in range(n):
        layer: Dict[int, Tuple[float, int]] = {}
        for j, (tag, a, cost) in enumerate(cands[i]):
            if i == 0:
                layer[j] = (-cost + _trans("START", tag), -1)
            else:
                bs, bk = max(((best[i - 1][k][0] + _trans(cands[i - 1][k][0], tag, inf[i]), k) for k in best[i - 1]),
                             key=lambda x: x[0])
                layer[j] = (bs - cost, bk)
        best.append(layer)
    if n == 0:
        return
    j = max(best[-1], key=lambda k: best[-1][k][0])
    for i in range(n - 1, -1, -1):
        tag, a, _ = cands[i][j]
        t = toks[i]
        t.tag, t.analysis, t.lemma, t.feats, t.known = tag, a, a.lemma, dict(a.feats), a.known
        j = best[i][j][1]


# --------------------------------------------------------------------------- noun phrases and discourse
def noun_number(head_tok: Tok, det: str, qty: str) -> str:
    if qty and qty in NUMBER_WORDS and NUMBER_WORDS[qty] != 1:
        return "pl"
    if det in PL_DET or qty in ("many", "few", "several", "all", "most"):
        return "pl"
    if det in SG_DET:
        return "sg"
    return head_tok.feats.get("number", "sg")


@dataclass
class Discourse:
    """Who was mentioned recently, for pronoun resolution."""
    entities: List[NP] = field(default_factory=list)
    subjects: List[NP] = field(default_factory=list)      # who was the subject of recent clauses (preferred antecedents)

    def mention(self, np: NP, subject: bool = False) -> None:
        if np.head and not np.pron:
            self.entities = [e for e in self.entities if e.key != np.key]
            self.entities.insert(0, np)
            del self.entities[8:]
            if subject:
                self.subjects = [e for e in self.subjects if e.key != np.key]
                self.subjects.insert(0, np)
                del self.subjects[4:]

    def resolve(self, pron: str, lex: Lexicon) -> Optional[NP]:
        info = PRONOUNS.get(pron)
        if info is None:
            return None
        num, kind, gender = info
        seen = set()
        for e in self.subjects + self.entities:        # the subject of the last clause is the likeliest antecedent
            if e.key in seen:
                continue
            seen.add(e.key)
            ek = e.kind or lex.kind(e.head)
            if pron in ("he", "him"):
                if e.number == "sg" and (e.head in MALE or ((ek == "person" or e.proper) and e.head not in FEMALE)):
                    return e
            elif pron == "she":
                if e.number == "sg" and e.head in FEMALE:
                    return e
            elif pron == "it":
                if ek != "person" and not e.proper and e.number == "sg":
                    return e
            elif pron in ("they", "them"):
                if e.number == "pl":
                    return e
        if pron in ("they", "them") and len(self.entities) >= 2:
            return self.entities[0]
        return None


class ClauseParser:
    """Reads tagged tokens into NPs, verb groups and complements."""

    def __init__(self, lex: Lexicon, discourse: Optional[Discourse] = None, ents: Optional[EntityStore] = None,
                 creating: bool = True, prefer: Optional[Dict[str, str]] = None) -> None:
        self.lex = lex
        self.discourse = discourse if discourse is not None else Discourse()
        self.ents = ents                    # None: no instances, every noun is its kind (the older behaviour)
        self.creating = creating            # reading statements may create instances; questions never do
        self.prefer = prefer or {}          # lemma -> entity id chosen by the user after a clarifying question
        self.ambiguous: List[Tuple[str, List[str]]] = []     # (lemma, candidate ids) of unresolved "the X"
        self._log: Dict[int, NP] = {}
        self._extra: List[Prop] = []

    # ---- noun phrases
    _POSS_DET = {"his": "he", "her": "she", "its": "it", "their": "they"}
    _NOT_THINGS = {"color", "sound", "number", "quantity", "time"}

    def read_np(self, toks: List[Tok], i: int) -> Tuple[Optional[NP], int]:
        """A noun phrase, with its possessor ("tom's red ball", "his ball") and, when entities are on, bound to
        the instance it refers to."""
        np, k = self._read_simple_np(toks, i)
        if np is None or np.pron:
            return np, k
        n = len(toks)
        if k < n and toks[k].tag == "POSS" and k + 1 < n:
            inner, k2 = self._read_simple_np(toks, k + 1)
            if inner is not None and not inner.pron:
                self._bind(np)
                owner = np.head if np.proper else np.key
                inner.owner, inner.text = owner, f"{np.text} 's {inner.text}"
                np, k = inner, k2
        elif np.det in self._POSS_DET:
            ante = self.discourse.resolve(self._POSS_DET[np.det], self.lex)
            if ante is not None:
                np.owner = ante.key
            np.det = "the"
        self._bind(np)
        self._log[id(toks[i])] = np
        return np, k

    def _concrete(self, np: NP) -> bool:
        return bool(np.head) and not np.proper and not np.pron and not np.of and np.number == "sg" \
            and self.lex.kind(np.head) not in self._NOT_THINGS

    def _bind(self, np: NP) -> None:
        """Entities on: point a noun phrase at the instance it names, if there is one."""
        if self.ents is None or np.ref or not self._concrete(np):
            return
        if np.owner:
            e = self.ents.find(np.head, np.owner)
            if e is None and self.creating:
                e = self.ents.new(np.head, owner=np.owner, number=np.number, adjs=np.adjs)
                self._extra.append(Prop(np.owner, "have", e.id, "one", False, "present", "sg", "sg",
                                        not is_instance(np.owner)))
            if e is not None:
                np.ref = e.id
            return
        if np.det in ("a", "an") or (not np.det and np.qty == "one"):
            np.indef = True
            return
        if np.det in ("the", "this", "that"):
            cands = [e.id for e in self.ents.of(np.head)]
            if not cands:
                return
            if np.head in self.prefer and self.prefer[np.head] in cands:
                np.ref = self.prefer[np.head]
            elif len(cands) == 1:
                np.ref = cands[0]
            elif self.creating:
                recent = {x.key for x in self.discourse.entities}
                np.ref = next((c for c in cands if c in recent), cands[0])
            else:
                self.ambiguous.append((np.head, cands))

    def _introduce(self, np: NP, subj: NP, owned: bool) -> None:
        """"tom has a red ball": a particular thing is being introduced, so it gets its own instance.
        A generic subject ("a bird has a beak", "the bird has a beak") describes the kind and creates nothing."""
        if self.ents is None or not self.creating or np is None or np.ref or not np.indef:
            return
        if not (subj.proper or is_instance(subj.key)):
            return
        owner = subj.key if owned else ""
        e = self.ents.find(np.head, owner, np.adjs) if owned else None   # telling it again strengthens, not duplicates
        if e is None:
            e = self.ents.new(np.head, owner=owner, number=np.number, adjs=np.adjs)
        np.ref = e.id
        self.discourse.mention(np)

    def _read_simple_np(self, toks: List[Tok], i: int) -> Tuple[Optional[NP], int]:
        n = len(toks)
        if i >= n:
            return None, i
        t = toks[i]
        if t.tag == "PRON":
            np = NP(head=t.text, pron=t.text, number=("pl" if PRONOUNS.get(t.text, ("sg",))[0] in ("pl",) else "sg"),
                    text=t.text)
            return np, i + 1
        det, qty, adjs = "", "", []
        j = i
        while j < n and toks[j].tag in ("DET", "QUANT", "NUM"):
            if toks[j].tag == "DET":
                det = toks[j].text
            else:
                qty = toks[j].text
            j += 1
        while j < n and toks[j].tag == "ADJ" and j + 1 < n and toks[j + 1].tag in ("ADJ", "N", "PN", "NUM"):
            adjs.append(toks[j].lemma)
            j += 1
        k = j
        while k < n and toks[k].tag in ("N", "PN"):
            k += 1
        if k == j:
            if j > i and toks[j - 1].tag == "NUM":     # "five" alone is a number, not a phrase
                return None, i
            return None, i
        head = toks[k - 1]
        proper = head.tag == "PN"
        name = head.lemma
        if proper:
            m = k - 1
            while m > j and toks[m - 1].tag == "PN":
                m -= 1
            name = " ".join(x.text for x in toks[m:k])
        np = NP(head=name, number=noun_number(head, det, qty), det=det, adjs=adjs, qty=qty, proper=proper,
                text=" ".join(x.text for x in toks[i:k]))
        s = head.analysis.senses[0] if head.analysis and head.analysis.senses else None
        np.kind = s.kind if s else ("person" if proper else "")
        # "the opposite of big"
        if head.lemma in ("opposite", "antonym") and k + 1 < n and toks[k].text == "of":
            np.of = toks[k + 1].lemma
            k += 2
        return np, k

    def resolve(self, np: Optional[NP]) -> Optional[NP]:
        if np is None or not np.pron:
            return np
        if np.pron in ("i", "you", "we", "me", "us"):
            return np
        ante = self.discourse.resolve(np.pron, self.lex)
        if ante is not None:
            return ante.copy()
        return np

    # ---- clause -> propositions
    def parse_declarative(self, toks: List[Tok]) -> List[Prop]:
        self._log, self._extra = {}, []
        props = self._parse_declarative(toks)
        props += self._extra
        # adjectives inside noun phrases ("a red ball") are properties of the noun
        seen = {(p.subj, p.rel, p.obj) for p in props}
        if self.ents is not None:
            for np in self._log.values():
                if np.adjs and np.head and not np.proper and not np.pron:
                    for a in np.adjs:
                        if (np.key, "attr", a) not in seen:
                            seen.add((np.key, "attr", a))
                            props.append(Prop(np.key, "attr", a, subj_num=np.number))
            return props
        i = 0
        toks2 = [t for t in toks if t.tag != "PUNCT"]
        while i < len(toks2):
            if toks2[i].tag == "ADJ" and i + 1 < len(toks2) and toks2[i + 1].tag in ("N", "ADJ"):
                j = i
                while j < len(toks2) and toks2[j].tag == "ADJ":
                    j += 1
                if j < len(toks2) and toks2[j].tag == "N" and i > 0 and toks2[i - 1].tag in ("DET", "NUM", "QUANT", "PREP", "AUX", "V"):
                    for k in range(i, j):
                        key = (toks2[j].lemma, "attr", toks2[k].lemma)
                        if key not in seen:
                            seen.add(key)
                            props.append(Prop(toks2[j].lemma, "attr", toks2[k].lemma,
                                              subj_num=toks2[j].feats.get("number", "sg")))
                i = j
            i += 1
        return props

    def _parse_declarative(self, toks: List[Tok]) -> List[Prop]:
        toks = [t for t in toks if t.tag != "PUNCT" or t.text == ","]
        n = len(toks)
        props: List[Prop] = []
        if n == 0:
            return props
        # "there is/are (NUM) N"
        if n >= 3 and toks[0].text == "there" and toks[1].text in ("is", "are"):
            np, _ = self.read_np(toks, 2)
            if np is not None:
                self.discourse.mention(np)
                qty = np.qty or ("one" if np.number == "sg" else "")
                return [Prop(np.head, "count", "", qty, subj_num=np.number)]
            return props
        # "the opposite of X is Y"
        subj, i = self.read_np(toks, 0)
        if subj is None:
            return props
        # coordination: "the boy and the girl"
        subjects = [subj]
        while i + 1 < n and toks[i].text == "and":
            s2, j = self.read_np(toks, i + 1)
            if s2 is None:
                break
            subjects.append(s2)
            i = j
        subj = self.resolve(subj)
        subjects[0] = subj
        if len(subjects) > 1:
            subj = subj.copy(number="pl")
        # verb group
        neg, tense, modal = False, "present", ""
        j = i
        verb_tok: Optional[Tok] = None
        be = False
        while j < n and toks[j].tag in ("AUX", "NEG", "ADV"):
            t = toks[j]
            if t.tag == "ADV" and be and verb_tok is None and t.text not in INTENSIFIERS and self.lex.senses(t.lemma, "a") \
                    and not any(x.tag == "V" for x in toks[j + 1:j + 2]):
                break                                   # "was late": the adverb-looking word is the predicate adjective
            if t.tag == "NEG":
                neg = True
            elif t.tag == "AUX":
                if t.text in BE_FORMS:
                    be = True
                    if t.text in ("was", "were"):
                        tense = "past"
                elif t.text in ("do", "does", "did"):
                    if t.text == "did":
                        tense = "past"
                elif t.text in HAVE_FORMS:
                    verb_tok = t
                    if t.text == "had":
                        tense = "past"
                else:
                    modal = t.text
            j += 1
            if verb_tok is not None:
                break
        if verb_tok is None and j < n and toks[j].tag == "V":
            verb_tok = toks[j]
            j += 1
            if verb_tok.feats.get("form") == "past" or verb_tok.text in ("ran", "ate", "saw"):
                tense = "past" if not (be and verb_tok.feats.get("form") == "ing") else tense
        rest = toks[j:]
        subj_head = subj.key
        for sub in subjects:
            self.discourse.mention(sub, subject=True)
        out: List[Prop] = []
        # ---- copula
        if be and verb_tok is None:
            out += self._copula(subj, rest, neg, tense)
        elif be and verb_tok is not None and verb_tok.feats.get("form") == "ing":
            out += self._verb_props(subj, verb_tok.lemma, rest, neg, tense="present")
        elif be and verb_tok is not None and verb_tok.feats.get("form") in ("past", "pp") and verb_tok.tag == "V" \
                and verb_tok.text not in HAVE_FORMS:
            out += self._passive(subj, verb_tok, rest, neg)
        elif verb_tok is not None:
            lem = verb_tok.lemma
            if modal:
                if modal in ("can", "could"):
                    out.append(Prop(subj_head, "can", lem, neg=neg, subj_num=subj.number))
                if modal not in ("can", "could") or any(t.tag != "PUNCT" for t in rest):
                    out += self._verb_props(subj, lem, rest, neg, "future" if modal in ("will", "shall", "would") else tense)
            elif lem == "have" or verb_tok.text in HAVE_FORMS:
                out += self._have(subj, rest, neg, tense)
            else:
                out += self._verb_props(subj, lem, rest, neg, tense)
        for s in subjects[1:]:
            out += [Prop(s.key, p.rel, p.obj, p.qty, p.neg, p.tense, s.number, p.obj_num) for p in out[:1]]
        proper = {x.key for x in subjects if x.proper}
        for p in out:
            if p.subj in proper:
                p.subj_proper = True
        return out

    def _passive(self, subj: NP, verb: Tok, rest: List[Tok], neg: bool) -> List[Prop]:
        """"the nut was eaten by the squirrel" -> (squirrel, eat, nut); "the statue was unveiled" -> (statue, attr, unveiled)."""
        for i, t in enumerate(rest):
            if t.text == "by":
                agent, _ = self.read_np(rest, i + 1)
                if agent is not None and not agent.pron:
                    return [Prop(agent.key, verb.lemma, subj.key, "", neg, "past", agent.number, subj.number, agent.proper)]
        return [Prop(subj.key, "attr", verb.text, neg=neg, tense="past", subj_num=subj.number)]

    def _copula(self, subj: NP, rest: List[Tok], neg: bool, tense: str) -> List[Prop]:
        sn = subj.number
        if rest and rest[0].tag == "V" and rest[0].feats.get("form") != "ing" and self.lex.senses(rest[0].lemma, "a"):
            rest[0].tag = "ADJ"             # "the sky is blue": after 'be' a word with an adjective reading is one
        elif rest and rest[0].tag == "ADV" and rest[0].text not in INTENSIFIERS and self.lex.senses(rest[0].lemma, "a") \
                and not (len(rest) > 1 and rest[1].tag == "ADJ"):
            rest[0].tag = "ADJ"             # "Buster was late that morning"
        if subj.of and rest:          # "the opposite of big is small"
            np, _ = self.read_np(rest, 0)
            w = rest[-1].lemma
            if np is not None:
                w = np.head
            elif rest[-1].tag in ("ADJ", "N", "ADV"):
                w = rest[-1].lemma
            return [Prop(subj.of, "antonym", w, neg=neg)]
        if not rest:
            return []
        # adjective(s)
        if rest[0].tag == "ADJ" or (rest[0].tag == "ADV" and len(rest) > 1 and rest[1].tag == "ADJ"):
            adj = next(t for t in rest if t.tag == "ADJ")
            return [Prop(subj.key, "attr", adj.lemma, neg=neg, tense=tense, subj_num=sn)]
        if rest[0].tag == "PREP":
            np, _ = self.read_np(rest, 1)
            if np is not None:
                return [Prop(subj.key, rest[0].text, np.head, neg=neg, tense=tense, subj_num=sn)]
            return []
        np, k = self.read_np(rest, 0)
        if np is not None:
            self.discourse.mention(np)
            # noun complement: "X is a Y" = isa; "X is (the) Y" with a colour/number is an attribute
            if np.det in ("a", "an") or np.number == "pl" or not np.det:
                s = self.lex.best_sense(np.head, "n")
                if s is not None and (s.kind == "color" or "color" in s.ancestors):
                    return [Prop(subj.key, "attr", np.head, neg=neg, tense=tense, subj_num=sn)]
                return [Prop(subj.key, "isa", np.head, neg=neg, tense=tense, subj_num=sn, obj_num=np.number)]
            return [Prop(subj.key, "isa", np.head, neg=neg, tense=tense, subj_num=sn, obj_num=np.number)]
        # bare number/adverb complement
        if rest[0].tag == "NUM":
            return [Prop(subj.key, "equals", rest[0].text, neg=neg)]
        return []

    def _have(self, subj: NP, rest: List[Tok], neg: bool, tense: str) -> List[Prop]:
        np, _ = self.read_np(rest, 0)
        if np is None:
            return []
        self._introduce(np, subj, owned=True)
        return [Prop(subj.key, "have", np.key, np.qty or ("one" if np.number == "sg" else ""), neg, tense,
                     subj.number, np.number, subj.proper)]

    def _verb_props(self, subj: NP, lemma: str, rest: List[Tok], neg: bool, tense: str) -> List[Prop]:
        out: List[Prop] = []
        sn = subj.number
        i = 0
        obj: Optional[NP] = None
        while rest and rest[0].text in PARTICLES and rest[0].tag != "N":
            rest = rest[1:]                       # "sat down", "stole up": the particle is not an object
        if lemma in LINKING and rest and rest[0].tag in ("ADJ", "ADV", "V") and self.lex.senses(rest[0].lemma, "a"):
            adj = [t for t in rest if t.tag in ("ADJ", "ADV", "V") and self.lex.senses(t.lemma, "a")]
            return [Prop(subj.key, lemma, "", "", neg, tense, sn)] + \
                [Prop(subj.key, "attr", adj[0].lemma, neg=neg, tense=tense, subj_num=sn)]
        if lemma == "say" and rest and all(t.tag == "PUNCT" for t in rest[1:]):
            w = next((t for t in rest if t.tag in ("N", "PN", "ADJ", "ADV", "V")), None)
            if w is not None:
                return [Prop(subj.key, "say", w.lemma if w.tag != "V" else w.text, neg=neg, tense=tense, subj_num=sn)]
        if rest and rest[0].tag != "PREP":
            obj, i = self.read_np(rest, 0)
            obj = self.resolve(obj)
            if obj is not None:
                self._introduce(obj, subj, owned=False)
                self.discourse.mention(obj)
                objs = [obj]
                while i + 1 < len(rest) and rest[i].text == "and":
                    o2, j = self.read_np(rest, i + 1)
                    if o2 is None:
                        break
                    objs.append(o2)
                    i = j
                for o in objs:
                    out.append(Prop(subj.key, lemma, o.key, o.qty, neg, tense, sn, o.number, subj.proper))
        # prepositional adjuncts: "lives in the farm", "walks to the park"
        while i < len(rest):
            if rest[i].tag == "PREP" and rest[i].text != "of":
                np, k = self.read_np(rest, i + 1)
                if np is not None:
                    out.append(Prop(subj.key, f"{lemma}_{rest[i].text}", np.head, "", neg, tense, sn, np.number))
                    i = k
                    continue
            i += 1
        if not out:
            out.append(Prop(subj.key, lemma, "", "", neg, tense, sn))
        return out

    # ---- questions about numbers
    def _number_question(self, w: List[str], raw: str) -> Optional[Query]:
        """what is 27 plus 15 | what comes after 39 | what comes before forty | count from 3 to 8 | which is bigger , 7 or 12"""
        isnum = lambda x: x.isdigit() or x in NUMWORDS or x in ("hundred", "thousand", "and")
        def numeral(words: List[str]) -> Optional[int]:
            while words and words[0] == "and":
                words = words[1:]
            while words and words[-1] == "and":
                words = words[:-1]
            return parse_number(words) if words else None
        ops = {"plus": "+", "add": "+", "minus": "-", "subtract": "-", "times": "*", "multiplied": "*", "divided": "/"}
        for i, x in enumerate(w):
            if x in ops and 0 < i < len(w) - 1:
                left, j = [], i - 1
                while j >= 0 and isnum(w[j]):
                    left.insert(0, w[j]); j -= 1
                right, k = [], i + 1
                if k < len(w) and w[k] in ("by", "from", "to"):
                    k += 1
                while k < len(w) and isnum(w[k]):
                    right.append(w[k]); k += 1
                a, b = numeral(left), numeral(right)
                if a is not None and b is not None:
                    if x == "subtract":                    # "subtract 3 from 10"
                        a, b = b, a
                    return Query("arithmetic", expr=(a, ops[x], b), raw=raw)
        def after(word: str) -> Optional[int]:
            if word in w:
                k = w.index(word) + 1
                tail = []
                while k < len(w) and isnum(w[k]):
                    tail.append(w[k]); k += 1
                return numeral(tail)
            return None
        for word, op in (("after", "succ"), ("before", "pred")):
            if word in w and ("comes" in w or "come" in w or "number" in w or "is" in w or "next" in w):
                n = after(word)
                if n is not None:
                    return Query("number", expr=(n, op, 0), raw=raw)
        if "next" in w and "number" in w:
            n = after("after") if "after" in w else None
            if n is not None:
                return Query("number", expr=(n, "succ", 0), raw=raw)
        if w and w[0] == "count":
            if "to" in w and "from" in w:
                a, b = after("from"), after("to")
            else:
                a, b = 0, after("to")
            if a is not None and b is not None:
                return Query("number", expr=(a, "count", b), raw=raw)
        cmp_words = {"bigger": "gt", "larger": "gt", "greater": "gt", "more": "gt", "smaller": "lt", "less": "lt", "fewer": "lt", "biggest": "gt", "largest": "gt", "smallest": "lt"}
        cw = next((x for x in w if x in cmp_words), None)
        if cw is not None:
            if "or" in w:
                i = w.index("or")
                left, j = [], i - 1
                while j >= 0 and isnum(w[j]):
                    left.insert(0, w[j]); j -= 1
                right, k = [], i + 1
                while k < len(w) and isnum(w[k]):
                    right.append(w[k]); k += 1
                a, b = numeral(left), numeral(right)
                if a is not None and b is not None and w[0] in ("which", "what"):
                    return Query("number", expr=(a, cmp_words[cw] + "?", b), raw=raw)
            if "than" in w and w[0] in ("is", "are"):
                i = w.index("than")
                left = [x for x in w[1:w.index(cw)] if isnum(x)]
                right = []
                k = i + 1
                while k < len(w) and isnum(w[k]):
                    right.append(w[k]); k += 1
                a, b = numeral(left), numeral(right)
                if a is not None and b is not None:
                    return Query("number", expr=(a, cmp_words[cw] + "yn", b), raw=raw)
        return None

    # ---- questions
    def parse_question(self, toks: List[Tok]) -> Optional[Query]:
        toks = [t for t in toks if t.tag != "PUNCT"]
        n = len(toks)
        if n == 0:
            return None
        raw = " ".join(t.text for t in toks)
        w0 = toks[0].text
        nq = self._number_question([t.text for t in toks], raw)
        if nq is not None:
            return nq
        if w0 == "how" and n > 2 and toks[1].text == "many":
            np, k = self.read_np(toks, 2)
            if np is None:
                return None
            rest = toks[k:]
            if rest and rest[0].text in ("does", "do", "did"):
                sub, k2 = self.read_np(rest, 1)
                sub = self.resolve(sub)
                if sub is not None and len(rest) > k2 and rest[k2].lemma in ("have",):
                    return Query("fact", subj=sub.key, rel="have", obj=np.head, want="qty", num=sub.number,
                                 subj_proper=sub.proper, raw=raw)
                if sub is not None and len(rest) > k2:
                    return Query("fact", subj=sub.key, rel=rest[k2].lemma, obj=np.head, want="qty", raw=raw)
            if rest and rest[0].text in ("are", "is") and len(rest) > 1 and rest[1].text == "there":
                return Query("count", subj=np.head, want="qty", raw=raw)
            return None
        if w0 in ("what", "which") and n >= 3 and toks[1].text in ("color", "colour"):
            sub, _ = self.read_np(toks, 3 if toks[2].text in BE_FORMS else 2)
            sub = self.resolve(sub)
            if sub is not None and toks[2].text in set(BE_FORMS) | {"does", "do"}:
                return Query("fact", subj=sub.key, rel="attr", want="obj", answer_type="color", num=sub.number,
                             subj_proper=sub.proper, raw=raw)
            return None
        if w0 == "what" and n >= 3 and toks[1].text in BE_FORMS:
            np, k = self.read_np(toks, 2)
            if np is not None and np.of:
                return Query("opposite", subj=np.of, raw=raw)
            if np is not None:
                return Query("define", subj=np.head, num=np.number, raw=raw)
            return None
        if w0 in ("what", "where") and n >= 4 and toks[1].text in ("does", "do", "did") and toks[-1].text.isalpha() \
                and self.lex.senses(toks[-1].lemma, "v"):
            toks[-1].tag = "V"                    # "what did crash make ?": the last word is the verb, not part of a noun group
            toks[-1].lemma = self.lex.analyze(toks[-1].text)[0].lemma if toks[-1].lemma == toks[-1].text else toks[-1].lemma
        if w0 == "what" and n >= 3 and toks[1].text in ("does", "do", "did"):
            sub, k = self.read_np(toks, 2)
            sub = self.resolve(sub)
            if sub is None or k >= n:
                return None
            v = toks[k]
            if v.lemma == "do" and k + 1 == n:
                return Query("actions", subj=sub.key, num=sub.number, subj_proper=sub.proper, raw=raw)
            if v.lemma == "mean":
                return Query("define", subj=sub.key, raw=raw)
            at = "sound" if v.lemma == "say" else ""
            return Query("fact", subj=sub.key, rel=v.lemma, want="obj", answer_type=at, num=sub.number,
                         subj_proper=sub.proper, raw=raw)
        if w0 == "who" and n >= 3:
            v = toks[1]
            if v.text in BE_FORMS:
                np, _ = self.read_np(toks, 2)
                if np is not None:
                    return Query("define", subj=np.head, raw=raw)
                return None
            np, _ = self.read_np(toks, 2)
            if np is not None:
                return Query("fact", rel="have" if v.text in HAVE_FORMS else v.lemma, obj=np.head, want="subj",
                             answer_type="person_or_animal",
                             obj_num=np.number, raw=raw)
            return None
        if w0 == "where" and n >= 3:
            if toks[1].text in BE_FORMS:
                np, _ = self.read_np(toks, 2)
                np = self.resolve(np)
                if np is not None:
                    return Query("fact", subj=np.key, rel="@loc", want="obj", answer_type="place", num=np.number,
                                 subj_proper=np.proper, raw=raw)
            if toks[1].text in ("does", "do", "did"):
                sub, k = self.read_np(toks, 2)
                sub = self.resolve(sub)
                if sub is not None and k < n:
                    return Query("fact", subj=sub.key, rel=toks[k].lemma + "_@", want="obj", answer_type="place",
                                 num=sub.number, subj_proper=sub.proper, raw=raw)
            return None
        # yes/no questions: "is the dog big", "is a zebra an animal", "does the dog eat bones", "can the bird fly"
        if toks[0].tag == "AUX":
            a = toks[0].text
            sub, k = self.read_np(toks, 1)
            sub = self.resolve(sub)
            if sub is None:
                return None
            rest = toks[k:]
            if a in BE_FORMS:
                props = self._copula(sub, rest, False, "present") if rest else []
                if props:
                    p = props[0]
                    return Query("yesno", subj=p.subj, rel=p.rel, obj=p.obj, num=sub.number, subj_proper=sub.proper, raw=raw)
                return None
            if a in AUX and a not in ("do", "does", "did") and rest:
                return Query("yesno", subj=sub.key, rel="can", obj=rest[0].lemma, raw=raw)
            if a in ("do", "does", "did") and rest and (rest[0].tag == "V" or rest[0].lemma == "have"):
                props = self._verb_props(sub, rest[0].lemma, rest[1:], False, "present")
                if props:
                    p = props[0]
                    return Query("yesno", subj=p.subj, rel=p.rel, obj=p.obj, qty=p.qty, num=sub.number, raw=raw)
            return None
        return None


def evaluate_arithmetic(expr: Tuple[int, str, int]) -> Optional[int]:
    a, op, b = expr
    return {"+": a + b, "-": a - b, "*": a * b}.get(op)


NUM_NAMES = {v: k for k, v in NUMBER_WORDS.items() if v <= 20}
NUM_NAMES.update({30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
                  100: "hundred"})


def number_word(n: int) -> str:
    if n in NUM_NAMES:
        return NUM_NAMES[n]
    if n < 0:
        return "minus " + number_word(-n)
    if 20 < n < 100:
        return NUM_NAMES[n // 10 * 10] + " " + NUM_NAMES[n % 10]
    return str(n)
