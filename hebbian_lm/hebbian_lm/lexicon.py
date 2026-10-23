"""Vocabulary and grammar rule engine: the part of the system that already "knows" a dictionary.

Three pieces, all deterministic and inspectable:

* ``Morphology``: a rule set for English inflection (plural -s/-es/-ies/-ves, irregular plurals, verb
  -s/-ed/-ing and irregular verbs, comparatives). Given a word form it proposes (lemma, part of speech,
  grammatical features), e.g. ``mice -> mouse, noun, plural`` and ``flies -> fly, verb, 3sg``.
* ``Lexicon``: looks a lemma up and returns its senses: part of speech, semantic kind (animal, person,
  food, colour ...), a definition, hypernyms (``zebra -> equine -> ... -> animal``), synonyms, antonyms.
  Two backends: a bundled mini dictionary (works offline; covers the demo world) and optionally the real
  WordNet database (``python fetch_wordnet.py``).
* Closed-class word lists (determiners, pronouns, auxiliaries, prepositions, number words ...) that the
  comprehension engine uses as grammar.

When the language model meets a word it has never seen, it asks this engine what the word means and what kind
of word it is, and learns that straight away (see ``workspace.py``).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------- closed-class grammar words
DETERMINERS = {"the", "a", "an", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these",
               "those", "some", "every", "each", "no", "any", "another"}
SG_DET = {"a", "an", "this", "that", "every", "each", "another"}
PL_DET = {"these", "those", "many", "few", "several"}
PRONOUNS = {"he": ("sg", "person", "m"), "she": ("sg", "person", "f"), "it": ("sg", "thing", ""),
            "they": ("pl", "any", ""), "i": ("sg1", "person", ""), "you": ("2", "person", ""),
            "we": ("pl", "person", ""), "him": ("sg", "person", "m"), "them": ("pl", "any", ""),
            "me": ("sg1", "person", ""), "us": ("pl", "person", "")}
SUBJ_PRON = {"he", "she", "it", "they", "i", "you", "we"}
BE_FORMS = {"is": "sg", "are": "pl", "am": "sg1", "was": "sg", "were": "pl", "be": "", "been": "", "being": ""}
AUX = {"do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must", "shall"}
HAVE_FORMS = {"has": "sg", "have": "pl", "had": ""}
PREPOSITIONS = {"in", "on", "at", "under", "near", "to", "from", "of", "with", "by", "over", "into", "behind",
                "above", "below", "inside", "outside", "beside", "between", "through", "across", "toward"}
CONJ = {"and", "or", "but", "because", "if", "then", "while", "so", "plus", "minus", "times", "divided"}
WH = {"what", "who", "whom", "where", "when", "why", "how", "which", "whose"}
NEG = {"not", "never", "n't"}
QUANT = {"many", "much", "few", "several", "all", "most", "some"}
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                     "ninety": 90, "hundred": 100})
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
FUNCTION_POS = {}
for _w in DETERMINERS:
    FUNCTION_POS[_w] = "det"
for _w in PRONOUNS:
    FUNCTION_POS[_w] = "pron"
for _w in list(BE_FORMS) + list(AUX) + list(HAVE_FORMS):
    FUNCTION_POS[_w] = "aux"
for _w in PREPOSITIONS:
    FUNCTION_POS[_w] = "prep"
for _w in CONJ:
    FUNCTION_POS[_w] = "conj"
for _w in WH:
    FUNCTION_POS[_w] = "wh"
for _w in NEG:
    FUNCTION_POS[_w] = "neg"
for _w in NUMBER_WORDS:
    FUNCTION_POS[_w] = "num"
for _w in ("many", "much", "few", "several", "all", "most"):
    FUNCTION_POS[_w] = "quant"
for _w in ("else", "also", "too", "very", "quite", "rather", "just", "still", "always", "often", "soon", "again", "even",
           "only", "already", "almost", "here", "indeed", "perhaps", "maybe", "ever", "however", "really", "sometimes",
           "once", "twice", "away", "together", "somewhat", "anyway", "yet"):
    FUNCTION_POS.setdefault(_w, "r")
for _w in ("himself", "herself", "itself", "themselves", "myself", "yourself", "ourselves", "mine", "yours", "ours", "theirs",
           "everybody", "everyone", "somebody", "someone", "nobody", "anybody", "anyone", "everything", "something",
           "nothing", "anything"):
    FUNCTION_POS.setdefault(_w, "pron")

# --------------------------------------------------------------------------- morphology rules
IRREGULAR_PLURALS = {"mice": "mouse", "men": "man", "women": "woman", "children": "child", "feet": "foot",
                     "teeth": "tooth", "geese": "goose", "people": "person", "oxen": "ox", "lice": "louse",
                     "leaves": "leaf", "wolves": "wolf", "knives": "knife", "lives": "life", "wives": "wife",
                     "loaves": "loaf", "shelves": "shelf", "thieves": "thief", "calves": "calf",
                     "halves": "half", "sheep": "sheep", "fish": "fish", "deer": "deer"}
IRREGULAR_VERBS = {  # form -> (lemma, feature)
    "ran": ("run", "past"), "ate": ("eat", "past"), "eaten": ("eat", "pp"), "saw": ("see", "past"),
    "seen": ("see", "pp"), "went": ("go", "past"), "gone": ("go", "pp"), "slept": ("sleep", "past"),
    "sat": ("sit", "past"), "swam": ("swim", "past"), "flew": ("fly", "past"), "drank": ("drink", "past"),
    "threw": ("throw", "past"), "held": ("hold", "past"), "fed": ("feed", "past"), "read": ("read", "past"),
    "took": ("take", "past"), "gave": ("give", "past"), "came": ("come", "past"), "made": ("make", "past"),
    "said": ("say", "past"), "told": ("tell", "past"), "found": ("find", "past"), "knew": ("know", "past"),
    "thought": ("think", "past"), "heard": ("hear", "past"), "left": ("leave", "past"), "felt": ("feel", "past"),
    "kept": ("keep", "past"), "stood": ("stand", "past"), "fell": ("fall", "past"), "began": ("begin", "past"),
    "wrote": ("write", "past"), "sang": ("sing", "past"), "grew": ("grow", "past"), "bought": ("buy", "past"),
    "caught": ("catch", "past"), "taught": ("teach", "past"), "brought": ("bring", "past"),
    "does": ("do", "3sg"), "has": ("have", "3sg"), "is": ("be", "3sg"), "goes": ("go", "3sg"),
    "had": ("have", "past"), "did": ("do", "past"), "was": ("be", "past"), "were": ("be", "past"),
}
VOWELS = set("aeiou")
_MORE_IRREGULAR = [   # (lemma, past, past participle)
    ("steal", "stole", "stolen"), ("shake", "shook", "shaken"), ("break", "broke", "broken"), ("choose", "chose", "chosen"),
    ("wear", "wore", "worn"), ("ride", "rode", "ridden"), ("rise", "rose", "risen"), ("hide", "hid", "hidden"),
    ("speak", "spoke", "spoken"), ("forget", "forgot", "forgotten"), ("get", "got", "gotten"), ("put", "put", "put"),
    ("let", "let", "let"), ("set", "set", "set"), ("cut", "cut", "cut"), ("hit", "hit", "hit"), ("shut", "shut", "shut"),
    ("send", "sent", "sent"), ("spend", "spent", "spent"), ("build", "built", "built"), ("lend", "lent", "lent"),
    ("bend", "bent", "bent"), ("lose", "lost", "lost"), ("meet", "met", "met"), ("pay", "paid", "paid"),
    ("lay", "laid", "laid"), ("sell", "sold", "sold"), ("mean", "meant", "meant"), ("lead", "led", "led"),
    ("fight", "fought", "fought"), ("seek", "sought", "sought"), ("slide", "slid", "slid"), ("stick", "stuck", "stuck"),
    ("strike", "struck", "struck"), ("swing", "swung", "swung"), ("ring", "rang", "rung"), ("sink", "sank", "sunk"),
    ("drive", "drove", "driven"), ("bite", "bit", "bitten"), ("blow", "blew", "blown"), ("draw", "drew", "drawn"),
    ("throw", "threw", "thrown"), ("fly", "flew", "flown"), ("show", "showed", "shown"), ("hang", "hung", "hung"),
    ("dig", "dug", "dug"), ("win", "won", "won"), ("wake", "woke", "woken"), ("tear", "tore", "torn"),
    ("bear", "bore", "borne"), ("spring", "sprang", "sprung"), ("shoot", "shot", "shot"), ("sweep", "swept", "swept"),
    ("creep", "crept", "crept"), ("weep", "wept", "wept"), ("deal", "dealt", "dealt"), ("dream", "dreamt", "dreamt"),
    ("burn", "burnt", "burnt"), ("learn", "learnt", "learnt"), ("lie", "lay", "lain"), ("do", "did", "done"),
    ("be", "was", "been"), ("have", "had", "had"), ("go", "went", "gone"), ("take", "took", "taken"), ("give", "gave", "given"),
    ("begin", "began", "begun"), ("swim", "swam", "swum"), ("drink", "drank", "drunk"), ("sing", "sang", "sung"),
    ("write", "wrote", "written"), ("grow", "grew", "grown"), ("know", "knew", "known"), ("see", "saw", "seen"),
    ("fall", "fell", "fallen"), ("eat", "ate", "eaten"), ("come", "came", "come"), ("run", "ran", "run"),
    ("become", "became", "become"), ("understand", "understood", "understood"), ("stand", "stood", "stood"),
    ("find", "found", "found"), ("bind", "bound", "bound"), ("hold", "held", "held"), ("feed", "fed", "fed"),
    ("leave", "left", "left"), ("feel", "felt", "felt"), ("keep", "kept", "kept"), ("sleep", "slept", "slept"),
    ("catch", "caught", "caught"), ("teach", "taught", "taught"), ("bring", "brought", "brought"), ("buy", "bought", "bought"),
    ("think", "thought", "thought"), ("say", "said", "said"), ("tell", "told", "told"), ("make", "made", "made"),
    ("hear", "heard", "heard"), ("sit", "sat", "sat"), ("read", "read", "read"),
]
for _l, _p, _pp in _MORE_IRREGULAR:
    if _p != _l:
        IRREGULAR_VERBS.setdefault(_p, (_l, "past"))
    if _pp not in (_p, _l):
        IRREGULAR_VERBS.setdefault(_pp, (_l, "pp"))


def _noun_candidates(w: str) -> List[Tuple[str, str]]:
    """(lemma, number) readings of a possible plural noun form."""
    out: List[Tuple[str, str]] = []
    if w in IRREGULAR_PLURALS:
        out.append((IRREGULAR_PLURALS[w], "pl"))
    if w.endswith("ies") and len(w) > 4:
        out.append((w[:-3] + "y", "pl"))
    if w.endswith("ves") and len(w) > 4:
        out += [(w[:-3] + "f", "pl"), (w[:-3] + "fe", "pl")]
    if w.endswith(("ses", "xes", "zes", "ches", "shes", "oes")) and len(w) > 4:
        out.append((w[:-2], "pl"))
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        out.append((w[:-1], "pl"))
    out.append((w, "sg"))
    return out


def _verb_candidates(w: str) -> List[Tuple[str, str]]:
    """(lemma, form) readings of a possible inflected verb: form in base, 3sg, past, ing."""
    out: List[Tuple[str, str]] = []
    if w in IRREGULAR_VERBS:
        out.append(IRREGULAR_VERBS[w])
    if w.endswith("ies") and len(w) > 4:
        out.append((w[:-3] + "y", "3sg"))
    if w.endswith(("ses", "xes", "zes", "ches", "shes", "oes")) and len(w) > 4:
        out.append((w[:-2], "3sg"))
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        out.append((w[:-1], "3sg"))
    if w.endswith("ied") and len(w) > 4:
        out.append((w[:-3] + "y", "past"))
    if w.endswith("ed") and len(w) > 3:
        stem = w[:-2]
        out += [(stem, "past"), (stem + "e", "past")]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in VOWELS:
            out.append((stem[:-1], "past"))
    if w.endswith("ing") and len(w) > 4:
        stem = w[:-3]
        out += [(stem, "ing"), (stem + "e", "ing")]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in VOWELS:
            out.append((stem[:-1], "ing"))
    out.append((w, "base"))
    return out


def _adj_candidates(w: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if w.endswith("est") and len(w) > 5:
        out += [(w[:-3], "sup"), (w[:-2], "sup")]
        if w[-4] == w[-5]:
            out.append((w[:-4], "sup"))
    if w.endswith("er") and len(w) > 4:
        out += [(w[:-2], "comp"), (w[:-1], "comp")]
        if w[-3] == w[-4]:
            out.append((w[:-3], "comp"))
    if w.endswith("ier") and len(w) > 4:
        out.append((w[:-3] + "y", "comp"))
    out.append((w, "base"))
    return out


def inflect_noun(lemma: str, number: str) -> str:
    if number != "pl":
        return lemma
    rev = {v: k for k, v in IRREGULAR_PLURALS.items() if v != k}
    if lemma in rev:
        return rev[lemma]
    if lemma in ("sheep", "fish", "deer"):
        return lemma
    if lemma.endswith(("s", "x", "z", "ch", "sh")):
        return lemma + "es"
    if lemma.endswith("y") and lemma[-2:-1] not in VOWELS:
        return lemma[:-1] + "ies"
    if lemma.endswith("fe"):
        return lemma[:-2] + "ves"
    if lemma.endswith("f") and lemma not in ("roof", "chief", "belief"):
        return lemma[:-1] + "ves"
    return lemma + "s"


def inflect_verb(lemma: str, form: str) -> str:
    rev = {(v[0], v[1]): k for k, v in IRREGULAR_VERBS.items()}
    if (lemma, form) in rev:
        return rev[(lemma, form)]
    if lemma == "be":
        return {"base": "be", "3sg": "is", "past": "was", "ing": "being"}.get(form, "be")
    if form == "base":
        return lemma
    if form == "3sg":
        if lemma.endswith(("s", "x", "z", "ch", "sh", "o")):
            return lemma + "es"
        if lemma.endswith("y") and lemma[-2:-1] not in VOWELS:
            return lemma[:-1] + "ies"
        return lemma + "s"
    if form == "past":
        if lemma.endswith("e"):
            return lemma + "d"
        if lemma.endswith("y") and lemma[-2:-1] not in VOWELS:
            return lemma[:-1] + "ied"
        if _doubles(lemma):
            return lemma + lemma[-1] + "ed"
        return lemma + "ed"
    if form == "ing":
        if lemma.endswith("ie"):
            return lemma[:-2] + "ying"
        if lemma.endswith("e") and not lemma.endswith(("ee", "ye", "oe")):
            return lemma[:-1] + "ing"
        if _doubles(lemma):
            return lemma + lemma[-1] + "ing"
        return lemma + "ing"
    return lemma


def _doubles(lemma: str) -> bool:
    """Short consonant-vowel-consonant verbs double the last letter (hop -> hopping, sit -> sitting)."""
    return (len(lemma) <= 4 and len(lemma) >= 3 and lemma[-1] not in VOWELS | set("wxy")
            and lemma[-2] in VOWELS and lemma[-3] not in VOWELS)


# --------------------------------------------------------------------------- data classes
@dataclass
class Sense:
    lemma: str
    pos: str                       # n v a r
    kind: str = ""                 # semantic kind: animal person food object place plant body substance time ...
    gloss: str = ""
    hypernyms: List[str] = field(default_factory=list)      # nearest first (one lemma per ancestor)
    ancestors: Set[str] = field(default_factory=set)        # every lemma of every ancestor
    synonyms: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    source: str = "mini"


@dataclass
class Analysis:
    """One reading of a word form."""
    token: str
    lemma: str
    pos: str                       # n v a r det pron aux prep conj wh neg num quant
    feats: Dict[str, str]          # number=sg/pl, form=base/3sg/past/ing, ...
    senses: List[Sense]
    known: bool                    # the lemma is in the dictionary (or is a closed-class word)
    score: float = 0.0             # plausibility of this reading (usage frequency, exact-match bonus)


# --------------------------------------------------------------------------- WordNet backend
WN_KIND = {"noun.animal": "animal", "noun.person": "person", "noun.food": "food", "noun.artifact": "object",
           "noun.plant": "plant", "noun.body": "body", "noun.substance": "substance", "noun.location": "place",
           "noun.time": "time", "noun.quantity": "quantity", "noun.attribute": "attribute", "noun.object": "natural",
           "noun.event": "event", "noun.act": "event", "noun.communication": "communication",
           "noun.cognition": "abstract", "noun.feeling": "abstract", "noun.group": "group", "noun.state": "abstract",
           "noun.phenomenon": "natural", "noun.possession": "abstract", "noun.process": "event",
           "noun.relation": "abstract", "noun.shape": "abstract", "noun.motive": "abstract", "noun.Tops": "abstract"}


class WordNetBackend:
    """Reads the plain-text WordNet 3.0 dict files with random access (nothing is held in memory but the index)."""

    POSMAP = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}

    def __init__(self, path: str) -> None:
        self.path = path
        self.index: Dict[Tuple[str, str], List[int]] = {}
        self.freq: Dict[Tuple[str, str], int] = {}
        self.exc: Dict[Tuple[str, str], str] = {}
        self.lexnames: Dict[int, str] = {}
        self._files: Dict[str, object] = {}
        self._cache: Dict[Tuple[str, int], dict] = {}
        self._load()

    def _load(self) -> None:
        with open(os.path.join(self.path, "lexnames")) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    self.lexnames[int(parts[0])] = parts[1]
        for pos, name in self.POSMAP.items():
            with open(os.path.join(self.path, f"index.{name}"), encoding="utf-8") as f:
                for line in f:
                    if line.startswith("  "):
                        continue
                    t = line.split()
                    p_cnt = int(t[3])
                    n_syn = int(t[2])
                    offs = [int(x) for x in t[6 + p_cnt:6 + p_cnt + n_syn]]
                    self.index[(t[0], pos)] = offs
                    self.freq[(t[0], pos)] = int(t[5 + p_cnt])
            exc = os.path.join(self.path, f"{name}.exc")
            if os.path.exists(exc):
                with open(exc) as f:
                    for line in f:
                        t = line.split()
                        if len(t) >= 2:
                            self.exc.setdefault((t[0], pos), t[1])
            self._files[pos] = open(os.path.join(self.path, f"data.{name}"), "rb")

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:
                pass
        self._files = {}

    def __del__(self) -> None:
        self.close()

    def has(self, lemma: str, pos: str) -> bool:
        return (lemma, pos) in self.index

    def _synset(self, pos: str, off: int) -> dict:
        key = (pos, off)
        s = self._cache.get(key)
        if s is not None:
            return s
        f = self._files[pos]
        f.seek(off)
        line = f.readline().decode("utf-8", errors="ignore")
        head, _, gloss = line.partition("|")
        t = head.split()
        lexfile = int(t[1])
        w_cnt = int(t[3], 16)
        words = [re.sub(r"\(.*?\)", "", t[4 + 2 * i]).lower() for i in range(w_cnt)]
        i = 4 + 2 * w_cnt
        p_cnt = int(t[i])
        ptrs = []
        for k in range(p_cnt):
            sym, poff, ppos, st = t[i + 1 + 4 * k:i + 5 + 4 * k]
            ptrs.append((sym, int(poff), "a" if ppos == "s" else ppos, int(st[:2], 16), int(st[2:], 16)))
        s = dict(pos=pos, off=off, lex=self.lexnames.get(lexfile, ""), words=words, ptrs=ptrs, gloss=gloss.strip())
        self._cache[key] = s
        return s

    def senses(self, lemma: str, pos: str, limit: int = 4) -> List[Sense]:
        out: List[Sense] = []
        for off in self.index.get((lemma, pos), [])[:limit]:
            syn = self._synset(pos, off)
            gl = syn["gloss"]
            definition = gl.split(";")[0].strip()
            examples = re.findall(r'"([^"]+)"', gl)
            chain: List[str] = []
            anc: Set[str] = set()
            cur = syn
            for _ in range(14):
                nxt = [p for p in cur["ptrs"] if p[0] in ("@", "@i")]
                if not nxt:
                    break
                cur = self._synset(nxt[0][2], nxt[0][1])
                chain.append(cur["words"][0])
                anc.update(cur["words"])
            ants: List[str] = []
            for sym, poff, ppos, src, tgt in syn["ptrs"]:
                if sym == "!" and tgt:
                    other = self._synset(ppos, poff)
                    if tgt <= len(other["words"]):
                        ants.append(other["words"][tgt - 1])
            out.append(Sense(lemma=lemma, pos=pos, kind=WN_KIND.get(syn["lex"], ""), gloss=definition,
                             hypernyms=chain, ancestors=anc, synonyms=[w for w in syn["words"] if w != lemma],
                             antonyms=ants, examples=examples, source="wordnet"))
        return out


# --------------------------------------------------------------------------- the lexicon
RESOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


class Lexicon:
    """Dictionary + morphology. ``Lexicon()`` loads the bundled mini dictionary; ``Lexicon(wordnet=path)`` adds
    WordNet (the mini dictionary still wins, so curated entries are never overridden)."""

    def __init__(self, mini: Optional[str] = None, wordnet: Optional[str] = None) -> None:
        self.entries: Dict[Tuple[str, str], Sense] = {}
        self.proper: Set[str] = set()
        self.wn: Optional[WordNetBackend] = None
        mini = mini or os.path.join(RESOURCE_DIR, "mini_lexicon.json")
        if os.path.exists(mini):
            self._load_mini(mini)
        if wordnet:
            self.wn = WordNetBackend(wordnet)
        self._cache: Dict[str, List[Analysis]] = {}

    @staticmethod
    def default() -> "Lexicon":
        """Mini dictionary, plus WordNet if it has been fetched (``python fetch_wordnet.py``)."""
        for cand in (os.environ.get("WORDNET_DIR", ""), os.path.join(os.getcwd(), "data", "wordnet"),
                     os.path.join(os.path.dirname(RESOURCE_DIR), "..", "data", "wordnet")):
            if cand and os.path.exists(os.path.join(cand, "index.noun")):
                return Lexicon(wordnet=cand)
        return Lexicon()

    def _load_mini(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for e in data["entries"]:
            hyp = e.get("hypernyms", [])
            s = Sense(lemma=e["lemma"], pos=e["pos"], kind=e.get("kind", ""), gloss=e.get("gloss", ""),
                      hypernyms=list(hyp), ancestors=set(hyp), synonyms=e.get("synonyms", []),
                      antonyms=e.get("antonyms", []), examples=e.get("examples", []), source="mini")
            self.entries[(e["lemma"], e["pos"])] = s
            if e.get("proper"):
                self.proper.add(e["lemma"])
        # ancestors also include the ancestors of ancestors found in the mini dictionary
        for s in self.entries.values():
            todo = list(s.hypernyms)
            while todo:
                h = todo.pop()
                for pos in ("n", "a", "v"):
                    hs = self.entries.get((h, pos))
                    if hs:
                        for x in hs.hypernyms:
                            if x not in s.ancestors:
                                s.ancestors.add(x)
                                todo.append(x)

    # ----------------------------------------------------------------- lookup
    def senses(self, lemma: str, pos: Optional[str] = None) -> List[Sense]:
        out: List[Sense] = []
        for p in ([pos] if pos else ["n", "v", "a", "r"]):
            s = self.entries.get((lemma, p))
            if s is not None:
                out.append(s)
            elif self.wn is not None and self.wn.has(lemma, p):
                out += self.wn.senses(lemma, p)
        return out

    def knows(self, lemma: str, pos: Optional[str] = None) -> bool:
        return bool(self.senses(lemma, pos)) or lemma in FUNCTION_POS

    def analyze(self, token: str) -> List[Analysis]:
        """All plausible readings of a word form, best first. Unknown words get a guess flagged ``known=False``."""
        t = token.lower()
        hit = self._cache.get(t)
        if hit is not None:
            return hit
        res: List[Analysis] = []
        if t in FUNCTION_POS:
            fp = FUNCTION_POS[t]
            feats: Dict[str, str] = {}
            if t in BE_FORMS:
                feats["number"] = BE_FORMS[t]
            if t in SG_DET:
                feats["number"] = "sg"
            if t in PL_DET:
                feats["number"] = "pl"
            if t in NUMBER_WORDS:
                feats["value"] = str(NUMBER_WORDS[t])
            res.append(Analysis(t, t, fp, feats, [], True))
        seen: Set[Tuple[str, str]] = set()
        for lemma, number in _noun_candidates(t):
            if self.senses(lemma, "n") and (lemma, "n") not in seen:
                seen.add((lemma, "n"))
                res.append(Analysis(t, lemma, "n", {"number": number}, self.senses(lemma, "n"), True))
        for lemma, form in _verb_candidates(t):
            if self.senses(lemma, "v") and (lemma, "v") not in seen:
                seen.add((lemma, "v"))
                res.append(Analysis(t, lemma, "v", {"form": form}, self.senses(lemma, "v"), True))
        for lemma, form in _adj_candidates(t):
            if self.senses(lemma, "a") and (lemma, "a") not in seen:
                seen.add((lemma, "a"))
                res.append(Analysis(t, lemma, "a", {"degree": form}, self.senses(lemma, "a"), True))
        if self.senses(t, "r") and not any(a.pos == "r" for a in res):
            res.append(Analysis(t, t, "r", {}, self.senses(t, "r"), True))
        for a in res:
            a.score = self._score(t, a)
        res.sort(key=lambda a: -a.score)
        if not res:
            res.append(self._guess(t))
        self._cache[t] = res
        return res

    def _freq(self, lemma: str, pos: str) -> int:
        if (lemma, pos) in self.entries:
            return 3
        if self.wn is not None:
            return self.wn.freq.get((lemma, pos), 0)
        return 0

    def _score(self, token: str, a: Analysis) -> float:
        """How plausible is this reading of the token: common words win, an exact spelling match gets a bonus,
        and colour words are adjectives first."""
        if a.pos in FUNCTION_POS.values() and a.lemma in FUNCTION_POS:
            return 10.0
        f = self._freq(a.lemma, a.pos)
        sc = math.log1p(f) + (0.7 if a.lemma == token else 0.0) - (0.8 if f == 0 and a.known else 0.0)
        if a.pos == "a" and any(x.kind == "color" for x in a.senses):
            sc += 0.6
        if a.pos == "r":
            sc -= 0.3
        if a.pos == "v" and token in IRREGULAR_VERBS and IRREGULAR_VERBS[token][0] == a.lemma:
            sc += 3.0                          # "made", "stole", "said" are verb forms before they are anything else
        return sc

    @staticmethod
    def _guess(t: str) -> Analysis:
        """Rules for a word the dictionary has never heard of: read its shape."""
        if t.endswith("ly") and len(t) > 4:
            return Analysis(t, t, "r", {}, [], False)
        if t.endswith(("ing",)) and len(t) > 5:
            return Analysis(t, t[:-3], "v", {"form": "ing"}, [], False)
        if t.endswith("ed") and len(t) > 4:
            return Analysis(t, t[:-2], "v", {"form": "past"}, [], False)
        if t.endswith(("ous", "ful", "ive", "able", "ible", "less", "ish", "al")) and len(t) > 4:
            return Analysis(t, t, "a", {}, [], False)
        if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            return Analysis(t, t[:-1], "n", {"number": "pl"}, [], False)
        return Analysis(t, t, "n", {"number": "sg"}, [], False)

    # ----------------------------------------------------------------- queries on meaning
    def best_sense(self, lemma: str, pos: str = "n") -> Optional[Sense]:
        s = self.senses(lemma, pos)
        return s[0] if s else None

    def kind(self, lemma: str, pos: str = "n") -> str:
        s = self.best_sense(lemma, pos)
        return s.kind if s else ""

    def is_a(self, lemma: str, ancestor: str, pos: str = "n") -> bool:
        if lemma == ancestor:
            return True
        for s in self.senses(lemma, pos):
            if ancestor in s.ancestors or ancestor in s.hypernyms:
                return True
        return False

    def isa_any_pos(self, lemma: str, ancestor: str) -> bool:
        return any(self.is_a(lemma, ancestor, p) for p in ("n", "a", "v"))

    def antonyms(self, lemma: str) -> List[str]:
        out: List[str] = []
        for p in ("a", "v", "n", "r"):
            for s in self.senses(lemma, p):
                out += [a for a in s.antonyms if a not in out]
        return out

    def define(self, lemma: str) -> Optional[str]:
        for p in ("n", "v", "a", "r"):
            s = self.best_sense(lemma, p)
            if s and s.gloss:
                return s.gloss
        return None

    def features(self, token: str) -> List[str]:
        """Symbolic features of a word form for the language model's category layer: its part of speech,
        number/tense, semantic kind and hypernym chain. Words that mean similar things share features."""
        feats: List[str] = []
        for rank, a in enumerate(self.analyze(token)[:2]):
            w = "" if rank == 0 else "2:"
            feats.append(f"{w}pos:{a.pos}")
            for k, v in a.feats.items():
                feats.append(f"{w}{k}:{v}")
            if a.senses:
                s = a.senses[0]
                if s.kind:
                    feats.append(f"{w}kind:{s.kind}")
                for h in s.hypernyms[:6]:
                    feats.append(f"{w}hyp:{h}")
            if not a.known:
                feats.append("unknown")
        return feats

    # ----------------------------------------------------------------- learning at run time
    def learn(self, lemma: str, pos: str, kind: str = "", hypernyms=(), gloss: str = "", antonyms=()) -> Sense:
        """Add (or enrich) an entry while the system is running, e.g. after the user explains a word."""
        hyp = list(hypernyms)
        anc: Set[str] = set(hyp)
        for h in hyp:
            for p in ("n", "a", "v"):
                for hs in self.senses(h, p):
                    anc.update(hs.hypernyms)
                    anc.update(hs.ancestors)
        old = self.entries.get((lemma, pos))
        s = Sense(lemma=lemma, pos=pos, kind=kind or (old.kind if old else ""), gloss=gloss or (old.gloss if old else ""),
                  hypernyms=hyp or (old.hypernyms if old else []), ancestors=anc or (old.ancestors if old else set()),
                  antonyms=list(antonyms) or (old.antonyms if old else []), source="learned")
        self.entries[(lemma, pos)] = s
        self._cache.clear()
        return s

    def learned_entries(self) -> List[dict]:
        return [dict(lemma=s.lemma, pos=s.pos, kind=s.kind, gloss=s.gloss, hypernyms=s.hypernyms, antonyms=s.antonyms)
                for s in self.entries.values() if s.source == "learned"]
