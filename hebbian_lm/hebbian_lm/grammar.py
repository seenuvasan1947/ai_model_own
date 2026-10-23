"""Grammar as explicit knowledge: word forms, word classes, voice and tense, and question forms.

The language layer learns grammar from examples and the rule engine (``lexicon.py``) holds the morphology. This module makes the
rules *usable on demand*, in two ways:

* **Asked about**: "what is the plural of mouse ?", "what is the past tense of go ?", "what is the past participle of eat ?",
  "what is the opposite of ...", "what kind of word is run ?", "is quickly a noun ?".
* **Used to say things another way**: ``variants`` turns one stored fact into its active, passive, negated, future and question forms,
  all built by rule (agreement, tense, a/an, the participle), so the same fact can be told in different words.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .lexicon import inflect_noun, inflect_verb

_FORMS = {"past tense": "past", "past": "past", "past participle": "pp", "participle": "pp", "present participle": "ing",
          "ing form": "ing", "third person": "3sg", "third person singular": "3sg", "3rd person": "3sg", "present tense": "3sg"}


def parse_grammar_question(w: List[str]) -> Optional[Tuple[str, str, str]]:
    """Words of a question -> (kind, form, word); kinds: form | plural | singular | class | isclass."""
    w = [x for x in w if x not in ("the", "a", "an")]
    n = len(w)
    if n >= 4 and w[0] == "what" and w[1] == "is" and w[-2] == "of":
        what = " ".join(w[2:-2])
        if what == "plural":
            return "plural", "pl", w[-1]
        if what == "singular":
            return "singular", "sg", w[-1]
        if what in _FORMS:
            return "form", _FORMS[what], w[-1]
        if what in ("comparative", "superlative"):
            return "degree", what, w[-1]
    if n >= 5 and w[0] == "what" and w[1] in ("kind", "part", "type", "sort") and w[2] == "of" and w[3] == "word" and w[4] == "is":
        return "class", "", w[-1]
    if n >= 3 and w[0] == "what" and w[1] == "is" and w[-1] == "word" and False:
        return None
    if n >= 3 and w[0] in ("is", "are") and w[-1] in ("noun", "verb", "adjective", "adverb"):
        return "isclass", w[-1], w[1]
    if n >= 5 and w[0] == "how" and w[1] == "do" and w[2] == "you" and w[3] == "say" and "plural" in w:
        return "plural", "pl", w[-1]
    return None


def degree(adj: str, which: str) -> str:
    suffix = "er" if which == "comparative" else "est"
    if adj.endswith("e"):
        return adj + suffix[1:]
    if adj.endswith("y") and adj[-2:-1] not in "aeiou":
        return adj[:-1] + "i" + suffix
    if len(adj) == 3 and adj[-1] not in "aeiou" and adj[-2] in "aeiou" and adj[-3] not in "aeiou":
        return adj + adj[-1] + suffix
    if len(adj) > 6:
        return ("more " if which == "comparative" else "most ") + adj
    return adj + suffix


def word_classes(lex, word: str) -> List[str]:
    names = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb", "det": "determiner", "pron": "pronoun", "aux": "auxiliary verb",
             "prep": "preposition", "conj": "conjunction", "wh": "question word", "neg": "negation word", "num": "number",
             "quant": "quantifier"}
    out: List[str] = []
    for a in lex.analyze(word):
        if a.known and names.get(a.pos) and names[a.pos] not in out:
            out.append(names[a.pos])
    return out


# ---------------------------------------------------------------------------------------------------- voice and tense
STATIVE = {"last", "become", "cost", "weigh", "have", "be", "seem", "lack", "fit", "resemble", "suit", "remain", "stay", "appear", "contain"}


def participle(verb: str) -> str:
    return inflect_verb(verb, "pp")


def variants(sub: str, verb: str, obj: str, past: bool, plural: bool, neg: bool = False,
             sub_np: Optional[str] = None, obj_np: Optional[str] = None) -> Dict[str, str]:
    """One fact, told in different grammatical forms. ``sub``/``obj`` are the noun phrases as they should be printed."""
    be_past, be_pres = ("were" if plural else "was"), ("are" if plural else "is")
    out: Dict[str, str] = {}
    vp = inflect_verb(verb, "past" if past else ("base" if plural else "3sg"))
    if neg:
        vp = ("did" if past else ("do" if plural else "does")) + " not " + verb
    out["active"] = f"{sub} {vp}" + (f" {obj}" if obj else "")
    if obj and not neg and verb not in STATIVE:
        obj_plural = obj.endswith("s") and obj.split()[-1] != "this"
        out["passive"] = f"{obj} {'were' if obj_plural else 'was'}{'' if past else ''} {participle(verb)} by {sub}" if past else \
            f"{obj} {'are' if obj_plural else 'is'} {participle(verb)} by {sub}"
    out["future"] = f"{sub} will {'not ' if neg else ''}{verb}" + (f" {obj}" if obj else "")
    out["question"] = f"{'did' if past else ('do' if plural else 'does')} {sub} {verb}" + (f" {obj}" if obj else "") + " ?"
    out["negative"] = f"{sub} {'did' if past else ('do' if plural else 'does')} not {verb}" + (f" {obj}" if obj else "")
    return out
