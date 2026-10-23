"""The reader: from a sentence of real English to propositions, by breaking it into clauses first.

The v1 comprehension rules (``comprehension.py``) understand one plain clause. Real text is made of longer sentences:
dialogue, "but"/"and" chains, "when/because/if" clauses, relatives ("a crash which made everybody turn"), infinitives
("began to say unkind things"), causatives ("made him feel queer"), passives, openers like "Then" and "By and by".
This module cuts a tagged sentence into clauses with rules, gives clauses that lack a subject the right one (the previous
subject, the noun a relative clause modifies, or the object of "made him ..."), and hands each clause to the v1 parser.

    "Buster sat down and scratched his head."        -> (buster, sit), (buster, scratch, head)
    "It made him feel very queer."                   -> (it, make, him), (him, feel)
    "He knew that everybody knew that it was true."  -> (he, know), (everybody, know), (it, attr, true)
    "a great crash which made everybody turn"        -> (crash, make, everybody), (everybody, turn)
    "'I guess so,' he muttered."                     -> (he, mutter)        (the quoted words are not asserted as facts)

It is a rule system with known gaps (see README): ellipsis, long-distance dependencies, nested relatives, most idioms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .comprehension import ClauseParser, Discourse, NP, Prop, Tok, _TO_PREP_VERBS
from .roles import time_phrase
from .entities import EntityStore

NEEDS_OBJECT = {"build", "make", "take", "write", "design", "destroy", "rescue", "hit", "carry", "collect", "launch", "attack", "find", "use",
                "give", "buy", "bring", "catch", "eat", "drink", "read", "draw", "paint", "cut", "break", "fix", "open", "close", "throw", "kick",
                "push", "pull", "lift", "save", "kill", "win", "lose", "cook", "bake", "wear", "pay", "send", "hold", "keep", "tell", "show", "teach",
                "order", "command", "serve", "lead", "rule", "cover", "test", "invent", "capture", "defeat", "fire", "paint", "plant", "grow"}
LEAD_WORDS = {"but", "and", "so", "then", "now", "yes", "no", "well", "oh", "why", "still", "yet", "also", "indeed",
              "however", "perhaps", "maybe", "only", "just", "next", "suddenly", "again", "even", "too", "anyway", "sir",
              "alas", "ah", "hello", "hurrah", "ugh", "hm", "please", "surely", "certainly", "presently", "soon",
              "meanwhile", "therefore", "thus", "later", "afterwards"}
LEAD_PHRASES = [("by", "and", "by"), ("pretty", "soon"), ("at", "last"), ("all", "at", "once"), ("of", "course"),
                ("once", "more"), ("all", "of", "a", "sudden"), ("in", "the", "end"), ("after", "a", "while"),
                ("just", "then"), ("right", "away"), ("at", "once"), ("for", "a", "moment"), ("by", "this", "time")]
CONJS = {"and", "but", "or", "nor", "yet", "so"}
SUBORD = {"when", "while", "until", "after", "before", "because", "since", "if", "although", "though", "whenever",
          "unless", "whereas", "wherever", "till", "as"}
RELATIVE = {"who", "whom", "which", "whose"}
SPEECH = {"say", "ask", "reply", "answer", "cry", "shout", "mutter", "whisper", "snap", "exclaim", "call", "growl", "sigh",
          "remark", "observe", "murmur", "scream", "declare", "add", "continue", "tell", "think"}
CAUSATIVE = {"make", "let", "help", "see", "watch", "hear", "feel", "have", "notice"}
PRON_OBJ = {"him", "them", "me", "us", "her", "it", "everybody", "everyone", "somebody", "someone", "nobody"}


_ABBREV = re.compile(r"\b(Mr|Mrs|Ms|Dr|St|Prof|Sr|Jr|Capt|Col|Gen|Rev|Hon|vs|etc)\.")
_BRACKET = re.compile(r"\[[^\]]*\]")
_SHOUT = re.compile(r"(?:\b(?:[IVXLC]+|[A-Z][A-Z'-]+)\b\s+){2,}(?=[A-Z][a-z]|\"|$)")


def clean_text(text: str) -> str:
    """Take out what is not a sentence of the story: [Illustration: ...] notes, ALL-CAPS headings, and the full stops of
    Mr./Mrs./Dr. (which would otherwise end the sentence)."""
    text = _BRACKET.sub(" ", text)
    text = _ABBREV.sub(lambda m: m.group(1), text)
    text = _SHOUT.sub(" ", text + " ")
    text = text.replace("_", "")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Clause:
    kind: str                        # main | sub | rel | comp | inf | caus
    toks: List[Tok]
    inherit: Optional[List[Tok]] = None   # subject tokens to put in front when the clause has none of its own
    skip: bool = False


def _pred(toks: List[Tok]) -> bool:
    """Does this stretch have a finite predicate (a verb that is not just 'to V')?"""
    for i, t in enumerate(toks):
        if t.tag in ("V", "AUX") and not (i and toks[i - 1].text == "to"):
            return True
    return False


def _first_content(toks: List[Tok]) -> Optional[Tok]:
    for t in toks:
        if t.tag not in ("ADV", "NEG", "PUNCT"):
            return t
    return None


def _strip(toks: List[Tok]) -> List[Tok]:
    out = list(toks)
    while out and out[-1].tag == "PUNCT":
        out.pop()
    while out and out[0].tag == "PUNCT":
        out.pop(0)
    return out


class Reader:
    def __init__(self, lex, discourse: Discourse, ents: Optional[EntityStore] = None,
                 aliases: Optional[Dict[str, str]] = None, def_alias: Optional[Dict[str, str]] = None) -> None:
        self.lex = lex
        self.def_alias = def_alias if def_alias is not None else {}
        self.parser = ClauseParser(lex, discourse, ents, def_alias=self.def_alias)
        self.aliases = aliases if aliases is not None else {}
        self.last_subj: Optional[List[Tok]] = None
        self.clauses_seen = 0

    # ------------------------------------------------------------------ sentence level
    def read(self, toks: List[Tok]) -> List[Prop]:
        narration = self._narration(toks)
        if not narration:
            return []
        props: List[Prop] = []
        main: Optional[Prop] = None
        said = self._said_clause(narration)
        if said is not None:
            props, narration = said
            if not narration:
                return self._finish(props)
        lead, narration = self._leading_adjunct(narration)
        narration = self._drop_openers(narration)
        for c in self.clauses(narration):
            if c.skip or not c.toks:
                continue
            got = self._parse_clause(c)
            if c.kind == "inf" and got and main is not None:
                p0 = got[0]
                verb = p0.rel.split("_")[0]
                from .entities import lemma_of
                o = lemma_of(p0.obj)
                obj = o and (o if p0.obj_proper or o in self.lex.proper else "the " + o)
                main.adj.setdefault("inf", f"{verb} {obj}".strip())        # "refused to use the cartridges"
            elif got and c.kind != "rel":
                main = got[0]
            if lead and got and not props:
                for role, text in lead.items():
                    got[0].adj.setdefault(role, text)              # "After the mutiny the Crown took control ..."
            props += got
        # "They built and flew the first airplane": the first verb shares the object of the coordinated one
        for a, b in zip(props, props[1:]):
            if (a.subj == b.subj and not a.obj and not a.adj and b.obj and a.rel == a.rel.split("_")[0] and b.rel == b.rel.split("_")[0]
                    and a.rel != b.rel and a.rel in NEEDS_OBJECT):
                a.obj, a.obj_num, a.obj_proper = b.obj, b.obj_num, b.obj_proper
        return self._finish(props)

    def _narration(self, toks: List[Tok]) -> List[Tok]:
        """The sentence without the words inside quotation marks (what a character says is not a fact about the world)."""
        out, inside, had_quote = [], False, False
        for t in toks:
            if t.text == '"':
                inside, had_quote = not inside, True
                continue
            if not inside:
                out.append(t)
        out = _strip(out)
        if had_quote and out and out[-1].text in (".", "?", "!"):
            out = out[:-1]
        return _strip(out)

    def _said_clause(self, toks: List[Tok]):
        """'said Alice' / 'snapped Little Joe, as ...' after a quotation: (Alice, say), then the rest is read normally."""
        if toks and toks[0].tag == "V" and toks[0].lemma in SPEECH:
            np, k = self.parser.read_np(toks, 1)
            if np is not None:
                self.parser.discourse.mention(np, subject=True)
                prop = Prop(self.parser.resolve(np).key, toks[0].lemma, "", "", False, "past", np.number, "sg", np.proper)
                rest = _strip(toks[k:])
                return [prop], rest
        return None

    def _leading_adjunct(self, toks: List[Tok]):
        """'After the mutiny the British Crown took control': the opening time phrase belongs to the main fact."""
        if toks and toks[0].tag == "PREP" and toks[0].text in ("after", "before", "during", "until", "since", "in", "on", "at", "by"):
            prep = toks[0].text
            tp = time_phrase(toks, 1, prep)
            if tp is not None:
                k = tp[2]
                if k < len(toks) and toks[k].text == ",":
                    k += 1
                if _pred(toks[k:]):
                    return {tp[0]: f"{prep} {tp[1]}"}, toks[k:]
            else:
                self.parser._break_name = True
                try:
                    np, k = self.parser.read_np(toks, 1)
                finally:
                    self.parser._break_name = False
                if np is not None and prep in ("after", "before", "during", "until", "since"):
                    j = k + 1 if k < len(toks) and toks[k].text == "," else k
                    if j < len(toks) and _pred(toks[j:]):
                        return {"when": f"{prep} {np.text}"}, toks[j:]
        return {}, toks

    def _drop_openers(self, toks: List[Tok]) -> List[Tok]:
        toks = list(toks)
        changed = True
        while changed and toks:
            changed = False
            low = [t.text for t in toks]
            for ph in LEAD_PHRASES:
                if tuple(low[:len(ph)]) == ph and len(toks) > len(ph):
                    toks = toks[len(ph):]
                    changed = True
                    break
            if toks and toks[0].text in LEAD_WORDS and len(toks) > 1:
                toks = toks[1:]
                changed = True
            while toks and toks[0].tag == "PUNCT":
                toks = toks[1:]
                changed = True
        # "You see, Buster walked ..." : a short parenthetical clause, then a comma
        for i, t in enumerate(toks[:5]):
            if t.text == "," and 1 < i <= 3 and toks[0].tag == "PRON" and toks[0].text in ("i", "you", "we") \
                    and toks[i - 1].lemma in ("see", "know", "guess", "think", "suppose", "mean", "say", "believe"):
                return toks[i + 1:]
        # "That morning, he ..." : a short opener with no verb, then a comma
        for i, t in enumerate(toks[:6]):
            if t.text == "," and i and not _pred(toks[:i]) and _pred(toks[i + 1:]) and toks[0].tag in ("ADV", "PREP", "CONJ", "ADJ", "N", "NUM") \
                    and not (toks[0].tag in ("N", "DET", "PN") and not self.lex.senses(toks[0].lemma, "r")):
                return toks[i + 1:]
        return toks

    # ------------------------------------------------------------------ clause splitting
    def clauses(self, toks: List[Tok], kind: str = "main", inherit: Optional[List[Tok]] = None) -> List[Clause]:
        toks = _strip(toks)
        if not toks:
            return []
        # 1. ; and :
        for i, t in enumerate(toks):
            if t.text in (";", ":") and i:
                return self.clauses(toks[:i], kind, inherit) + self.clauses(toks[i + 1:], "main")
        # 2. coordination: "X did a and did b", "X did a, but Y did b"
        for i, t in enumerate(toks):
            if i and t.text in CONJS and t.tag in ("CONJ", "ADV", "PREP") and (t.text != "so" or toks[i - 1].text == ","):
                left, right = _strip(toks[:i]), toks[i + 1:]
                stop = next((j for j, x in enumerate(right) if x.text == ","), len(right))
                if _pred(left) and _pred(right[:stop]) and self._starts_clause(right):
                    return self.clauses(left, kind, inherit) + self.clauses(right, "main")
        # 3. subordinate / relative / complement clauses
        for i, t in enumerate(toks):
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if t.text == "as" and nxt is not None and nxt.text in ("if", "though") and i + 2 < len(toks):
                return self.clauses(toks[:i], kind, inherit) + self.clauses(toks[i + 2:], "sub")
            if t.text == "as" and (i == 0 or not _pred(toks[i + 1:]) or toks[i + 1].tag not in ("PRON", "PN", "DET", "N")):
                continue                                           # "as big as", "as a bear": not a clause
            if t.text in SUBORD and nxt is not None and _pred(toks[i + 1:]):
                if i == 0:
                    e = next((j for j in range(1, len(toks)) if toks[j].text == "," and _pred(toks[j + 1:])), None)
                    if e is not None:
                        return self.clauses(toks[1:e], "sub") + self.clauses(toks[e + 1:], "main")
                    return self.clauses(toks[1:], "sub")
                return self.clauses(toks[:i], kind, inherit) + self.clauses(toks[i + 1:], "sub")
            if i and t.text in RELATIVE | {"that"} and nxt is not None:
                prev = toks[i - 1]
                head = self._np_before(toks, i)
                if t.text == "that":
                    if nxt.tag in ("V", "AUX") and head and prev.tag in ("N", "PN", "PRON"):
                        return self.clauses(toks[:i], kind, inherit) + self.clauses(toks[i + 1:], "rel", head)
                    if _pred(toks[i + 1:]) and (prev.tag in ("V", "ADJ", "ADV", "N", "PN") or prev.text in ("so",)):
                        return self.clauses(toks[:i], kind, inherit) + self.clauses(toks[i + 1:], "comp")
                    continue
                if prev.tag == "V":                              # "knew who had spoiled it": a question word, not a relative
                    return self.clauses(toks[:i], kind, inherit)
                if head and _pred(toks[i + 1:]):
                    if t.text == "whose":
                        return self.clauses(toks[:i], kind, inherit)
                    return self.clauses(toks[:i], kind, inherit) + self.clauses(toks[i + 1:], "rel", head)
        # 4. causative: "made him feel queer"
        for i, t in enumerate(toks[:-2]):
            if t.tag == "V" and t.lemma in CAUSATIVE and not (i + 1 < len(toks) and toks[i + 1].text == "to"):
                np, k = self.parser.read_np(toks, i + 1)
                if np is not None and k < len(toks) and toks[k].tag in ("V", "AUX") and toks[k].text != "to" \
                        and (toks[k].feats.get("form") in ("base", None, "") or toks[k].lemma == "be"):
                    return [Clause(kind, toks[:k], inherit), Clause("caus", toks[i + 1:])] \
                        if k - (i + 1) > 0 else [Clause(kind, toks, inherit)]
        # 5. infinitive: "tried to pull it off", "a word to say"
        for i, t in enumerate(toks[:-1]):
            if t.text == "to" and toks[i + 1].tag in ("V", "AUX") and i:
                before = self.lex.analyze(toks[i - 1].text)[0] if toks[i - 1].text.isalpha() else None
                if before is not None and before.pos == "v" and before.lemma in _TO_PREP_VERBS:
                    continue
                head = toks[:i]
                while head and head[-1].text in {"not", "so", "as", "order", "in", "," , "way"}:
                    head = head[:-1]
                tail = toks[i + 1:]
                subj_toks = None   # filled in from the head's own subject when the clause is parsed
                return self.clauses(head, kind, inherit) + self.clauses(tail, "inf")
        return [Clause(kind, toks, inherit)]

    def _starts_clause(self, toks: List[Tok]) -> bool:
        f = _first_content(toks)
        return f is not None and f.tag in ("PRON", "DET", "N", "PN", "V", "AUX", "NUM", "QUANT", "ADJ", "WH") or \
            (f is not None and f.text in SUBORD)

    def _np_before(self, toks: List[Tok], end: int) -> Optional[List[Tok]]:
        """The noun phrase that ends just before toks[end], skipping back over a prepositional phrase ('something in the field')."""
        j = end
        while j > 0 and toks[j - 1].tag == "PUNCT":
            j -= 1
        k = j
        while k > 0 and toks[k - 1].tag in ("N", "PN", "ADJ", "DET", "QUANT", "NUM", "POSS", "PRON"):
            k -= 1
        if k == j:
            return None
        if k > 0 and toks[k - 1].tag == "PREP" and k - 1 > 0:
            inner = self._np_before(toks, k - 1)
            if inner:
                return inner
        return toks[k:j]

    # ------------------------------------------------------------------ one clause -> propositions
    def _parse_clause(self, c: Clause) -> List[Prop]:
        toks = list(c.toks)
        first = _first_content(toks)
        needs = first is not None and first.tag in ("V", "AUX") and c.kind != "caus"
        if c.kind == "rel" and c.inherit:
            toks = list(c.inherit) + toks
        elif needs or c.kind == "inf":
            if self.last_subj:
                toks = list(self.last_subj) + toks
            else:
                return []
        else:
            np, k = self.parser.read_np(toks, 0)
            if np is not None:
                self.last_subj = toks[:k]
        sub_subj = self.last_subj
        props = self.parser.parse_declarative(toks)
        if c.kind in ("rel", "caus") or needs or c.kind == "inf":
            self.last_subj = sub_subj
        self.clauses_seen += 1
        return [p for p in props if p.subj and p.subj not in ("there", "else", "what", "one")]

    # ------------------------------------------------------------------ names
    def _finish(self, props: List[Prop]) -> List[Prop]:
        """"Buster Bear" and later "Buster" are the same: register every word of a longer name and map short forms to it."""
        for p in props:
            if p.subj_proper and " " in p.subj:
                last = p.subj.split(" of ")[0].split()[-1]               # "great fire of london" -> fire
                if self.lex.senses(last, "n"):                       # "the mutiny" later means the Sepoy Mutiny
                    if self.def_alias.setdefault(last, p.subj) != p.subj:
                        self.def_alias[last] = ""                    # two names end in the same noun: ambiguous, do not guess
                words = p.subj.split(" of ")[0].split()
                for n in range(1, len(words)):                       # "little joe" and "joe" for "little joe otter"
                    for a in range(0, len(words) - n + 1):
                        span = " ".join(words[a:a + n])
                        if n > 1 or (a == 0 and not self.lex.senses(span, "a")) or not self.lex.senses(span):
                            self.aliases.setdefault(span, p.subj)
        for p in props:
            if p.rel == "called" and p.obj_proper:                    # "The airplane was called the Wright Flyer": one thing, two names
                self.def_alias[p.subj] = p.obj
                from .comprehension import NP
                self.parser.discourse.mention(NP(head=p.obj, proper=True, det="the" if p.obj_the else "", number=p.subj_num), subject=True)
        out = []
        for p in props:
            subj, obj = p.subj, p.obj
            if p.subj_proper and subj in self.aliases:
                subj = self.aliases[subj]
            if obj in self.aliases and not self.lex.senses(obj, "n") and obj not in self.lex.proper:
                obj = self.aliases[obj]
            out.append(Prop(subj, p.rel, obj, p.qty, p.neg, p.tense, p.subj_num, p.obj_num, p.subj_proper, adj=p.adj,
                            obj_proper=p.obj_proper, subj_the=p.subj_the, obj_the=p.obj_the))
        return out
