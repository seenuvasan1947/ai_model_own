"""The Hebbian Semantic Workspace: reading with understanding, exact storage, grammatical answers.

Architecture (each part does what it is good at):

    text ──► Lexicon + Morphology rules ──► tagger/chunker (comprehension) ──► propositions ──► Semantic memory
                  │  unknown word?                                                                 │
                  │  look it up, learn it                                                          ▼
                  └──► Hebbian language model (grammar, categories, fluency) ◄── realisation ◄── retrieval + inference

* The **dictionary** supplies meaning for words the system has never met (definition, kind, "is a" chain,
  antonyms) and the **rule engine** supplies inflection and part of speech, so a new word is usable at once.
* **Comprehension** converts each sentence into propositions while reading, resolving pronouns.
* **Semantic memory** stores them exactly (Hebbian strengthening, competition between rival values, recency,
  inheritance through "is a") so retrieval is reliable instead of statistical.
* The **Hebbian language model** keeps learning grammar and word categories from everything read, scores how
  fluent a sentence is, and (with the rule engine) checks and repairs grammar.
* Answers are **realised** with the morphology rules (agreement, plurals, a/an), so they are grammatical by construction.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .comprehension import (END, NP, ClauseParser, Discourse, Prop, Query, Tok, number_word, tag_sentence, tokenize_cased)
from .entities import is_instance, lemma_of
from .notes import CUES, Note, NoteKeeper
from .numbers import WORD_VALUE, CannotCount, NumberSense, parse_number, to_words
from .reader import Reader, clean_text
from .lexicon import PL_DET, PRONOUNS, SG_DET, Lexicon, inflect_noun, inflect_verb
from .semantic import NON_ACTIONS, SPATIAL, Answer, SemanticMemory

VOWEL_SOUND = set("aeiou")
NO_ARTICLE = {"something", "nothing", "everything", "anything", "everybody", "everyone", "somebody", "someone", "nobody",
              "himself", "herself", "itself", "themselves", "myself", "yourself", "here", "there", "home"}


@dataclass
class Reply:
    text: str
    kind: str = "answer"             # answer | idk | unknown_word | not_understood | learned | clarify
    confidence: float = 0.0
    inferred: bool = False
    note: str = ""
    fluency: Optional[float] = None  # mean log-probability per token under the language model


@dataclass
class ReadReport:
    props: List[Prop] = field(default_factory=list)
    new_words: List[str] = field(default_factory=list)       # looked up in the dictionary and learned
    unknown_words: List[str] = field(default_factory=list)   # the dictionary could not explain these
    defined: List[str] = field(default_factory=list)         # explained by the text itself ("a flib is an animal")
    questions: int = 0
    number_notes: List[str] = field(default_factory=list)    # what was learned about counting and arithmetic


@dataclass
class Issue:
    kind: str                        # agreement | number | article | unknown
    index: int
    message: str
    suggestion: str = ""


def a_an(word: str) -> str:
    return "an" if word[:1] in VOWEL_SOUND and not word.startswith(("uni", "use", "one")) else "a"


class Workspace:
    def __init__(self, model=None, lexicon: Optional[Lexicon] = None, memory: Optional[SemanticMemory] = None,
                 instances: bool = True, real_text: bool = True, take_notes: bool = True, style: str = "plain",
                 seed: int = 0) -> None:
        self.lex = lexicon or Lexicon.default()
        self.model = model
        if model is not None and getattr(model, "lexicon", None) is None:
            model.attach_lexicon(self.lex)
        self.mem = memory or SemanticMemory(self.lex)
        self.discourse = Discourse()
        self.seen: Set[str] = set()                 # lemmas already processed (so each is looked up once)
        self.entities: Set[str] = set()             # every subject/object lemma stored so far
        self.pending: Optional[str] = None          # a word the user has been asked to explain
        self.numbers = NumberSense()                # counting and arithmetic, learned (see numbers.py)
        self.notes = NoteKeeper(self.lex)           # sentences kept word for word, indexed by content words
        self.take_notes = take_notes                # write down every sentence that is read
        self.style = style                          # "plain": one fixed wording | "varied": own-words answers
        self.rng = random.Random(seed)
        self.turn_time = 0.0                        # time units that pass with every chat turn (0 = time stands still)
        self.real_text = real_text                  # v2 reader: clauses, relatives, infinitives, dialogue, names
        self.aliases: Dict[str, str] = {}           # "buster" -> "buster bear"
        self.instances = instances                  # separate entities for separate mentions ("a red ball", "tom's ball")
        self.clarifying: Optional[tuple] = None     # (question, lemma, candidate ids) after a "which ball ?" question

    # ================================================================== reading
    def new_document(self) -> None:
        self.discourse = Discourse()

    def read(self, text: str, told: bool = True, learn_lm: bool = True, take_notes: Optional[bool] = None) -> ReadReport:
        """Read statements: look up unknown words, understand each sentence, store what it says."""
        rep = ReadReport()
        text = clean_text(text)
        noting = take_notes if take_notes is not None else self.take_notes
        items = []                                   # (tokens, propositions, number of propositions that were new)
        for sent in self._sentences(text):
            end = sent[-1].text if sent and sent[-1].text in END else "."
            if end == "?":
                rep.questions += 1
                continue
            tag_sentence(sent, self.lex)
            handled = self._number_statement(sent, rep)
            if handled:
                continue
            self._mark_names(sent)
            self._meet_words(sent, rep)
            if self.real_text:
                props = Reader(self.lex, self.discourse, self._ents(), self.aliases).read(sent)
            else:
                props = ClauseParser(self.lex, self.discourse, self._ents()).parse_declarative(sent)
            fresh = 0
            for p in props:
                self._definitional(p, rep)
                fresh += self._is_new(p)
                self.mem.add(p, told=told)
                self.entities.update(lemma_of(x) for x in (p.subj, p.obj) if x)
                rep.props.append(p)
            rep.unknown_words = [u for u in rep.unknown_words if u not in rep.defined]
            items.append((sent, props, fresh))
            if learn_lm and self.model is not None:
                s = " ".join(t.text for t in sent if t.text not in END and t.text != '"' and t.tag != "POSS") + " ."
                if told:
                    self.model.tell(s)
                else:
                    self.model.read(s, learn=True)
        if noting and items:
            self._take_notes(items, told)
        return rep

    # ---- notes: only what is worth keeping
    def _is_new(self, p: Prop) -> bool:
        grp = self.mem.slots.get((p.subj, self.mem._slot_rel(p.rel)), {})
        f = grp.get((p.obj, p.qty, p.neg))
        return f is None or f.strength < 0.3

    def _prop_text(self, p: Prop) -> str:
        pl = p.subj_num == "pl"
        sub = self._np(p.subj, "pl" if pl else "sg", p.subj_proper)
        past = p.tense == "past"
        be = ("were" if pl else "was") if past else ("are" if pl else "is")
        obj = self._obj_np(p.obj, p.obj_num) if p.obj else ""
        if p.rel == "attr":
            return f"{sub} {be}{' not' if p.neg else ''} {p.obj}"
        if p.rel == "isa":
            return f"{sub} {be}{' not' if p.neg else ''} {a_an(p.obj)} {p.obj}"
        if p.rel == "can":
            return f"{sub} {'cannot' if p.neg else 'can'} {p.obj}"
        if p.rel in ("count", "equals", "antonym"):
            return ""
        verb, _, prep = p.rel.partition("_")
        if p.rel == "have":
            v = "had" if past else ("have" if pl else "has")
        else:
            v = inflect_verb(verb, "past" if past else ("base" if pl else "3sg"))
        if p.neg:
            v = f"{'did' if past else ('do' if pl else 'does')} not {verb}"
        tail = f"{prep} {obj}" if prep and prep != "@" else obj
        return f"{sub} {v} {tail}".strip()

    def _gist(self, props: List[Prop], limit: int = 3) -> str:
        """The few strongest propositions of a long sentence, as short sentences the reader can read again."""
        seen, out = set(), []
        usable = [p for p in props if p.subj not in ("i", "you", "we", "me", "us") and p.rel != "have"]
        for p in sorted(usable or props, key=lambda p: (p.subj in PRONOUNS, not p.obj, p.rel in ("attr",), p.rel.count("_"))):
            t = self._prop_text(p)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return " . ".join(out[:limit]) + " ." if out else ""

    def _take_notes(self, items, told: bool) -> None:
        """Decide what is worth writing down. A short sentence is kept as it is; a long one as a gist of its main points plus
        its key terms. Sentences that add nothing new, or that are filler, are not noted at all."""
        nk = self.notes
        entries = []
        doc: Dict[str, int] = {}
        for sent, props, fresh in items:
            toks = [t for t in sent if t.text not in END and t.text != '"' and t.tag != "POSS"]
            words = nk.content(" ".join(t.text for t in toks))
            for w in set(words):
                doc[w] = doc.get(w, 0) + 1
            entries.append((toks, props, fresh, words))
        scored, cand, always = [], {}, set()
        for i, (toks, props, fresh, words) in enumerate(entries):
            if not words:
                continue
            surface = " ".join(t.text for t in toks) + " ."
            old = nk.find(surface)
            if old is not None:
                nk._strengthen(old)                                  # said again: strengthened, not written twice
                continue
            cue = any(t.text in CUES for t in toks) and (told or len(toks) <= 12)
            score, keys = nk.importance(words, doc, len(props), cue, sum(t.tag == "PN" for t in toks),
                                        any(t.tag == "NUM" or t.text.isdigit() for t in toks),
                                        any(p.rel == "isa" for p in props), any(p.neg for p in props))
            if props:
                score += 0.6 if fresh else -0.5                     # a new fact matters, a known one does not
            cand[i] = (surface, props, keys, score, len(toks))
            scored.append((score, i))
            if cue:
                always.add(i)
        for i in nk.select(scored, always):
            surface, props, keys, score, n_tok = cand[i]
            if n_tok <= 12:
                text = surface
            else:                                    # long: a gist of the main points, or just the key terms
                text = self._gist(props) or ("key terms : " + " , ".join(keys) + " .")
            nk.add(text, "told" if told else "read", importance=score, keys=keys)
        for toks, props, fresh, words in entries:
            nk.read_words(words)

    # ---- numbers: counting runs, "N comes after M", "a plus b is c"
    _OPS = {"plus": "+", "minus": "-", "times": "*", "divided": "/"}

    def _numerals(self, words: List[str]) -> Optional[List[int]]:
        """A run of number words / digits as a list of numbers ('twenty one twenty two' -> [21, 22]); None if anything else is in it."""
        out, i = [], 0
        words = [w for w in words if w != ","]
        while i < len(words):
            best = None
            for j in range(len(words), i, -1):
                seg = words[i:j]
                v = parse_number(seg)
                if v is not None and to_words(v).replace(" and ", " ").split() == [x for x in seg if x != "and"] or \
                        (v is not None and len(seg) == 1 and seg[0].isdigit()):
                    best = (v, j)
                    break
            if best is None:
                return None
            out.append(best[0])
            i = best[1]
        return out

    def _number_statement(self, sent: List[Tok], rep: ReadReport) -> bool:
        words = [t.text for t in sent if t.text not in END and t.text != '"']
        if not words or not any(w.isdigit() or w in WORD_VALUE for w in words):
            return False
        # 1. a counting run: "one two three four five"
        seq = self._numerals(words)
        if seq is not None and len(seq) >= 3:
            used = self.numbers.hear(seq)
            if used:
                rep.number_notes.append(f"counted {to_words(seq[0])} to {to_words(seq[-1])} ( {used} steps )")
            return True
        # 2. "four comes after three" / "after three comes four" / "the number after three is four"
        for pat in (("after", "comes"), ("comes", "after"), ("after", "is"), ("after", "come")):
            if all(p in words for p in pat):
                nums = [self._numerals(seg) for seg in self._split_on(words, set(pat) | {"the", "number", "that", "which"})]
                nums = [n[0] for n in nums if n]
                if len(nums) == 2:
                    after, comes = (nums[0], nums[1]) if words.index(pat[0]) < words.index(pat[1]) else (nums[1], nums[0])
                    if pat[0] == "after" and pat[1] in ("comes", "come", "is") and words.index("after") < words.index(pat[1]):
                        a, b = nums[0], nums[1]
                    else:
                        b, a = nums[0], nums[1]
                    if self.numbers.hear([a, b]):
                        rep.number_notes.append(f"{to_words(b)} comes after {to_words(a)}")
                    return True
        # 3. "two plus three is five" / "2 + 3 = 5"
        for i, w in enumerate(words):
            if w in self._OPS and 0 < i < len(words) - 2:
                right_part = words[i + 1:]
                if right_part and right_part[0] == "by":
                    right_part = right_part[1:]
                eq = next((k for k, x in enumerate(right_part) if x in ("is", "equals", "are", "makes", "make", "gives")), None)
                if eq is None:
                    return False
                a, b, c = parse_number(words[:i]), parse_number(right_part[:eq]), parse_number(right_part[eq + 1:])
                if None in (a, b, c):
                    return False
                if not self.numbers.can_count():
                    rep.number_notes.append("i have not learned to count yet , so i cannot check it ; teach me : one two three four ...")
                    return True
                try:
                    ok, r = self.numbers.check_taught(self._OPS[w], a, b, c)
                except CannotCount as e:
                    rep.number_notes.append(str(e))
                    return True
                word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[self._OPS[w]]
                if ok:
                    rep.number_notes.append(f"practised {to_words(a)} {word} {to_words(b)} is {to_words(c)} ( {r.method} )")
                else:
                    rep.number_notes.append(f"that does not match my counting : {to_words(a)} {word} {to_words(b)} is {to_words(r.value)}")
                return True
        return False

    @staticmethod
    def _split_on(words: List[str], stop: Set[str]) -> List[List[str]]:
        out, cur = [], []
        for w in words:
            if w in stop:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(w)
        if cur:
            out.append(cur)
        return out

    def school(self, count_to: int = 100, repeats: int = 3, tables: int = 10) -> Dict[str, int]:
        """Send the number sense to school: count to ``count_to`` again and again, then practise the small tables until they are
        remembered rather than counted. Returns how many traces exist afterwards."""
        self.numbers.teach_counting(count_to, repeats)
        for _ in range(repeats):
            for a in range(0, tables + 1):
                for b in range(0, tables + 1):
                    self.numbers.compute("+", a, b)
                    if a >= b:
                        self.numbers.compute("-", a, b)
            for a in range(1, tables + 1):
                for b in range(1, tables + 1):
                    self.numbers.compute("*", a, b)
        return dict(traces=len(self.numbers.traces), counted_to=self.numbers.max_taught)

    def _number_reply(self, q: Query) -> Reply:
        a, op, b = q.expr
        ns = self.numbers
        word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}
        if q.kind == "arithmetic":
            if not ns.can_count():
                from .comprehension import evaluate_arithmetic
                v = evaluate_arithmetic((a, op, b)) if op != "/" else (a // b if b else None)
                if v is None:
                    return Reply("i do not know .", kind="idk")
                return Reply(f"{to_words(a)} {word[op]} {to_words(b)} is {to_words(v)} .", confidence=1.0,
                             note="by rule : i have not learned to count yet ( teach me : one two three four ... )")
            try:
                r = ns.compute(op, a, b)
            except CannotCount as e:
                return Reply(self._cannot(e), kind="idk", note="cannot count that")
            tail = f" remainder {to_words(r.remainder)}" if op == "/" and r.remainder else ""
            how = "recalled" if r.method == "recalled" else f"counted , {r.steps} steps"
            return Reply(f"{to_words(a)} {word[op]} {to_words(b)} is {to_words(r.value)}{tail} .", confidence=1.0, note=how)
        try:
            if op == "succ":
                return Reply(f"after {to_words(a)} comes {to_words(ns.succ(a))} .", confidence=1.0, note="counted")
            if op == "pred":
                return Reply(f"before {to_words(a)} comes {to_words(ns.pred(a))} .", confidence=1.0, note="counted")
            if op == "count":
                if b < a or b - a > 40:
                    return Reply("i do not want to count that many ." if b > a else "i can only count upwards .", kind="idk")
                seq, x = [a], a
                while x != b:
                    x = ns.succ(x)
                    seq.append(x)
                return Reply(" ".join(to_words(n) for n in seq) + " .", confidence=1.0, note="counted")
            if op in ("gt?", "lt?"):
                big = (op == "gt?")
                first_less = ns.less(a, b)
                pick = (b if first_less else a) if big else (a if first_less else b)
                return Reply(f"{to_words(pick)} is {'bigger' if big else 'smaller'} .", confidence=1.0, note="compared on the number line")
            if op in ("gtyn", "ltyn"):
                if a == b:
                    return Reply(f"no , {to_words(a)} is the same as {to_words(b)} .", confidence=1.0)
                yes = (not ns.less(a, b)) if op == "gtyn" else ns.less(a, b)
                rel = "more" if op == "gtyn" else "less"
                return Reply(f"{'yes' if yes else 'no'} , {to_words(a)} is {'' if yes else 'not '}{rel} than {to_words(b)} .", confidence=1.0)
        except CannotCount as e:
            return Reply(self._cannot(e), kind="idk", note="cannot count that")
        return Reply("i do not know .", kind="idk")

    def _cannot(self, e: Exception) -> str:
        top = self.numbers.max_taught
        return f"{e} i can count to {to_words(top)} so far ." if top >= 0 and self.numbers.can_count() else f"{e} teach me : one two three four ..."

    def _ents(self):
        return self.mem.ents if self.instances else None

    @staticmethod
    def _sentences(text: str) -> List[List[Tok]]:
        """Sentences, with quotation marks respected: a full stop inside "..." does not end the sentence unless the quote
        closes right there ("Come and get it." But ...)."""
        out, cur, inside = [], [], False
        toks = tokenize_cased(text)
        for i, t in enumerate(toks):
            cur.append(t)
            if t.text == '"':
                inside = not inside
                if not inside and len(cur) > 1 and cur[-2].text in END and (i + 1 >= len(toks) or toks[i + 1].cap):
                    out.append(cur)
                    cur = []
                continue
            if t.text in END and not inside:
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if nxt is not None and nxt.text == '"' and not (i + 2 < len(toks) and toks[i + 2].text.isalpha() and
                                                                  not toks[i + 2].cap):
                    continue          # the closing quote belongs to this sentence
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        return out

    def _meet_words(self, sent: List[Tok], rep: ReadReport) -> None:
        for t in sent:
            if t.tag not in ("N", "PN", "ADJ", "V", "ADV") or t.lemma in self.seen:
                continue
            self.seen.add(t.lemma)
            self.learn_word(t.lemma, t.text, t.known, rep, t.tag)

    def learn_word(self, lemma: str, surface: str, known: bool, rep: Optional[ReadReport] = None, tag: str = "N") -> bool:
        """The first time a word is met: ask the dictionary what it means, and learn it."""
        if known:
            gloss = self.lex.define(lemma)
            if gloss and lemma not in self.mem.defs:
                self.mem.defs[lemma] = gloss
            if self.model is not None:
                self.model.seed_category(surface)
                if tag == "N":
                    s = self.lex.best_sense(lemma, "n")
                    if s is not None and s.hypernyms and lemma not in self.lex.proper:
                        self.model.read(f"{a_an(lemma)} {lemma} is a kind of {s.hypernyms[0]} .", learn=True)
            if rep is not None:
                rep.new_words.append(lemma)
            return True
        if rep is not None and lemma not in rep.unknown_words:
            rep.unknown_words.append(lemma)
        self.pending = lemma
        return False

    def _definitional(self, p: Prop, rep: ReadReport) -> None:
        """'a flib is an animal': the text itself explains an unknown word, so the dictionary learns it."""
        if p.rel == "isa" and not p.neg and is_instance(p.subj):
            return
        if p.rel == "isa" and not p.neg and not self.lex.senses(p.subj, "n"):
            parent = p.obj
            kind = self.lex.kind(parent) or (parent if parent in ("animal", "person", "food", "place", "plant") else "")
            hyp = [parent] + [a for a, _ in self.mem.ancestors(parent)]
            self.lex.learn(p.subj, "n", kind=kind, hypernyms=hyp)
            self.mem.defs.setdefault(p.subj, f"a kind of {parent}")
            if self.model is not None:
                self.model.seed_category(p.subj)
            rep.defined.append(p.subj)
            if self.pending == p.subj:
                self.pending = None
        elif p.rel == "attr" and not self.lex.senses(p.subj, "n"):
            pass

    # ================================================================== questions
    def ask(self, question: str, prefer: Optional[Dict[str, str]] = None) -> Reply:
        """Answer from memory; if the memory has faded or has nothing, look in the notes; then (if on) put it in own words."""
        self.notes.attend(self.notes.anchors(question))              # asking about it marks it as worth attention
        reply, q, ans = self._ask_core(question, prefer)
        if reply.kind == "answer" and ans is not None and not reply.note.startswith("answered by"):
            self.mem.rehearse(ans.items[:1])                       # using a fact keeps it
        if reply.kind in ("idk", "not_understood") and self.take_notes and len(self.notes):
            noted = self._from_notes(question, q, reply)
            if noted is not None:
                reply = noted
        if self.model is not None and reply.kind == "not_understood":
            # only for questions the rules could not read at all. A question that was understood and has no answer stays
            # "i do not know": the language layer would answer a different relation ("what color is the lion ?" -> "says roar")
            a = self.model.ask(question, abstain=True)       # diffuse knowledge learned from bulk reading
            if a.strip() != "i do not know .":
                reply = Reply(a, kind="answer", note="answered by the language model alone")
        if self.model is not None and reply.kind == "answer":
            reply.fluency = self.fluency(reply.text)
        if reply.kind in ("answer", "idk") and self.style == "varied":
            reply = self._vary(reply)
        return reply

    def _ask_core(self, question: str, prefer: Optional[Dict[str, str]] = None):
        toks = tokenize_cased(question)
        tag_sentence(toks, self.lex)
        self._mark_names(toks)
        nq = ClauseParser(self.lex, self.discourse)._number_question([t.text for t in toks if t.tag != "PUNCT"], question)
        if nq is not None:                                           # a question about numbers: no word to look up
            return self._number_reply(nq), nq, None
        unk = [t for t in toks if t.tag in ("N", "PN", "ADJ", "V") and not t.known and not self._known(t.lemma)]
        if unk:
            w = unk[0].lemma
            self.pending = w
            return Reply(f"i do not know the word {w} . what is it ?", kind="unknown_word", note=w), None, None
        parser = ClauseParser(self.lex, self.discourse, self._ents(), creating=False, prefer=prefer)
        q = parser.parse_question(toks)
        if q is not None:
            q.subj = self.aliases.get(q.subj, q.subj) if q.subj_proper or q.subj in self.aliases else q.subj
        if q is not None and parser.ambiguous:
            return self._clarify(question, parser.ambiguous[0]), q, None
        self.clarifying = None
        if q is None:
            return Reply("i did not understand the question .", kind="not_understood"), None, None
        if q.kind in ("arithmetic", "number"):
            return self._number_reply(q), q, None
        ans = self.mem.query(q)
        return self._realise(q, ans), q, ans

    def _mark_names(self, toks: List[Tok]) -> None:
        """Names met while reading ("Buster Bear", then "buster") are names in a lowercase question too."""
        known = set(self.mem.proper) | set(self.aliases) | set(self.aliases.values())
        if not known:
            return
        i = 0
        while i < len(toks):
            for span in (3, 2, 1):
                name = " ".join(t.text for t in toks[i:i + span])
                if i + span <= len(toks) and name in known:
                    for t in toks[i:i + span]:
                        t.tag, t.known = "PN", True
                    i += span - 1
                    break
            i += 1

    # ---- notes: when memory fails, look it up, read it again, and so remember it
    def _from_notes(self, question: str, q: Optional[Query], reply: Reply) -> Optional[Reply]:
        hits = self.notes.search(question, k=3)
        if not hits:
            return None
        best, score = hits[0]
        if q is not None and q.kind != "define":
            saved = self.discourse
            self.discourse = Discourse()
            try:
                self.read(best.text, told=False, learn_lm=False, take_notes=False)     # read it again: recollection
            finally:
                self.discourse = saved
            ans = self.mem.query(q)
            again = self._realise(q, ans)
            if again.kind == "answer":
                self.notes.recalled(best, question)
                self.mem.rehearse(ans.items[:1])
                return Reply("my notes remind me : " + again.text, kind="answer", confidence=again.confidence,
                             inferred=again.inferred, note=f"from my notes : {best.text}")
            reply.note = f"related note : {best.text}"
            return None
        texts = " ".join(n.text if n.text.rstrip().endswith((".", "!", "?")) else n.text + " ." for n, _ in hits[:2])
        self.notes.recalled(best, question)
        return Reply("my notes say : " + texts, kind="answer", confidence=score, note="from my notes")

    def note(self, text: str) -> Optional[Note]:
        """Take a note on purpose: it is kept word for word and also read, so it is understood as well as written down."""
        n = self.notes.add(text, "note", importance=3.0)
        self.read(text, told=True, learn_lm=True, take_notes=False)
        return n

    def recall(self, cue: str, k: int = 3) -> List[Note]:
        return [n for n, _ in self.notes.search(cue, k=k, min_cover=0.3, require_anchors=False)]

    # ---- time: forgetting and sleep
    def sleep(self, units: float = 1.0, replay: bool = True, consolidate_lm: bool = False) -> Dict[str, int]:
        """Time passes. Facts and notes fade (each with a half-life that grows with use); during sleep the most recently used
        notes are replayed, which refreshes the facts they hold; the language layer can consolidate its categories."""
        self.numbers.advance(units)
        forgot = self.mem.advance(units)
        dropped = self.notes.advance(units)
        replayed = 0
        if replay and len(self.notes):
            recent = sorted((n for n in self.notes.notes.values() if n.recalls), key=lambda n: -n.last)[:5]
            saved = self.discourse
            self.discourse = Discourse()
            try:
                for n in recent:
                    self.read(n.text, told=False, learn_lm=False, take_notes=False)
                    replayed += 1
            finally:
                self.discourse = saved
        if consolidate_lm and self.model is not None:
            self.model.consolidate()
        return dict(facts_forgotten=forgot, notes_dropped=dropped, notes_replayed=replayed)

    # ---- own-words answers
    _FRAMES = {
        "sure": ["{s}", "{s}", "i know that {s0} .", "as far as i know , {s0} ."],
        "think": ["i think {s0} .", "i believe {s0} .", "if i remember right , {s0} ."],
        "unsure": ["i vaguely remember that {s0} .", "i am not sure , but i think {s0} ."],
    }

    def _vary(self, reply: Reply) -> Reply:
        """Say the same thing in another way: the wording follows how sure the memory is, and among the candidates the
        language layer's fluency (plus a little randomness, so repeated questions do not get identical replies) decides."""
        if reply.kind != "answer" or reply.note.startswith(("answered by", "from my notes")) or reply.text.startswith(
                ("yes", "no", "my notes", "i think", "which")):
            return reply
        t = reply.text.strip()
        if not t.endswith(" .") or t.count(" . ") or " : " in t:
            return reply
        s0 = t[:-2]
        s0 = s0[0].lower() + s0[1:] if s0 else s0
        c = reply.confidence
        mood = "sure" if c >= 0.8 else ("think" if c >= 0.45 else "unsure")
        cands = [f.format(s=t, s0=s0) for f in self._FRAMES[mood]]
        cands = list(dict.fromkeys(cands))
        if self.model is not None:
            scored = [(self.model.logprob(x) / max(1, len(x.split())), x) for x in cands]
            top = sorted(scored, reverse=True)[:2]
            pick = self.rng.choice(top)[1] if len(top) > 1 and top[0][0] - top[1][0] < 0.6 else top[0][1]
        else:
            pick = self.rng.choice(cands)
        return Reply(pick, kind=reply.kind, confidence=reply.confidence, inferred=reply.inferred, note=reply.note,
                     fluency=reply.fluency)

    # ---- which one? (several instances of the same noun)
    def _label(self, eid: str, among: List[str]) -> str:
        """How to name one instance so it can be told apart from the others: tom's ball / the red ball / the second ball."""
        e = self.mem.ents.items[eid]
        owners = [self.mem.ents.items[x].owner for x in among]
        if e.owner and owners.count(e.owner) == 1:
            return f"{self._owner_name(e.owner)}'s {e.lemma}"
        adjs = [f.obj for r, f in self.mem.facts_about(eid) if r == "attr" and not f.neg][:1] or list(e.adjs[:1])
        same = [x for x in among if x != eid and (adjs and adjs[0] in
                ([f.obj for r, f in self.mem.facts_about(x) if r == "attr"] + list(self.mem.ents.items[x].adjs)))]
        if adjs and not same:
            prefix = f"{self._owner_name(e.owner)}'s " if e.owner else "the "
            return f"{prefix}{adjs[0]} {e.lemma}"
        rank = sorted(among, key=lambda x: self.mem.ents.items[x].born).index(eid) + 1
        return f"the {['first', 'second', 'third', 'fourth', 'fifth'][min(rank, 5) - 1]} {e.lemma}"

    def _owner_name(self, owner: str) -> str:
        return self._label(owner, [owner]) if is_instance(owner) and owner in self.mem.ents.items else owner

    def _clarify(self, question: str, amb) -> Reply:
        lemma, ids = amb
        self.clarifying = (question, lemma, ids)
        names = [self._label(i, ids) for i in ids]
        return Reply(f"which {lemma} ? {' or '.join(names)} ?", kind="clarify", note=f"{len(ids)} {lemma}s", confidence=0.0)

    def _answer_clarification(self, text: str) -> Optional[Reply]:
        """The user replies to 'which ball ?' with 'tom's' / 'the red one' / 'the second': ask the question again."""
        question, lemma, ids = self.clarifying
        words = {t.text for t in tokenize_cased(text)}
        ordinals = ["first", "second", "third", "fourth", "fifth"]
        hits = []
        for rank, i in enumerate(sorted(ids, key=lambda x: self.mem.ents.items[x].born)):
            e = self.mem.ents.items[i]
            keys = {e.owner, ordinals[min(rank, 4)]} | set(e.adjs) | \
                {f.obj for r, f in self.mem.facts_about(i) if r == "attr" and not f.neg}
            if keys & words:
                hits.append(i)
        if len(hits) != 1:
            return None
        self.clarifying = None
        return self.ask(question, prefer={lemma: hits[0]})

    def _known(self, lemma: str) -> bool:
        return lemma in self.entities or lemma in self.mem.defs or bool(self.lex.senses(lemma)) or \
            lemma in self.lex.proper or (lemma, "isa") in {(s, r) for (s, r) in self.mem.slots}

    def fluency(self, text: str) -> Optional[float]:
        if self.model is None:
            return None
        n = max(1, len(text.split()))
        return self.model.logprob(text) / n

    # ================================================================== realisation (grammar rules)
    def _np(self, lemma: str, number: str = "sg", proper: bool = False) -> str:
        if lemma in PRONOUNS or lemma in NO_ARTICLE:
            return lemma
        if is_instance(lemma) and lemma in self.mem.ents.items:
            same = [e.id for e in self.mem.ents.of(self.mem.ents.items[lemma].lemma)]
            return self._label(lemma, same) if len(same) > 1 else "the " + self.mem.ents.items[lemma].lemma
        if proper or lemma in self.lex.proper or lemma in self.mem.proper:
            return lemma
        if number == "pl":
            return "the " + inflect_noun(lemma, "pl")
        if self.lex.kind(lemma) in ("drink", "substance", "food") and lemma in ("milk", "water", "juice", "tea", "coffee", "rice", "bread", "sand", "meat", "cheese"):
            return lemma
        return "the " + lemma

    def _inst_desc(self, eid: str) -> str:
        """'a red ball': an instance described by what is known of it (colour first)."""
        e = self.mem.ents.items[eid]
        attrs = [f.obj for r, f in self.mem.facts_about(eid) if r == "attr" and not f.neg and f.strength > 0.15]
        attrs.sort(key=lambda a: 1 if self.mem._color(a) else 0)       # a big red ball
        words = attrs[:2] + [e.lemma]
        return f"{a_an(words[0])} {' '.join(words)}"

    def _obj_np(self, lemma: str, number: str, qty: str = "") -> str:
        if lemma in PRONOUNS or lemma in NO_ARTICLE:
            return lemma
        if is_instance(lemma) and lemma in self.mem.ents.items:
            return self._inst_desc(lemma)
        if number == "pl":
            return inflect_noun(lemma, "pl")
        if self.lex.kind(lemma) == "drink" or lemma in ("rice", "bread", "meat", "cheese", "sand"):
            return lemma
        return "the " + lemma

    def _verb(self, lemma: str, number: str, proper: bool = False, past: bool = False) -> str:
        if past:
            return inflect_verb(lemma, "past")
        return inflect_verb(lemma, "base" if number == "pl" else "3sg")

    def _realise(self, q: Query, ans: Answer) -> Reply:
        idk = Reply("i do not know .", kind="idk")
        if q.kind == "arithmetic":
            if not ans.text_hint:
                return idk
            a, op, b = q.expr
            word = {"+": "plus", "-": "minus", "*": "times"}[op]
            return Reply(f"{number_word(a)} {word} {number_word(b)} is {number_word(int(ans.text_hint))} .", confidence=1.0)
        if q.kind == "actions":
            if not ans.items:
                return idk
            sub = self._np(q.subj, "pl" if q.num == "pl" else "sg", q.subj_proper)
            vps = [self._vp(f, q.num == "pl") for f in ans.items]
            tail = ""
            if len(vps) > 1:
                tail = (" , " + " , ".join(vps[1:-1]) if len(vps) > 2 else "") + " and " + vps[-1]
            return Reply(f"{sub} {vps[0]}{tail} .", confidence=ans.confidence)
        if q.kind == "define":
            facts = self._about(q.subj, q.num == "pl", q.subj_proper)
            if facts and (q.subj in self.mem.proper or not (ans.chain or ans.text_hint)):
                return Reply(facts, confidence=1.0, note="from what i have read")
            if not ans.chain and not ans.text_hint:
                return idk
            parts = []
            if ans.chain:
                w = q.subj
                parts.append(f"{a_an(w)} {w} is a kind of {ans.chain[0]} .")
            if ans.text_hint and not ans.text_hint.startswith("a kind of "):
                parts.append(f"{q.subj} means : {ans.text_hint} .")
            return Reply(" ".join(parts), confidence=1.0, note=f"from {ans.via}")
        if q.kind == "opposite":
            if not ans.items:
                return idk
            return Reply(f"the opposite of {q.subj} is {ans.items[0].obj} .", confidence=1.0, note=f"from {ans.via}")
        if q.kind == "yesno":
            if ans.verdict == "unknown" or not ans.verdict:
                return idk
            return self._yesno_reply(q, ans)
        if not ans.items:
            return idk
        top = ans.items[0]
        conf = ans.confidence
        hedge = "i think " if ans.inferred else ""
        note = f"inferred from {ans.via}" if ans.inferred else ""
        pl = q.num == "pl"
        sub = self._np(q.subj, "pl" if pl else "sg", q.subj_proper)
        if q.kind == "count":
            n = top.qty
            if n == "one":
                return Reply(f"there is one {q.subj} .", confidence=conf)
            return Reply(f"there are {n} {inflect_noun(q.subj, 'pl')} .", confidence=conf)
        if q.want == "subj":
            nums = [self.mem.subj_num.get(f.obj, "sg") for f in ans.items[:2]]
            subs = [self._np(f.obj, n) for f, n in zip(ans.items[:2], nums)]
            plural = len(subs) > 1 or nums[0] == "pl"
            obj = self._obj_np(q.obj, q.obj_num)
            if q.rel == "have" and q.obj_num == "sg" and self.mem.ents.of(q.obj):
                obj = f"{a_an(q.obj)} {q.obj}"
            return Reply(f"{' and '.join(subs)} {self._verb(q.rel, 'pl' if plural else 'sg')} {obj} .", confidence=conf)
        if q.want == "qty":
            qty = top.qty
            noun = q.obj if qty == "one" else inflect_noun(q.obj, "pl")
            return Reply(f"{hedge}{sub} {'have' if pl else 'has'} {qty} {noun} .", confidence=conf, inferred=ans.inferred, note=note)
        rel = q.rel
        if rel == "attr":
            be = ("were" if pl else "was") if top.tense == "past" else ("are" if pl else "is")
            return Reply(f"{hedge}{sub} {be} {top.obj} .", confidence=conf, inferred=ans.inferred, note=note)
        if rel == "say" and not top.obj:
            return idk
        if rel == "say":
            return Reply(f"{hedge}{sub} {self._verb('say', 'pl' if pl else 'sg')} {top.obj} .", confidence=conf, inferred=ans.inferred, note=note)
        if rel == "@loc" or rel in SPATIAL:
            prep = top.rel if top.rel in SPATIAL else "in"
            be = "are" if pl else "is"
            return Reply(f"{hedge}{sub} {be} {prep} the {top.obj} .", confidence=conf, inferred=ans.inferred, note=note)
        if rel.endswith("_@") or "_" in rel:
            # relations with a preposition: live_in, walk_to ...
            r = top.rel if "_" in top.rel else rel
            if r == "@loc":
                be = "are" if pl else "is"
                return Reply(f"{hedge}{sub} {be} in the {top.obj} .", confidence=conf, inferred=ans.inferred, note=note)
            verb, _, prep = r.partition("_")
            return Reply(f"{hedge}{sub} {self._verb(verb, 'pl' if pl else 'sg')} {prep} the {top.obj} .", confidence=conf,
                         inferred=ans.inferred, note=note)
        if rel == "can":
            return Reply(f"{hedge}{sub} can {top.obj} .", confidence=conf, inferred=ans.inferred, note=note)
        if rel == "have":
            things = [f for f in ans.items if is_instance(f.obj)][:3]
            if things and is_instance(top.obj):
                parts = ' and '.join(self._inst_desc(f.obj) for f in things)
                return Reply(f"{hedge}{sub} {'have' if pl else 'has'} {parts} .", confidence=conf)
            return Reply(f"{hedge}{sub} {'have' if pl else 'has'} {self._obj_np(top.obj, top.obj_num)} .", confidence=conf)
        if "_" in top.rel and top.rel != rel and not rel.endswith("_@"):
            verb, _, prep = top.rel.partition("_")
            return Reply(f"{hedge}{sub} {self._verb(verb, 'pl' if pl else 'sg', past=top.tense == 'past')} {prep} "
                         f"{self._obj_np(top.obj, 'sg')} .", confidence=conf, inferred=ans.inferred, note=note)
        # ordinary verb: the dog eats bones (list the strongest few)
        objs = [f for f in ans.items if f.obj][:3]
        if not objs:
            return idk
        parts = [self._obj_np(f.obj, f.obj_num) for f in objs]
        past = objs[0].tense == "past"
        return Reply(f"{hedge}{sub} {self._verb(rel, 'pl' if pl else 'sg', past=past)} {' and '.join(parts)} .", confidence=conf,
                     inferred=ans.inferred, note=note)

    def _vp(self, f, pl: bool) -> str:
        """The verb phrase of one stored fact: 'ate it', 'jumped on the pail', 'sat'."""
        verb, _, prep = f.rel.partition("_")
        v = self._verb(verb, "pl" if pl else "sg", past=f.tense == "past")
        if not f.obj:
            return v
        obj = self._obj_np(f.obj, f.obj_num)
        return f"{v} {prep + ' ' if prep and prep != '@' else ''}{obj}".strip()

    def _about(self, subj: str, pl: bool, proper: bool) -> str:
        """Everything stored about a thing, as a few sentences: what it is, what it is like, what it did."""
        facts = self.mem.facts_about(subj)
        sub = self._np(subj, "pl" if pl else "sg", proper)
        out = []
        isa = [f for r, f in facts if r == "isa" and not f.neg and f.strength > 0.15][:1]
        if isa:
            out.append(f"{sub} {'are' if pl else 'is'} {'' if pl else a_an(isa[0].obj) + ' '}{isa[0].obj} .")
        attrs = [f.obj for r, f in facts if r == "attr" and not f.neg and f.strength > 0.15][:3]
        if attrs:
            out.append(f"{sub} {'are' if pl else 'is'} {' and '.join(attrs)} .")
        acts = self.mem.query(Query("actions", subj=subj)).items[:3]
        if acts:
            vps = [self._vp(f, pl) for f in acts]
            out.append(f"{sub} {vps[0]}{(' , ' + ' , '.join(vps[1:-1]) if len(vps) > 2 else '')}{' and ' + vps[-1] if len(vps) > 1 else ''} .")
        return " ".join(out)

    def _yesno_reply(self, q: Query, ans: Answer) -> Reply:
        pl = q.num == "pl"
        sub = self._np(q.subj, "pl" if pl else "sg", q.subj_proper)
        be = "are" if pl else "is"
        if q.rel == "isa":
            stmt = f"{sub} {be} {'' if pl else a_an(q.obj) + ' '}{inflect_noun(q.obj, 'pl') if pl else q.obj}"
        elif q.rel == "attr":
            stmt = f"{sub} {be} {q.obj}"
        elif q.rel == "can":
            stmt = f"{sub} can {q.obj}"
        else:
            obj = self._obj_np(q.obj, 'sg') if q.obj else ''
            if q.rel == "have" and q.obj and self.mem.ents.of(q.obj):
                obj = f"{a_an(q.obj)} {q.obj}"
            stmt = f"{sub} {self._verb(q.rel, 'pl' if pl else 'sg')} {obj}".strip()
        if ans.verdict == "yes":
            return Reply(f"yes , {stmt} .", confidence=ans.confidence, inferred=ans.inferred,
                         note=f"inferred from {ans.via}" if ans.inferred else "")
        if q.rel == "attr" and ans.items:
            return Reply(f"no , {sub} {be} {ans.items[0].obj} .", confidence=ans.confidence)
        neg = stmt.replace(f" {be} ", f" {be} not ", 1) if f" {be} " in stmt else stmt.replace(" can ", " cannot ", 1)
        return Reply(f"no , {neg} .", confidence=ans.confidence)

    # ================================================================== chat entry point
    def converse(self, text: str) -> Reply:
        t = text.strip()
        if self.turn_time > 0:
            self.sleep(self.turn_time, replay=False)
        if self.clarifying is not None and not t.endswith("?"):
            r = self._answer_clarification(t)
            if r is not None:
                return r
        if t.endswith("?"):
            return self.ask(t)
        rep = self.read(t if t.endswith((".", "!")) else t + " .", told=True)
        bits = []
        bits += rep.number_notes
        if rep.new_words:
            bits.append("looked up " + ", ".join(rep.new_words[:6]))
        if rep.defined:
            bits.append("learned the word " + ", ".join(rep.defined))
        if rep.unknown_words:
            bits.append("not in my dictionary : " + ", ".join(rep.unknown_words) + " ; say for example ' a " + rep.unknown_words[0] + " is a ... ' to teach me")
        extra = f" ( {' ; '.join(bits)} )" if bits else ""
        ack = "ok , learned it ." if self.style != "varied" else self.rng.choice(
            ["ok , learned it .", "got it .", "i will remember that .", "noted .", "i see , thank you ."])
        return Reply(f"{ack}{extra}", kind="learned", note=f"{len(rep.props)} facts")

    # ================================================================== a word, explained
    def describe(self, word: str) -> Dict[str, object]:
        out: Dict[str, object] = {"word": word}
        a = self.lex.analyze(word)[0]
        out["reading"] = f"{a.lemma} ({a.pos}) {a.feats}" + ("" if a.known else " [not in the dictionary]")
        for pos in ("n", "v", "a"):
            s = self.lex.best_sense(a.lemma, pos)
            if s:
                out["kind"] = s.kind
                out["meaning"] = s.gloss
                out["is a kind of"] = s.hypernyms[:6]
                out["synonyms"] = s.synonyms[:6]
                out["opposites"] = s.antonyms[:4]
                break
        out["known facts"] = [(r, f.obj, round(f.strength, 1)) for r, f in self.mem.facts_about(a.lemma)[:8]]
        insts = self.mem.ents.of(a.lemma)
        if insts:
            ids = [e.id for e in insts]
            out["instances"] = {self._label(i, ids): [(r, lemma_of(f.obj), round(f.strength, 1))
                                                       for r, f in self.mem.facts_about(i)[:6]] for i in ids}
        return out

    # ================================================================== grammar checking
    def check(self, text: str) -> List[Issue]:
        """Rule-based grammar report: subject-verb agreement, determiner-noun number, a/an, unknown words."""
        issues: List[Issue] = []
        for sent in self._sentences(text):
            tag_sentence(sent, self.lex)
            issues += self._check_sentence(sent)
        return issues

    def _check_sentence(self, toks: List[Tok]) -> List[Issue]:
        out: List[Issue] = []
        words = [t for t in toks if t.text not in END and t.text != '"']
        for i, t in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else None
            if t.text in ("a", "an") and nxt is not None and nxt.tag in ("N", "ADJ", "PN"):
                want = a_an(nxt.text)
                if want != t.text:
                    out.append(Issue("article", i, f"use '{want}' before '{nxt.text}'", want))
            if t.text in SG_DET and nxt is not None and nxt.tag == "N" and nxt.feats.get("number") == "pl" and nxt.known:
                sing = nxt.lemma
                out.append(Issue("number", i + 1, f"'{t.text}' needs a singular noun", sing))
            if t.text in PL_DET and nxt is not None and nxt.tag == "N" and nxt.feats.get("number") == "sg" and nxt.known \
                    and nxt.text not in ("sheep", "fish", "deer"):
                out.append(Issue("number", i + 1, f"'{t.text}' needs a plural noun", inflect_noun(nxt.text, "pl")))
        out += self._check_agreement(words)
        for i, t in enumerate(words):
            if t.tag in ("N", "ADJ", "V") and not t.known and not self._known(t.lemma):
                out.append(Issue("unknown", i, f"unknown word '{t.text}'", ""))
        return out

    def _check_agreement(self, words: List[Tok]) -> List[Issue]:
        out: List[Issue] = []
        if not words:
            return out
        parser = ClauseParser(self.lex, Discourse())
        inverted = words[0].tag == "AUX" or words[0].tag == "WH"
        start = 0
        if inverted:
            j = 1 if words[0].tag == "AUX" else (2 if len(words) > 1 and words[1].tag == "AUX" else 1)
            aux = words[j - 1] if words[0].tag == "AUX" else (words[1] if len(words) > 1 else words[0])
            if aux.tag != "AUX":
                return out
            np, k = parser.read_np(words, j)
            if np is None:
                return out
            if not (np.pron in ("i", "you", "we", "they") or np.number == "pl"):
                pl = False
            else:
                pl = True
            self._agree_aux(aux, j - 1, pl, np, out)
            # main verb after inverted aux do/does/did/modal must be base
            if aux.text in ("do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must") and k < len(words):
                v = words[k]
                if v.tag == "V" and v.feats.get("form") in ("3sg", "past") and v.known:
                    out.append(Issue("agreement", k, f"after '{aux.text}' use the base form", v.lemma))
            return out
        np, k = parser.read_np(words, start)
        if np is None or k >= len(words):
            return out
        sing3 = not (np.number == "pl" or np.pron in ("i", "you", "we", "they"))
        v = words[k]
        if v.tag == "AUX":
            self._agree_aux(v, k, not sing3, np, out)
            if v.text in ("do", "does", "did", "can", "could", "will", "would") and k + 1 < len(words):
                w = k + 1 + (1 if words[k + 1].tag == "NEG" else 0)
                if w < len(words) and words[w].tag == "V" and words[w].feats.get("form") in ("3sg", "past") and words[w].known:
                    out.append(Issue("agreement", w, f"after '{v.text}' use the base form", words[w].lemma))
        elif v.tag == "V" and v.known:
            form = v.feats.get("form", "")
            if sing3 and form == "base" and np.pron not in ("i", "you"):
                out.append(Issue("agreement", k, f"singular subject '{np.text}' needs '{inflect_verb(v.lemma, '3sg')}'",
                                 inflect_verb(v.lemma, "3sg")))
            elif not sing3 and form == "3sg":
                out.append(Issue("agreement", k, f"plural subject '{np.text}' needs '{v.lemma}'", v.lemma))
        return out

    def _agree_aux(self, aux: Tok, idx: int, plural: bool, np: NP, out: List[Issue]) -> None:
        t = aux.text
        pron = np.pron
        table_sg = {"are": "is", "have": "has", "do": "does", "were": "was"}
        table_pl = {"is": "are", "has": "have", "does": "do", "was": "were"}
        if pron == "i":
            if t in ("is", "are"):
                out.append(Issue("agreement", idx, "use 'am' with 'i'", "am"))
            return
        if pron == "you" or plural:
            if t in table_pl:
                out.append(Issue("agreement", idx, f"plural subject needs '{table_pl[t]}'", table_pl[t]))
        else:
            if t in table_sg:
                out.append(Issue("agreement", idx, f"singular subject needs '{table_sg[t]}'", table_sg[t]))

    def correct(self, text: str) -> str:
        """Apply the rule-based fixes (and nothing else). The language model can then confirm the result."""
        sents = self._sentences(text)
        out: List[str] = []
        for sent in sents:
            tag_sentence(sent, self.lex)
            words = [t for t in sent if t.text not in END and t.text != '"']
            end = [t.text for t in sent if t.text in END]
            for iss in self._check_sentence(sent):
                if iss.suggestion and iss.kind != "unknown":
                    words[iss.index] = Tok(iss.suggestion, False, words[iss.index].tag)
            out.append(" ".join(t.text for t in words).replace(" 's", "'s") + " " + (end[0] if end else "."))
        return " ".join(out)

    # ================================================================== persistence
    @staticmethod
    def open(directory: str, lexicon: Optional[Lexicon] = None) -> "Workspace":
        """Reopen a saved workspace: its dictionary additions, semantic memory and language model."""
        from .model import HebbianLM
        lex = lexicon or Lexicon.default()
        lm_path = os.path.join(directory, "language_model.pkl")
        model = HebbianLM.load(lm_path, lexicon=lex) if os.path.exists(lm_path) else None
        ws = Workspace(model=model, lexicon=lex)
        ws.load(directory)
        return ws

    def preload(self, statements: List[str], told: bool = False) -> int:
        """Store a batch of known facts (no language-model learning); returns how many propositions were stored."""
        n = 0
        base = self.mem.base_stab
        self.mem.base_stab = base * 20            # schooling: background knowledge is long-term, it fades far more slowly
        try:
            for st in statements:
                n += len(self.read(st, told=told, learn_lm=False, take_notes=False).props)
        finally:
            self.mem.base_stab = base
        return n

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.mem.save(os.path.join(directory, "memory.json"))
        with open(os.path.join(directory, "notes.json"), "w", encoding="utf-8") as f:
            json.dump(self.notes.to_dict(), f)
        with open(os.path.join(directory, "numbers.json"), "w", encoding="utf-8") as f:
            json.dump(self.numbers.to_dict(), f)
        with open(os.path.join(directory, "lexicon_learned.json"), "w") as f:
            json.dump(self.lex.learned_entries(), f, indent=1)
        with open(os.path.join(directory, "workspace.json"), "w") as f:
            json.dump(dict(seen=sorted(self.seen), entities=sorted(self.entities), aliases=self.aliases), f)
        if self.model is not None:
            self.model.save(os.path.join(directory, "language_model.pkl"))

    def load(self, directory: str) -> None:
        self.mem.load(os.path.join(directory, "memory.json"))
        p = os.path.join(directory, "numbers.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                self.numbers.from_dict(json.load(f))
        p = os.path.join(directory, "notes.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                self.notes.from_dict(json.load(f))
        p = os.path.join(directory, "lexicon_learned.json")
        if os.path.exists(p):
            with open(p) as f:
                for e in json.load(f):
                    self.lex.learn(e["lemma"], e["pos"], e["kind"], e["hypernyms"], e["gloss"], e["antonyms"])
        p = os.path.join(directory, "workspace.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            self.seen, self.entities = set(d["seen"]), set(d["entities"])
            self.aliases = d.get("aliases", {})
        p = os.path.join(directory, "language_model.pkl")
        if os.path.exists(p) and self.model is not None:
            from .model import HebbianLM
            self.model = HebbianLM.load(p, lexicon=self.lex)
