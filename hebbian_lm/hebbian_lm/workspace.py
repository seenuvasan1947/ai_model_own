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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .comprehension import (END, NP, ClauseParser, Discourse, Prop, Query, Tok, number_word, tag_sentence, tokenize_cased)
from .entities import is_instance, lemma_of
from .lexicon import PL_DET, SG_DET, Lexicon, inflect_noun, inflect_verb
from .semantic import SPATIAL, Answer, SemanticMemory

VOWEL_SOUND = set("aeiou")


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
                 instances: bool = True) -> None:
        self.lex = lexicon or Lexicon.default()
        self.model = model
        if model is not None and getattr(model, "lexicon", None) is None:
            model.attach_lexicon(self.lex)
        self.mem = memory or SemanticMemory(self.lex)
        self.discourse = Discourse()
        self.seen: Set[str] = set()                 # lemmas already processed (so each is looked up once)
        self.entities: Set[str] = set()             # every subject/object lemma stored so far
        self.pending: Optional[str] = None          # a word the user has been asked to explain
        self.instances = instances                  # separate entities for separate mentions ("a red ball", "tom's ball")
        self.clarifying: Optional[tuple] = None     # (question, lemma, candidate ids) after a "which ball ?" question

    # ================================================================== reading
    def new_document(self) -> None:
        self.discourse = Discourse()

    def read(self, text: str, told: bool = True, learn_lm: bool = True) -> ReadReport:
        """Read statements: look up unknown words, understand each sentence, store what it says."""
        rep = ReadReport()
        for sent in self._sentences(text):
            end = sent[-1].text if sent and sent[-1].text in END else "."
            if end == "?":
                rep.questions += 1
                continue
            tag_sentence(sent, self.lex)
            self._meet_words(sent, rep)
            props = ClauseParser(self.lex, self.discourse, self._ents()).parse_declarative(sent)
            for p in props:
                self._definitional(p, rep)
                self.mem.add(p, told=told)
                self.entities.update(lemma_of(x) for x in (p.subj, p.obj) if x)
                rep.props.append(p)
            rep.unknown_words = [u for u in rep.unknown_words if u not in rep.defined]
            if learn_lm and self.model is not None:
                s = " ".join(t.text for t in sent if t.text not in END and t.tag != "POSS") + " ."
                if told:
                    self.model.tell(s)
                else:
                    self.model.read(s, learn=True)
        return rep

    def _ents(self):
        return self.mem.ents if self.instances else None

    @staticmethod
    def _sentences(text: str) -> List[List[Tok]]:
        out, cur = [], []
        for t in tokenize_cased(text):
            cur.append(t)
            if t.text in END:
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
        toks = tokenize_cased(question)
        tag_sentence(toks, self.lex)
        unk = [t for t in toks if t.tag in ("N", "PN", "ADJ", "V") and not t.known and not self._known(t.lemma)]
        if unk:
            w = unk[0].lemma
            self.pending = w
            return Reply(f"i do not know the word {w} . what is it ?", kind="unknown_word", note=w)
        parser = ClauseParser(self.lex, self.discourse, self._ents(), creating=False, prefer=prefer)
        q = parser.parse_question(toks)
        if q is not None and parser.ambiguous:
            return self._clarify(question, parser.ambiguous[0])
        self.clarifying = None
        if q is None:
            if self.model is not None:
                a = self.model.ask(question, abstain=True)
                if a.strip() != "i do not know .":
                    return Reply(a, kind="answer", note="answered by the language model alone")
            return Reply("i did not understand the question .", kind="not_understood")
        ans = self.mem.query(q)
        reply = self._realise(q, ans)
        if reply.kind == "idk" and self.model is not None and q.kind == "fact":
            a = self.model.ask(question, abstain=True)       # diffuse knowledge learned from bulk reading
            if a.strip() != "i do not know .":
                return Reply(a, kind="answer", note="answered by the language model alone")
        if self.model is not None and reply.kind == "answer":
            reply.fluency = self.fluency(reply.text)
        return reply

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
        if q.kind == "define":
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
            be = "are" if pl else "is"
            return Reply(f"{hedge}{sub} {be} {top.obj} .", confidence=conf, inferred=ans.inferred, note=note)
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
        # ordinary verb: the dog eats bones (list the strongest few)
        objs = [f for f in ans.items if f.obj][:3]
        if not objs:
            return idk
        parts = [self._obj_np(f.obj, f.obj_num) for f in objs]
        return Reply(f"{hedge}{sub} {self._verb(rel, 'pl' if pl else 'sg')} {' and '.join(parts)} .", confidence=conf,
                     inferred=ans.inferred, note=note)

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
        if self.clarifying is not None and not t.endswith("?"):
            r = self._answer_clarification(t)
            if r is not None:
                return r
        if t.endswith("?"):
            return self.ask(t)
        rep = self.read(t if t.endswith((".", "!")) else t + " .", told=True)
        bits = []
        if rep.new_words:
            bits.append("looked up " + ", ".join(rep.new_words[:6]))
        if rep.defined:
            bits.append("learned the word " + ", ".join(rep.defined))
        if rep.unknown_words:
            bits.append("not in my dictionary : " + ", ".join(rep.unknown_words) + " ; say for example ' a " + rep.unknown_words[0] + " is a ... ' to teach me")
        extra = f" ( {' ; '.join(bits)} )" if bits else ""
        return Reply(f"ok , learned it .{extra}", kind="learned", note=f"{len(rep.props)} facts")

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
        words = [t for t in toks if t.text not in END]
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
            words = [t for t in sent if t.text not in END]
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
        for st in statements:
            n += len(self.read(st, told=told, learn_lm=False).props)
        return n

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.mem.save(os.path.join(directory, "memory.json"))
        with open(os.path.join(directory, "lexicon_learned.json"), "w") as f:
            json.dump(self.lex.learned_entries(), f, indent=1)
        with open(os.path.join(directory, "workspace.json"), "w") as f:
            json.dump(dict(seen=sorted(self.seen), entities=sorted(self.entities)), f)
        if self.model is not None:
            self.model.save(os.path.join(directory, "language_model.pkl"))

    def load(self, directory: str) -> None:
        self.mem.load(os.path.join(directory, "memory.json"))
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
        p = os.path.join(directory, "language_model.pkl")
        if os.path.exists(p) and self.model is not None:
            from .model import HebbianLM
            self.model = HebbianLM.load(p, lexicon=self.lex)
