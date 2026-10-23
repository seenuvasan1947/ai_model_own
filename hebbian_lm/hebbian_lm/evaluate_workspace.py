"""Does the workspace store and retrieve information better than the language model alone, and does it understand?

Benchmarks (all generated, all deterministic given a seed):

A. **Storage and retrieval at scale.** Many animals the language model never saw in training (zebra, sparrow, ...) each get
   five facts (colour, sound, home, number of legs, food), told once in plain sentences; then every fact is asked back.
   Compared: the language model alone (told once and told three times, with and without a dictionary) against the workspace.
   The number of stored facts grows, so interference between facts shows up.
B. **Words it has never met.** Statements and questions that use dictionary words absent from training.
C. **Pronouns across sentences.** Short stories ("the zebra is big . it eats grass .") followed by questions.
D. **Reasoning through "is a".** "is a zebra an animal ?", "is a dog a bird ?" and an animal told only by definition.
E. **Grammar repair.** Sentences with one agreement/number error: does ``correct`` restore the original, and does it
   leave correct sentences alone?
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List

import numpy as np

from .curriculum import Language
from .lexicon import Lexicon
from .model import HebbianLM
from .workspace import Workspace

OOV_ANIMALS = ["zebra", "sparrow", "kitten", "puppy", "wolf", "tiger", "giraffe", "goat", "fox", "snake", "turtle",
               "eagle", "squirrel", "sheep", "chicken", "dolphin", "whale", "shark", "butterfly", "bee"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink", "gray", "purple"]
SOUNDS = ["woof", "meow", "moo", "quack", "oink", "roar", "neigh", "ribbit", "hoot", "tweet", "squeak", "growl", "hiss", "buzz"]
PLACES = ["park", "garden", "farm", "forest", "jungle", "pond", "house", "river", "zoo", "field", "beach", "village"]
FOODS = ["grass", "apple", "carrot", "cheese", "bread", "meat", "rice", "corn"]
LEGS = ["two", "four", "six", "eight"]


def make_facts(n_animals: int, seed: int = 0) -> List[dict]:
    """Five facts per animal: statement, question, and the value that must appear in the answer."""
    rng = random.Random(seed)
    facts = []
    for a in OOV_ANIMALS[:n_animals]:
        c, s, p, l, f = rng.choice(COLORS), rng.choice(SOUNDS), rng.choice(PLACES), rng.choice(LEGS), rng.choice(FOODS)
        facts += [
            dict(rel="color", subj=a, value=c, statement=f"the {a} is {c} .", question=f"what color is the {a} ?"),
            dict(rel="sound", subj=a, value=s, statement=f"the {a} says {s} .", question=f"what does the {a} say ?"),
            dict(rel="home", subj=a, value=p, statement=f"the {a} lives in the {p} .", question=f"where does the {a} live ?"),
            dict(rel="legs", subj=a, value=l, statement=f"the {a} has {l} legs .", question=f"how many legs does the {a} have ?"),
            dict(rel="food", subj=a, value=f, statement=f"the {a} eats the {f} .", question=f"what does the {a} eat ?"),
        ]
    return facts


def _hit(answer: str, value: str) -> bool:
    return value in answer.replace(".", " ").split() and "do not know" not in answer


def benchmark_storage(lm: HebbianLM, lm_dict: HebbianLM, lexicon: Lexicon, sizes=(2, 6, 12, 20), seed: int = 0) -> List[dict]:
    rows = []
    for n in sizes:
        facts = make_facts(n, seed)
        row = dict(animals=n, facts=len(facts))
        # ---- language model alone (told once, told three times; without / with a dictionary)
        for label, base, reps in [("lm_no_dict_x1", lm, 1), ("lm_no_dict_x3", lm, 3), ("lm_dict_x1", lm_dict, 1), ("lm_dict_x3", lm_dict, 3)]:
            m = copy.deepcopy(base)
            for f in facts:
                m.tell(f["statement"], repeats=reps)
            rng = np.random.default_rng(0)
            ok = sum(_hit(m.ask(f["question"], rng=rng, abstain=True), f["value"]) for f in facts)
            row[label] = round(ok / len(facts), 3)
        # ---- workspace
        ws = Workspace(model=None, lexicon=Lexicon())
        for f in facts:
            ws.read(f["statement"], told=True)
        ok = sum(_hit(ws.ask(f["question"]).text, f["value"]) for f in facts)
        row["workspace"] = round(ok / len(facts), 3)
        ws2 = Workspace(model=copy.deepcopy(lm_dict), lexicon=Lexicon())
        for f in facts:
            ws2.read(f["statement"], told=True)
        ok = sum(_hit(ws2.ask(f["question"]).text, f["value"]) for f in facts)
        row["workspace_with_lm"] = round(ok / len(facts), 3)
        if n == sizes[-1]:
            rng = np.random.default_rng(0)
            m = copy.deepcopy(lm_dict)
            for f in facts:
                m.tell(f["statement"], repeats=3)
            per: Dict[str, List[bool]] = {}
            per_ws: Dict[str, List[bool]] = {}
            for f in facts:
                per.setdefault(f["rel"], []).append(_hit(m.ask(f["question"], rng=rng, abstain=True), f["value"]))
                per_ws.setdefault(f["rel"], []).append(_hit(ws.ask(f["question"]).text, f["value"]))
            row["by_relation_lm_dict_x3"] = {k: round(sum(v) / len(v), 2) for k, v in per.items()}
            row["by_relation_workspace"] = {k: round(sum(v) / len(v), 2) for k, v in per_ws.items()}
        rows.append(row)
    return rows


def benchmark_unknown_words(lm_dict: HebbianLM, lm: HebbianLM) -> dict:
    """Words the model has never seen but the dictionary knows: are they usable straight away?"""
    words = ["zebra", "sparrow", "kitten", "puppy", "wolf", "tiger", "giraffe", "goat", "fox", "snake"]
    out: Dict[str, object] = {}
    for label, base in (("without_dictionary", lm), ("with_dictionary", lm_dict)):
        m = copy.deepcopy(base)
        k = m.category_of("dog")
        hit = sum(m.seed_category(w) == k for w in words)
        out[f"{label}: right category before any sentence"] = f"{hit}/{len(words)}"
    ws = Workspace(model=None, lexicon=Lexicon())
    right = 0
    for w in words:
        ws.read(f"the {w} is big .")
        r = ws.ask(f"is a {w} an animal ?").text
        right += r.startswith("yes")
    out["workspace: 'is a X an animal?' right after one sentence"] = f"{right}/{len(words)}"
    right = 0
    for w in words:
        r = ws.ask(f"what is a {w} ?").text
        right += "kind of" in r
    out["workspace: 'what is a X?' answered from the dictionary"] = f"{right}/{len(words)}"
    return out


def benchmark_pronouns(seed: int = 0) -> dict:
    rng = random.Random(seed)
    ok = n = 0
    for _ in range(40):
        a, b = rng.sample(OOV_ANIMALS, 2)
        fa, fb = rng.sample(FOODS, 2)
        ca, cb = rng.sample(COLORS, 2)
        ws = Workspace(model=None, lexicon=Lexicon())
        ws.read(f"the {a} is {ca} . it eats the {fa} . the {b} is {cb} . it eats the {fb} .")
        for q, v in [(f"what does the {a} eat ?", fa), (f"what does the {b} eat ?", fb), (f"what color is the {a} ?", ca), (f"what color is the {b} ?", cb)]:
            n += 1
            ok += _hit(ws.ask(q).text, v)
    return dict(questions=n, correct=ok, accuracy=round(ok / n, 3))


def benchmark_reasoning() -> dict:
    ws = Workspace(model=None, lexicon=Lexicon())
    ws.read("a flib is an animal . a florp is a flib . the florp is big .")
    tests = [("is a zebra an animal ?", "yes"), ("is a dog a bird ?", "no"), ("is a sparrow a bird ?", "yes"), ("is a tiger a mammal ?", "yes"),
             ("is a florp an animal ?", "yes"), ("is a florp a bird ?", "no"), ("is a zebra a car ?", "no"), ("is a apple a fruit ?", "yes"),
             ("is the florp big ?", "yes"), ("is the florp small ?", "no")]
    ok = 0
    detail = []
    for q, want in tests:
        a = ws.ask(q).text
        good = a.startswith(want)
        ok += good
        detail.append((q, a, good))
    return dict(correct=ok, total=len(tests), detail=detail)


def benchmark_grammar_repair(lang: Language, n: int = 400, seed: int = 1) -> dict:
    ws = Workspace(model=None, lexicon=Lexicon())
    pairs = lang.minimal_pairs(n * 3, seed=seed)
    by_kind: Dict[str, List[bool]] = {}
    for good, bad, kind in pairs:
        if kind not in ("agreement", "number"):
            continue
        fixed = ws.correct(bad).replace(" ?", " ?")
        by_kind.setdefault(kind, []).append(fixed.strip() == good.strip())
    # false alarms on correct sentences
    sample = random.Random(seed).sample(sorted(lang.grammar), 400)
    false_alarm = sum(bool([i for i in ws.check(s) if i.kind != "unknown"]) for s in sample)
    out = {k: round(sum(v) / len(v), 3) for k, v in by_kind.items()}
    out["n_pairs"] = {k: len(v) for k, v in by_kind.items()}
    out["false_alarms_on_correct_sentences"] = f"{false_alarm}/{len(sample)}"
    return out


THINGS = ["ball", "hat", "book", "cup", "box", "bag", "car", "toy"]
OWNERS = ["tom", "anna", "ben", "lily", "sam", "mia"]


def benchmark_entities(instances: bool = True, n: int = 60, seed: int = 0) -> dict:
    """F. Two owners each have a different thing of the same kind ("tom has a red ball . anna has a blue ball ."):
    are they kept apart? With ``instances=False`` the old behaviour (one noun = one entity) is measured."""
    rng = random.Random(seed)
    score = {"possessive_color": 0, "asks_which": 0, "follow_up": 0, "what_do_they_have": 0, "how_many": 0, "same_thing_again": 0}
    for _ in range(n):
        thing = rng.choice(THINGS)
        (o1, o2), (c1, c2) = rng.sample(OWNERS, 2), rng.sample(COLORS, 2)
        ws = Workspace(model=None, lexicon=Lexicon(), instances=instances)
        ws.read(f"{o1} has a {c1} {thing} . {o2} has a {c2} {thing} .")
        score["possessive_color"] += all(_hit(ws.ask(f"what color is {o}'s {thing} ?").text, c) for o, c in ((o1, c1), (o2, c2)))
        r = ws.ask(f"what color is the {thing} ?")
        score["asks_which"] += r.kind == "clarify"
        r2 = ws.converse(f"{o2}'s") if r.kind == "clarify" else r
        score["follow_up"] += r.kind == "clarify" and _hit(r2.text, c2) and c1 not in r2.text.split()
        score["what_do_they_have"] += all(f"{c} {thing}" in ws.ask(f"what does {o} have ?").text for o, c in ((o1, c1), (o2, c2)))
        score["how_many"] += ws.ask(f"how many {thing}s does {o1} have ?").text.startswith(f"{o1} has one {thing}")
        ws.read(f"{o1} has a {c1} {thing} .")                         # the same thing, told again
        score["same_thing_again"] += ws.ask(f"how many {thing}s does {o1} have ?").text.startswith(f"{o1} has one {thing}")
    out = {k: round(v / n, 3) for k, v in score.items()}
    out["stories"] = n
    return out


def _fact_pool(n: int, seed: int = 0) -> List[dict]:
    return make_facts(min(20, (n + 4) // 5), seed=seed)[:n]


def benchmark_forgetting(n_facts: int = 40, times=(0, 100, 200, 400, 600, 800, 1000, 1500), seed: int = 0) -> List[dict]:
    """G. Human-like forgetting and the note keeper. 40 facts are told once; time passes; each fact is asked.

    * ``plain``: nothing is done in between (no rehearsal) and there are no notes: facts fade away.
    * ``rehearsed``: every fact is asked every 100 time units (each answer strengthens it and slows its forgetting).
    * ``notes``: no rehearsal, but every sentence was also written down; when memory fails the note is looked up and read again.
    * ``false_answers``: questions about things never told, with notes on: should stay 'i do not know'.
    Numbers are the share of facts answered correctly."""
    facts = _fact_pool(n_facts, seed)
    never = [f"what color is the {a} ?" for a in ("hippo", "panda", "camel", "otter", "moose", "gecko", "llama", "parrot")]
    rows = []
    for T in times:
        row = {"time": T}
        for name, take_notes, rehearse in (("plain", False, False), ("rehearsed", False, True), ("notes", True, False)):
            ws = Workspace(model=None, lexicon=Lexicon(), take_notes=take_notes)
            for f in facts:
                ws.read(f["statement"])
            t = 0
            while t < T:
                step = min(100, T - t)
                ws.sleep(step, replay=False)
                t += step
                if rehearse and t < T:
                    for f in facts:
                        ws.ask(f["question"])
            row[name] = round(sum(_hit(ws.ask(f["question"]).text, f["value"]) for f in facts) / len(facts), 3)
            if name == "notes":
                row["false_answers"] = round(sum("do not know" not in ws.ask(q).text for q in never) / len(never), 3)
        rows.append(row)
    return rows


def benchmark_real_qa(path: str = "") -> dict:
    """H. Real sentences in, questions out. Each labelled sentence (resources/real_gold.json) is read alone by a fresh
    workspace; for every labelled (subject, verb, object) with a noun/name subject and an object, ask "what did S V ?" and
    check that the object appears in the answer. A pronoun subject stays a pronoun (no antecedent in a single sentence) and is asked as such."""
    from .evaluate_real import PRON, load_gold
    out = {}
    for split in ("dev", "test"):
        n = ok = 0
        for item in load_gold(path)[split]:
            qs = [(s, v, o) for s, v, o in item["gold"] if o and v != "be" and o not in PRON]
            if not qs:
                continue
            ws = Workspace(model=None, lexicon=Lexicon.default(), take_notes=False)
            ws.read(item["s"])
            for s, v, o in qs:
                n += 1
                ok += o in ws.ask(f"what did {s} {v} ?").text.replace(".", " ").split()
        out[split] = dict(questions=n, correct=ok, accuracy=round(ok / max(1, n), 3))
    return out
