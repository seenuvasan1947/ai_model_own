"""How well does comprehension handle real fiction? Hand-labelled sentences from Project Gutenberg books.

``resources/real_gold.json`` holds 25 development sentences (used while building the reader) and 26 test sentences
(from other books, labelled before the v2 reader was written and scored once). Each sentence lists the main
propositions a person would extract: (subject, relation, object). Scoring is lenient on purpose:

* relation = the verb lemma; ``be`` matches ``attr`` or ``isa``; a stored ``can`` matches its verb;
* an empty gold object means "don't care"; names match if one contains the other ("joe" ~ "little joe");
* a pronoun in the gold subject or object accepts any subject or object.

Reported: coverage (sentences giving at least one proposition), recall (gold propositions found), precision (extracted
propositions that match some gold proposition; extras that are true but unlabelled count against it, so it is a
pessimistic number).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Tuple

PRON = {"i", "you", "he", "she", "it", "we", "they", "him", "her", "them"}
Prop3 = Tuple[str, str, str]


def _name_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    return bool(ta and tb) and (set(ta) <= set(tb) or set(tb) <= set(ta))


def _rel_eq(gold_rel: str, p) -> bool:
    r = p[1].split("_")[0]
    if gold_rel == "be":
        return r in ("attr", "isa", "be")
    return r == gold_rel or (p[1] == "can" and p[2] == gold_rel)


def match(gold: Prop3, p: Prop3) -> bool:
    gs, gr, go = gold
    if not _rel_eq(gr, p):
        return False
    if not (gs in PRON or _name_eq(gs, p[0])):
        return False
    if go and p[1] == "can":
        return False
    return not go or go in PRON or _name_eq(go, p[2])


def load_gold(path: str = "") -> Dict[str, List[dict]]:
    path = path or os.path.join(os.path.dirname(__file__), "resources", "real_gold.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score(sentences: List[dict], extract: Callable[[str], List[Prop3]]) -> dict:
    n_cov = n_gold = found = n_sys = good = 0
    for item in sentences:
        sys_props = [tuple(p) for p in extract(item["s"])]
        gold = [tuple(g) for g in item["gold"]]
        n_cov += bool(sys_props)
        n_gold += len(gold)
        found += sum(any(match(g, p) for p in sys_props) for g in gold)
        n_sys += len(sys_props)
        good += sum(any(match(g, p) for g in gold) for p in sys_props)
    return dict(sentences=len(sentences), coverage=round(n_cov / len(sentences), 3),
                recall=round(found / max(1, n_gold), 3), precision=round(good / max(1, n_sys), 3),
                gold_props=n_gold, extracted=n_sys)


def report(extract: Callable[[str], List[Prop3]], verbose: bool = False) -> dict:
    data = load_gold()
    out = {k: score(data[k], extract) for k in ("dev", "test")}
    if verbose:
        for k in ("dev", "test"):
            for item in data[k]:
                print(item["s"][:80], "=>", extract(item["s"]))
    return out
