"""Evaluation: what did the model actually learn?

All tests use the exact grammar in ``curriculum.Language`` so "grammatical" is not a guess.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .curriculum import Language
from .model import HebbianLM
from .tokenizer import tokenize


# ----------------------------------------------------------------------------- categories
def _nmi(a: List[str], b: List[int], w: List[float]) -> float:
    n = float(sum(w))
    ca, cb, cab = defaultdict(float), defaultdict(float), defaultdict(float)
    for x, y, wt in zip(a, b, w):
        ca[x] += wt
        cb[y] += wt
        cab[(x, y)] += wt
    ha = -sum(v / n * math.log(v / n) for v in ca.values())
    hb = -sum(v / n * math.log(v / n) for v in cb.values())
    mi = sum(v / n * math.log((v / n) / ((ca[x] / n) * (cb[y] / n))) for (x, y), v in cab.items())
    return 2 * mi / (ha + hb) if ha + hb > 0 else 0.0


def category_report(model: HebbianLM, lang: Language) -> dict:
    gold = lang.gold_tags()
    fine, pred, wt = [], [], []
    for w, tag in gold.items():
        i = model.idx.get(w, -1)
        if i >= 0 and model.cls[i] >= 0 and model.cnt[i] > 0:
            fine.append(tag)
            pred.append(int(model.cls[i]))
            wt.append(math.sqrt(model.cnt[i]))
    coarse = [t.split("_")[0] for t in fine]
    # purity: share of words whose category's majority tag equals their tag
    by_cat = defaultdict(Counter)
    for t, p, wgt in zip(fine, pred, wt):
        by_cat[p][t] += wgt
    pur = sum(c.most_common(1)[0][1] for c in by_cat.values()) / sum(wt)
    return dict(nmi_fine=round(_nmi(fine, pred, wt), 3), nmi_coarse=round(_nmi(coarse, pred, wt), 3),
                purity=round(pur, 3), categories=int(model.active.sum()), words_scored=len(fine))


# ----------------------------------------------------------------------------- minimal pairs
def minimal_pair_report(model: HebbianLM, pairs, **flags) -> dict:
    ok, tot = Counter(), Counter()
    for good, bad, kind in pairs:
        tot[kind] += 1
        tot["all"] += 1
        if model.logprob(good, **flags) > model.logprob(bad, **flags):
            ok[kind] += 1
            ok["all"] += 1
    return {k: round(ok[k] / tot[k], 3) for k in sorted(tot)} | {"n": tot["all"]}


# ----------------------------------------------------------------------------- perplexity
def perplexity(model: HebbianLM, sentences: List[str], **flags) -> float:
    lp, n = 0.0, 0
    for s in sentences:
        toks = tokenize(s)
        lp += model.logprob(toks, **flags)
        n += len(toks)
    return math.exp(-lp / max(n, 1))


# ----------------------------------------------------------------------------- generation
def generation_report(model: HebbianLM, lang: Language, n: int = 400, seed: int = 0,
                      temperature: float = 0.8, **flags) -> dict:
    rng = np.random.default_rng(seed)
    seen = new = bad = 0
    bad_examples, new_examples = [], []
    uniq = set()
    for _ in range(n):
        toks = model.generate("", rng=rng, temperature=temperature, **flags)
        s = " ".join(toks)
        uniq.add(s)
        if s in lang.trained:
            seen += 1
        elif s in lang.accepted:
            new += 1
            if len(new_examples) < 8:
                new_examples.append(s)
        else:
            bad += 1
            if len(bad_examples) < 10:
                bad_examples.append(s)
    return dict(n=n, grammatical_seen=round(seen / n, 3), grammatical_new=round(new / n, 3),
                ungrammatical=round(bad / n, 3), distinct=len(uniq),
                new_examples=new_examples, bad_examples=bad_examples)


# ----------------------------------------------------------------------------- question answering
def _hit(answer: str, value: str) -> bool:
    return value in answer.split()


def qa_report(model: HebbianLM, lang: Language, teach_reps: int = 1, seed: int = 0, **flags) -> dict:
    """1) taught facts asked back, 2) never-seen facts (chance level), 3) never-seen facts
    taught ONLY as plain statements after training, then asked (continual learning)."""
    rng = np.random.default_rng(seed)
    ask_kw = dict(rng=rng, temperature=0.3, top_k=5, **flags)
    out: Dict[str, object] = {}

    def acc(m, facts, rels=None):
        per = defaultdict(lambda: [0, 0])
        for f in facts:
            if rels and f.rel not in rels:
                continue
            a = m.ask(f.question, **ask_kw)
            per[f.rel][1] += 1
            per[f.rel][0] += (a.strip() == f.answer.strip())      # the whole answer sentence must be right
        tot = sum(v[1] for v in per.values())
        return {r: round(v[0] / v[1], 3) for r, v in sorted(per.items())} | {
            "all": round(sum(v[0] for v in per.values()) / max(tot, 1), 3), "n": tot}

    nosum = [f for f in lang.facts_train if f.rel != "sum"]
    out["taught_facts"] = acc(model, nosum)
    out["taught_sums"] = acc(model, [f for f in lang.facts_train if f.rel == "sum"])
    held = [f for f in lang.facts_held if f.rel != "sum"]
    out["unknown_facts_before_teaching"] = acc(model, held)
    m2 = copy.deepcopy(model)
    for f in held:
        m2.tell(f.statement, repeats=teach_reps)
    out["held_facts_after_statement_teaching"] = acc(m2, held)
    out["taught_facts_after_new_learning"] = acc(m2, nosum)
    examples = []
    for f in held[:6]:
        examples.append((f.question, m2.ask(f.question, **ask_kw), f.answer))
    out["examples_after_teaching"] = examples
    return out


# ----------------------------------------------------------------------------- continual learning
def _animal_category_share(model: HebbianLM, word: str) -> float:
    """Share of known animal nouns that sit in the same category as `word`."""
    from .curriculum import ANIMALS
    k = model.category_of(word)
    names = [a for a, _ in ANIMALS if a in model.idx]
    return sum(model.category_of(a) == k for a in names) / max(len(names), 1)


def continual_report(model: HebbianLM, lang: Language, pairs, seed: int = 0) -> dict:
    """What happens when the model keeps learning after training?

    1. New word: a made-up animal word is taught with 3 sentences. Does it land in the animal
       category and get used like one (checked by swapping it for 'dog' and asking the grammar)?
    2. Retention: minimal-pair accuracy and old facts before vs after the new learning.
    3. Updating a fact: how many repetitions of "the sky is red ." overwrite "the sky is blue ."?
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, object] = {}
    before_pairs = minimal_pair_report(model, pairs)["all"]
    m = copy.deepcopy(model)
    # ---- 1. new word
    for s in ["the zorp runs .", "a zorp sleeps .", "the zorp is big ."]:
        m.tell(s)
    m.consolidate()
    cat_share = _animal_category_share(m, "zorp")
    ok = n = 0
    samples = []
    for prompt in ("the zorp", "does the zorp", "can the zorp", "a zorp"):
        for _ in range(15):
            toks = m.generate(prompt, rng=rng, temperature=0.7)
            sent = prompt + " " + " ".join(toks)
            n += 1
            ok += sent.replace("zorp", "dog") in lang.accepted
            if len(samples) < 6 and _ < 2:
                samples.append(sent)
    out["new_word"] = dict(word="zorp", sentences_taught=3, in_animal_category=round(cat_share, 2),
                           generated_like_an_animal=round(ok / n, 2), samples=samples)
    # ---- 2. retention
    after_pairs = minimal_pair_report(m, pairs)["all"]
    out["retention"] = dict(minimal_pairs_before=before_pairs, minimal_pairs_after_new_word=after_pairs)
    # ---- 3. updating a fact
    flip = None
    m3 = copy.deepcopy(model)
    asked = [f for f in lang.facts_train if f.rel == "color" and f.subj == "sky"]
    if asked:
        q = asked[0].question
        for reps in range(0, 16):
            if reps:
                m3.tell("the sky is red .")
            ans = m3.ask(q, rng=np.random.default_rng(0), temperature=0.2, top_k=3)
            if ans.strip() == "the sky is red .":
                flip = reps
                break
        out["fact_update"] = dict(question=q, repetitions_needed_to_overwrite=flip,
                                  note="None = still answering the old fact after 15 repetitions")
    return out


# ----------------------------------------------------------------------------- ablations
ABLATIONS = {
    "full model": {},
    "no sentence-frame memory": {"use_frame": False},
    "no category grammar": {"use_class": False},
    "no topic memory": {"use_topic": False},
    "no copy/pointer gate": {"echo_gain": 0.0},
    "word n-grams only": {"use_class": False, "use_topic": False, "use_frame": False, "echo_gain": 0.0},
}


def ablation_report(model: HebbianLM, lang: Language, pairs, held_sentences, n_gen: int = 300) -> Dict[str, dict]:
    """Switch each mechanism off in turn (nothing is retrained) and re-measure."""
    rows: Dict[str, dict] = {}
    for name, spec in ABLATIONS.items():
        flags = {k: v for k, v in spec.items() if k.startswith("use_")}
        saved = model.cfg.echo_gain
        if "echo_gain" in spec:
            model.cfg.echo_gain = spec["echo_gain"]
        try:
            mp = minimal_pair_report(model, pairs, **flags)
            ppl = perplexity(model, held_sentences, **flags)
            gen = generation_report(model, lang, n=n_gen, temperature=0.7, **flags)
            qa = qa_report(model, lang, **flags)
        finally:
            model.cfg.echo_gain = saved
        rows[name] = dict(minimal_pairs=mp["all"], agreement=mp["agreement"], selection=mp["selection"],
                          perplexity=round(ppl, 2), ungrammatical=gen["ungrammatical"],
                          new_grammatical=gen["grammatical_new"], taught_facts=qa["taught_facts"]["all"],
                          told_once_facts=qa["held_facts_after_statement_teaching"]["all"])
    return rows


def abstention_report(model: HebbianLM, lang: Language, seed: int = 0) -> dict:
    """Does it say "i do not know" when it should, and answer when it can?"""
    kw = dict(temperature=0.3, top_k=5, abstain=True)
    rng = np.random.default_rng(seed)
    held = [f for f in lang.facts_held if f.rel != "sum"]
    taught = [f for f in lang.facts_train if f.rel != "sum"]

    def stats(m, facts):
        n = len(facts)
        idk = right = wrong = 0
        for f in facts:
            a = m.ask(f.question, rng=rng, **kw).strip()
            if a == "i do not know .":
                idk += 1
            elif a == f.answer:
                right += 1
            else:
                wrong += 1
        return dict(n=n, says_idk=round(idk / n, 3), answers_right=round(right / n, 3),
                    answers_wrong=round(wrong / n, 3))

    # known words, but a relation that was never taught for them (legs/sound/home exist only as facts)
    from .curriculum import Fact
    untaught = []
    for subj in ("rabbit", "monkey", "mouse", "dog", "cat"):
        untaught.append(Fact("legs", subj, "", "", f"how many legs does the {subj} have ?", ""))
    for subj in ("rabbit", "monkey", "mouse"):
        untaught.append(Fact("sound", subj, "", "", f"what does the {subj} say ?", ""))
    for subj in ("dog", "cat", "rabbit", "mouse"):
        untaught.append(Fact("home", subj, "", "", f"where does the {subj} live ?", ""))
    taught_q = {f.question for f in lang.facts_train}
    untaught = [f for f in untaught if f.question not in taught_q]
    out = dict(taught_facts=stats(model, taught), never_told_facts=stats(model, held))
    out["known_words_but_fact_never_taught"] = stats(model, untaught)
    m2 = copy.deepcopy(model)
    for f in held:
        m2.tell(f.statement)
    out["same_facts_after_being_told_once"] = stats(m2, held)
    return out
