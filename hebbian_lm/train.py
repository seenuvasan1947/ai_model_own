#!/usr/bin/env python3
"""Train the Hebbian language learner on the child-English curriculum and report what it learned.

    python train.py                      # 30k sentences in one mixed pass, about 15 s on a laptop CPU
    python train.py --docs 100000        # more data
    python train.py --stages             # staged curriculum (simple -> rich -> facts); see README: it did not help
    python train.py --quick              # tiny run (5k sentences) for a smoke test

Learning is purely local (Hebbian-family rules, see DESIGN.md); there is no backpropagation and
no GPU is needed. The trained model is saved to --out and can be loaded by chat.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from hebbian_lm import Config, HebbianLM
from hebbian_lm import evaluate as ev
from hebbian_lm.curriculum import Language


def progress_line(m: HebbianLM, lang: Language, pairs, label: str, t0: float) -> str:
    cat = ev.category_report(m, lang)
    mp = ev.minimal_pair_report(m, pairs)
    gen = ev.generation_report(m, lang, n=150, temperature=0.7)
    return (f"[{label:>9}] {time.time() - t0:5.1f}s  words {m.V:4d}  categories {cat['categories']:2d}  "
            f"category-NMI {cat['nmi_fine']:.2f}  minimal pairs {mp['all']:.2f}  "
            f"ungrammatical output {gen['ungrammatical']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=int, default=30000, help="number of training sentences (default 30000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/hebbian_lm.pkl")
    ap.add_argument("--stages", action="store_true", help="staged curriculum instead of one mixed pass")
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    ap.add_argument("--no-eval", action="store_true", help="skip the final evaluation")
    ap.add_argument("--ablations", action="store_true", help="also switch each mechanism off in turn and re-measure")
    ap.add_argument("--report", default="", help="write the final evaluation as JSON to this path")
    args = ap.parse_args()
    if args.quick:
        args.docs = 5000

    lang = Language()
    s = lang.stats()
    print(f"world: {s['grammar_sentences']:,} grammatical sentences, {s['heldout']:,} of them never shown in "
          f"training; {s['facts_train']} facts to learn, {s['facts_held']} held out")
    pairs = lang.minimal_pairs(400, seed=1)
    m = HebbianLM(Config(seed=args.seed))
    t0 = time.time()
    if not args.stages:
        m.read_lines(lang.sample_docs(args.docs, seed=args.seed))
        m.consolidate()
        print(progress_line(m, lang, pairs, "all", t0))
    else:
        plan = [(1, int(args.docs * 0.15)), (2, int(args.docs * 0.35)), (3, int(args.docs * 0.50))]
        names = {1: "stage 1", 2: "stage 2", 3: "stage 3"}
        print("stage 1: short present-tense sentences, 'be' + adjective, counting")
        print("stage 2: + past/progressive tense, transitive verbs, adjectives, places, 'there are N'")
        print("stage 3: everything (negation, questions, modals) + facts and question/answer practice")
        for stage, n in plan:
            m.read_lines(lang.stage_docs(stage, n, seed=args.seed))
            m.consolidate()
            print(progress_line(m, lang, pairs, names[stage], t0))
    print(f"trained: {m.stats()}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    m.save(args.out)
    print(f"saved to {args.out}")
    if args.no_eval:
        return

    held = random.Random(3).sample(lang.heldout, min(200, len(lang.heldout)))
    report = {}
    print("\n== categories the model invented on its own (never given part-of-speech labels) ==")
    report["categories"] = ev.category_report(m, lang)
    print(report["categories"])
    for k, words in list(m.categories().items())[:12]:
        print(f"  unit {k:2d}: {' '.join(words[:12])}")
    print("\n== grammar: is the grammatical sentence more probable than a minimally broken one? ==")
    pairs_big = lang.minimal_pairs(800, seed=2)
    report["minimal_pairs"] = ev.minimal_pair_report(m, pairs_big)
    print("   (pairs are built from sentence combinations that were never in training)", report["minimal_pairs"])
    print("\n== free generation (400 samples starting from nothing) ==")
    g = ev.generation_report(m, lang, n=400, temperature=0.7)
    report["generation"] = {k: v for k, v in g.items() if "examples" not in k}
    print(report["generation"])
    print("  new grammatical sentences it made up:", *g["new_examples"][:5], sep="\n    ")
    print("  outside the training world:", *g["bad_examples"][:5], sep="\n    ")
    print("\n== questions and facts ==")
    q = ev.qa_report(m, lang)
    report["qa"] = {k: v for k, v in q.items() if k != "examples_after_teaching"}
    for k in ("taught_facts", "taught_sums", "unknown_facts_before_teaching", "held_facts_after_statement_teaching"):
        print(f"  {k}: {q[k]}")
    print("  examples (fact told once as a plain statement, then asked):")
    for qq, a, gold in q["examples_after_teaching"][:4]:
        print(f"    {qq}  ->  {a}")
    print("\n== does it know what it doesn't know? (answers 'i do not know .' when unsupported) ==")
    report["abstention"] = ev.abstention_report(m, lang)
    for k, v in report["abstention"].items():
        print(f"  {k}: {v}")
    print("\n== keeps learning ==")
    c = ev.continual_report(m, lang, pairs)
    report["continual"] = c
    print(json.dumps({k: v for k, v in c.items()}, indent=2))
    if args.ablations:
        print("\n== what each mechanism contributes (switched off one at a time, nothing retrained) ==")
        rows = ev.ablation_report(m, lang, pairs_big, held)
        report["ablations"] = rows
        cols = ["minimal_pairs", "agreement", "selection", "perplexity", "ungrammatical", "new_grammatical",
                "taught_facts", "told_once_facts"]
        print(f"  {'':27}" + "".join(f"{c[:11]:>12}" for c in cols))
        for name, r in rows.items():
            print(f"  {name:27}" + "".join(f"{r[c]:12}" for c in cols))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
