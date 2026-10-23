#!/usr/bin/env python3
"""Talk to the Hebbian Semantic Workspace. It reads with understanding, remembers exactly, and keeps learning.

    python train.py                       # once: builds models/workspace/ (about half a minute)
    python chat.py                        # then talk
    python chat.py --workspace other_dir  # a different saved workspace

What you type:
    a statement                  ->  understood and stored, e.g.  the zebra eats grass .   /  a flib is an animal .
    a question (ends with ?)     ->  answered from memory, the dictionary and inference, e.g.  what does the zebra eat ?
    /word <w>                    ->  what the dictionary and memory know about a word (meaning, kind, opposites ...)
    /facts <noun>                ->  everything stored about a noun
    /check <sentence>            ->  grammar report (agreement, a/an, number, unknown words)
    /fix <sentence>              ->  the sentence with the rule-based grammar fixes applied
    /lm <question>               ->  ask the language layer alone (its diffuse knowledge)
    /say <start>                 ->  the language layer continues a sentence;   /free [n]  free sentences
    /classes                     ->  the word categories the language layer invented for itself
    /save [dir]  /stats  /help  /quit

If you use a word the dictionary cannot explain, it says so; explain it in a sentence ("a flib is an animal .") and
it learns the word.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from hebbian_lm.lexicon import Lexicon
from hebbian_lm.tokenizer import tokenize
from hebbian_lm.workspace import Workspace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default="models/workspace")
    ap.add_argument("--dictionary", choices=["auto", "mini", "wordnet"], default="auto",
                    help="auto = WordNet if data/wordnet exists, else the bundled mini dictionary")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.workspace, "memory.json")):
        sys.exit(f"{args.workspace}/ not found. Build one first:  python train.py")
    lex = Lexicon() if args.dictionary == "mini" else (Lexicon(wordnet="data/wordnet") if args.dictionary == "wordnet" else Lexicon.default())
    ws = Workspace.open(args.workspace, lexicon=lex)
    rng = np.random.default_rng(args.seed)
    src = "WordNet + mini dictionary" if lex.wn else "mini dictionary"
    print(f"workspace loaded ({src}): {ws.mem.size()} stored facts, language layer: {ws.model.stats() if ws.model else 'none'}")
    print("type /help for commands; Ctrl-D to leave.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                print(__doc__)
            elif cmd == "stats":
                print(f"facts: {ws.mem.size()}   words seen: {len(ws.seen)}   language layer: {ws.model.stats() if ws.model else None}")
            elif cmd == "word" and rest:
                for k, v in ws.describe(rest).items():
                    print(f"  {k}: {v}")
            elif cmd == "facts" and rest:
                for r, f in ws.mem.facts_about(rest)[:20]:
                    print(f"  {rest} {r} {f.qty + ' ' if f.qty else ''}{f.obj}{' (not)' if f.neg else ''}   strength {f.strength:.1f}{' told' if f.told else ''}")
            elif cmd == "check" and rest:
                issues = ws.check(rest)
                print("  no problems found" if not issues else "")
                for i in issues:
                    print(f"  {i.kind}: {i.message}" + (f"  -> {i.suggestion}" if i.suggestion else ""))
            elif cmd == "fix" and rest:
                print("ai>", ws.correct(rest))
            elif cmd == "lm" and rest and ws.model:
                print("ai>", ws.model.ask(rest, rng=rng, abstain=True))
            elif cmd == "say" and ws.model:
                print("ai>", " ".join(tokenize(rest) + ws.model.generate(rest, rng=rng)))
            elif cmd == "free" and ws.model:
                for _ in range(int(rest) if rest.isdigit() else 5):
                    print("ai>", " ".join(ws.model.generate("", rng=rng)))
            elif cmd == "classes" and ws.model:
                for k, words in ws.model.categories().items():
                    print(f"  unit {k:2d} ({len(words):3d} words): {' '.join(words[:14])}")
            elif cmd == "save":
                ws.save(rest or args.workspace)
                print(f"saved to {rest or args.workspace}")
            else:
                print("unknown command or missing argument; /help lists them")
            continue
        r = ws.converse(line)
        print("ai>", r.text + (f"    [{r.note}]" if r.note else ""))


if __name__ == "__main__":
    main()
