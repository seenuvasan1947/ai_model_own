#!/usr/bin/env python3
"""Talk to the Hebbian language learner. It keeps learning from everything you tell it.

    python chat.py                       # loads models/hebbian_lm.pkl (run train.py first)
    python chat.py --model other.pkl

What you type:
    a statement                ->  learned on the spot, e.g.  the sky is green .
    a question (ends with ?)   ->  answered from what it has learned, e.g.  what color is the sky ?
    /say <start of a sentence> ->  it continues the sentence in its own words
    /free [n]                  ->  n sentences it makes up on its own
    /teach <question> => <answer>   practise a question/answer pair
    /classes                   ->  the word categories it invented for itself
    /stats  /save [path]  /help  /quit

The model only knows the words it has met. Unknown words in a question are ignored; unknown words
in a statement are learned (a new word gets its category from the sentences you use it in).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from hebbian_lm import HebbianLM
from hebbian_lm.tokenizer import tokenize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/hebbian_lm.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(args.model):
        sys.exit(f"{args.model} not found. Train one first:  python train.py")
    m = HebbianLM.load(args.model)
    rng = np.random.default_rng(args.seed)
    print(f"loaded {args.model}: {m.stats()}")
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
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "help":
                print(__doc__)
            elif cmd == "stats":
                print(m.stats())
            elif cmd == "classes":
                for k, ws in m.categories().items():
                    print(f"  unit {k:2d} ({len(ws):3d} words): {' '.join(ws[:14])}")
            elif cmd == "say":
                print("ai>", " ".join(tokenize(rest) + m.generate(rest, rng=rng)))
            elif cmd == "free":
                for _ in range(int(rest) if rest.strip().isdigit() else 5):
                    print("ai>", " ".join(m.generate("", rng=rng)))
            elif cmd == "teach":
                if "=>" not in rest:
                    print("usage: /teach <question> => <answer>")
                else:
                    q, _, a = rest.partition("=>")
                    m.teach_qa(q, a)
                    print("ai> practised.")
            elif cmd == "save":
                path = rest.strip() or args.model
                m.save(path)
                print(f"saved to {path}")
            else:
                print("unknown command; /help lists them")
            continue
        before = m.V
        if line.endswith("?"):
            print("ai>", m.ask(line, rng=rng, abstain=True))
        else:
            m.tell(line if line.endswith((".", "!")) else line + " .")
            new = m.V - before
            print("ai> learned it" + (f" ({new} new word{'s' if new != 1 else ''})" if new else "") + ".")


if __name__ == "__main__":
    main()
