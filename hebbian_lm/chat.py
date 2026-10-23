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
    /retell [topic]              ->  tell what it has read (about a topic) in its own sentences
    /rephrase <sentence>         ->  the same sentence as a passive, a negative, a future and a question
    /note <text>                 ->  take a note on purpose (kept word for word and also understood)
    /notes [cue]                 ->  look in the notes (all of them if no cue)
    /sleep [n]                   ->  time passes (n units, default 100): facts and notes fade, used ones last longer
    /forget <note id>            ->  throw a note away
    /save [dir]  /stats  /help  /quit

Every sentence you tell it is also written down as a note. When it cannot remember a fact it looks in the notes, reads
the note again and so remembers. Each chat turn also lets a little time pass (see --time-per-turn).

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
    ap.add_argument("--style", choices=["plain", "varied"], default="varied",
                    help="varied = answers in its own words (wording follows how sure the memory is)")
    ap.add_argument("--time-per-turn", type=float, default=1.0, help="time units that pass with every chat turn (0 = none)")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.workspace, "memory.json")):
        sys.exit(f"{args.workspace}/ not found. Build one first:  python train.py")
    lex = Lexicon() if args.dictionary == "mini" else (Lexicon(wordnet="data/wordnet") if args.dictionary == "wordnet" else Lexicon.default())
    ws = Workspace.open(args.workspace, lexicon=lex)
    ws.style, ws.turn_time = args.style, args.time_per_turn
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
                print(f"facts: {ws.mem.size()}   notes: {len(ws.notes)}   time: {ws.mem.clock:.0f}   words seen: {len(ws.seen)}   "
                      f"language layer: {ws.model.stats() if ws.model else None}")
            elif cmd == "word" and rest:
                for k, v in ws.describe(rest).items():
                    print(f"  {k}: {v}")
            elif cmd == "facts" and rest:
                ids = [rest] + [e.id for e in ws.mem.ents.of(rest)]          # the kind, then each instance of it
                for who in ids:
                    for r, f in ws.mem.facts_about(who)[:20]:
                        print(f"  {who} {r} {f.qty + ' ' if f.qty else ''}{f.obj}{' (not)' if f.neg else ''}   strength {f.strength:.1f}{' told' if f.told else ''}")
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
            elif cmd == "retell":
                print("ai>", ws.retell(rest) or "i have nothing to tell yet .")
            elif cmd == "rephrase" and rest:
                for k, v in (ws.rephrase(rest) or {"": "i could not find a verb with an object in that sentence ."}).items():
                    print(f"  {k:9}{v}")
            elif cmd == "note" and rest:
                n = ws.note(rest)
                print(f"ai> noted ( note {n.id if n else '-'} )")
            elif cmd == "notes":
                found = ws.notes.search(rest, k=8, min_cover=0.3, require_anchors=False) if rest else \
                    [(n, ws.notes.retention(n)) for n in list(ws.notes.notes.values())[-12:]]
                print("  no notes" if not found else "")
                for n, sc in found:
                    print(f"  [{n.id}] {n.text}   (remembered {ws.notes.retention(n):.2f}, used {n.recalls}x)")
            elif cmd == "sleep":
                units = float(rest) if rest.replace(".", "", 1).isdigit() else 100.0
                print("  ", ws.sleep(units, consolidate_lm=True))
            elif cmd == "forget" and rest.isdigit():
                print("  forgotten" if ws.notes.forget(int(rest)) else "  no such note")
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
