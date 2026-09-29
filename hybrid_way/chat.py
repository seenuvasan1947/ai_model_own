"""Chat with the hybrid agent: give it ANY text, ask in your own words.

  python chat.py [--state hybrid_state] [--encoder minilm|hash] [--cutoff 0.25]

Teach : paste any sentence or PARAGRAPH   (split into sentences, saved instantly)
        /learn <file path>                (read a whole text file)
        question => answer                (teach a specific pair)
        !text                             (mark as IMPORTANT)
Ask   : anything ending with ?            (matches by MEANING, not exact words)
Feedback: /wrong <right answer>   /good
Memory: /sleep  (consolidate into the neural memory)   /stats   /trace on|off
        /neural on|off   (let the neural memory answer when the index can't)
        /forget   /quit

Everything is saved after each change (hybrid_state/), so restarting keeps it.
"""
import argparse
import json
import os
import re
import time

from agent import HybridAgent


def read_source(arg):
    """Text, a plain file, a markdown file, or a JSON chat export -> clean text."""
    path = os.path.expanduser(arg)
    if not os.path.isfile(path):
        return arg
    raw = open(path, encoding="utf-8", errors="ignore").read()
    if raw.lstrip().startswith(("{", "[")):
        try:
            found = []
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "text" and isinstance(v, str) and len(v) > 20:
                            found.append(v)
                        else:
                            walk(v)
                elif isinstance(o, list):
                    for x in o:
                        walk(x)
            walk(json.loads(raw))
            if found:
                raw = "\n".join(found)
        except ValueError:
            pass
    raw = re.sub(r"```.*?```", " ", raw, flags=re.S)          # code blocks
    raw = re.sub(r"[#*_`>|]+", " ", raw)                        # markdown symbols
    raw = re.sub(r"^\s*[-\d]+[.)]?\s+", "", raw, flags=re.M)   # list markers
    return raw


def learn(a, text, trace, important=False):
    def show(i, n, r):
        if n <= 12:
            if trace:
                print(f"  [{i}/{n}] saved: {r['sentence'][:70]}")
        elif i % 200 == 0 or i == n:
            print(f"  ... {i}/{n} sentences")
    rs = a.learn_text(text, important, show)
    if not rs:
        print("  (no sentences of 3+ words found)"); return
    print(f"  learned {len(rs)} sentence(s). Ask me about them, or /sleep to consolidate.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="hybrid_state")
    p.add_argument("--encoder", default="minilm", choices=["minilm", "hash"])
    p.add_argument("--cutoff", type=float, default=0.25,
                   help="minimum meaning-similarity to answer; below it, 'I don't know'")
    args = p.parse_args()
    a = HybridAgent(args.state, encoder=args.encoder, thr_idx=args.cutoff)
    a.use_neural = False          # the neural memory can't refuse (see HYBRID_APPROACH.md)
    print(__doc__.split("Everything is saved")[0])
    print(f"[encoder: {args.encoder} | cutoff {args.cutoff}]",
          f"[loaded {len(a.idx)} facts]" if a.loaded else "[fresh memory]")
    trace = True
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        elif line == "/stats":
            print(" ", a.stats())
        elif line.startswith("/trace"):
            trace = line.endswith("on"); print("  trace", "on" if trace else "off")
        elif line.startswith("/neural"):
            a.use_neural = line.endswith("on"); print("  neural fallback", "on" if a.use_neural else "off")
        elif line == "/forget":
            a = HybridAgent(args.state + "_tmp_wipe", encoder=args.encoder, thr_idx=args.cutoff)
            a.state = args.state; a.use_neural = False; a.save(); print("  memory wiped")
        elif line == "/sleep":
            r = a.sleep()
            print(f"  slept: consolidated {r['consolidated']} facts into the neural memory ({r['updates']} updates)")
        elif line == "/good":
            a.good(); print("  noted")
        elif line.startswith("/wrong "):
            try:
                a.wrong(line[7:].strip()); print(f"  learned the correction ({len(a.mistakes)} mistakes remembered)")
            except RuntimeError as e:
                print(" ", e)
        elif line.startswith("/learn "):
            t0 = time.time()
            learn(a, read_source(line[7:].strip()), trace)
            print(f"  ({time.time() - t0:.1f}s)")
        elif line.endswith("?"):
            r = a.answer(line)
            print("  " + r["text"])
            if trace:
                sims = ", ".join(f"{n} {s:.2f}" for n, s in r["tried"])
                print(f"  [path: {r['path']} | similarity: {sims} | cutoff {args.cutoff}]")
                if r["path"] != "refused":
                    for i, c in a.idx.search(a._k(line), 3)[1:]:
                        if c >= args.cutoff:
                            print(f"  [also related ({c:.2f}): {a.idx.items[i]['a'][:80]}]")
        else:
            imp = line.startswith("!")
            body = line[1:].strip() if imp else line
            if "=>" in body:
                q, ans = (s.strip() for s in body.split("=>", 1))
                r = a.teach(q, ans, imp)
                print("  saved" + (f" (replaced earlier answer '{r['conflict']}')" if r["conflict"] else ""))
            elif len(body.split()) < 3:
                print("  (too short to learn. Paste a sentence or paragraph, or ask a question ending with ?)")
            else:
                learn(a, body, trace, imp)
    a.save()
    print("saved to", a.state)


if __name__ == "__main__":
    main()
