"""Chat with the Titan-way agent and watch what it does.

  python chat.py [--state titan_state] [--encoder hash|minilm] [--decoder decoder_v2.pt]

Teach : question => answer        (a pair)
        !question => answer       (IMPORTANT: learned harder, kept longer)
        paste any sentence or PARAGRAPH  (split into sentences, each learned)
        /learn <file path>        (read a whole text file)
Ask   : anything ending with ?
Feedback: /wrong <right answer>   /good
Other : /sleep  /stats  /trace on|off  /forget  /quit

State is saved after EVERY change (titan_state/memory.pt + texts.json), so
stopping and restarting keeps everything it learned.
"""
import argparse
import os

from agent import TitanAgent, NO_ANSWER
from decoder import load_decoder


def learn_text(c, text, trace, important=False):
    def show(i, n, r):
        if trace:
            print(f"  [{i}/{n}] surprise {r['surprise_before']:.2f} -> {r['surprise_after']:.2f} "
                  f"in {r['updates']} updates: {r['sentence'][:60]}")
    rs = c.learn_text(text, important, show)
    if not rs:
        print("  (no sentences of 3+ words found)"); return
    print(f"  learned {len(rs)} sentence(s), {sum(r['updates'] for r in rs)} updates"
          + (f", {sum(1 for r in rs if r['conflict'])} replaced a conflicting belief" if any(r['conflict'] for r in rs) else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", default="titan_state")
    p.add_argument("--encoder", default="hash", choices=["hash", "minilm"],
                   help="hash = word overlap (default); minilm = pretrained, matches by meaning")
    p.add_argument("--decoder", default="decoder_v2.pt")
    a = p.parse_args()
    dec = load_decoder(a.decoder) if os.path.exists(a.decoder) else None
    c = TitanAgent(a.state, dec, encoder=a.encoder)
    print(__doc__.split("State is saved")[0])
    print(f"[decoder: {dec.backbone if dec else 'none (answers = stored text)'}]",
          f"[loaded {len(c.texts)} answers, {c.updates} updates]" if c.loaded else "[fresh memory]")
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
            print(" ", c.stats())
        elif line.startswith("/trace"):
            trace = line.endswith("on"); print("  trace", "on" if trace else "off")
        elif line == "/forget":
            c = TitanAgent(a.state + "_wiped_tmp", dec, encoder=a.encoder); c.state = a.state; c.save(); print("  memory wiped")
        elif line.startswith("/learn "):
            arg = line[7:].strip()
            text = open(os.path.expanduser(arg)).read() if os.path.isfile(os.path.expanduser(arg)) else arg
            learn_text(c, text, trace)
        elif line == "/sleep":
            r = c.sleep(); print(f"  slept: replayed {r['replayed']} batches; forgot from rehearsal: {r['dropped'] or 'nothing'}")
        elif line == "/good":
            c.good(); print("  noted (strengthened)")
        elif line.startswith("/wrong "):
            try:
                r = c.wrong(line[7:].strip())
                print(f"  learned the correction; mistake remembered ({len(c.mistakes)} total)")
                if trace:
                    print(f"  [surprise {r['surprise_before']:.2f} -> {r['surprise_after']:.2f} in {r['updates']} updates]")
            except RuntimeError as e:
                print(" ", e)
        elif line.endswith("?"):
            r = c.answer(line)
            print("  " + r["text"] + ("   (decoder output failed the self-check; showing stored text)" if r.get("fallback") else ""))
            if trace:
                print("  [recalled: " + " | ".join(f"{t[:30]} {s:.2f}" for t, s in r["ranked"][:3]) + "]")
                for k in sorted(r["candidates"], key=lambda k: -k["score"])[:4]:
                    print(f"  [imagined (T={k['temp']}): \"{k['text'][:60]}\" support {k['support']:.2f} "
                          f"grounded {'yes' if k['grounded'] else 'NO'} mistake-penalty {k['penalty']:.0f} -> score {k['score']:.2f}]")
        else:
            imp = line.startswith("!")
            body = line[1:].strip() if imp else line
            if "=>" in body:
                q, ans = (s.strip() for s in body.split("=>", 1))
                r = c.teach(q, ans, important=imp)
                if r["conflict"]:
                    print(f"  ! conflict: I believed '{r['conflict']}' for this; replaced with the new answer")
                if trace:
                    print(f"  [surprise {r['surprise_before']:.2f} -> {r['surprise_after']:.2f} in {r['updates']} updates | "
                          f"weights moved {r['moved']:.2f} | salience {r['sal']:.1f}{' IMPORTANT' if imp else ''}]")
            elif len(body.split()) < 3:
                print("  (too short to learn. Teach with 'question => answer', a sentence or paragraph, or ask a question ending with ?)")
            else:
                learn_text(c, body, trace, imp)
    c.save()
    print("saved to", c.state)


if __name__ == "__main__":
    main()
