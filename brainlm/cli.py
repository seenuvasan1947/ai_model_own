import argparse
import os
import sys

from .agent import BrainAgent

HELP = """commands:
  /learn <text or file path>   read and learn
  <question>?                  ask (anything ending in ?)
  /wrong <right answer>        the last answer was wrong: correct it
  /ok                          the last answer was right
  /stats                       memory statistics
  /save  /quit"""


def show(ans):
    if ans.text is None:
        print("  I don't know yet. (teach me with /learn)")
        return
    print(f"  {ans.text}   [score {ans.score:.2f}]")
    for _, s, t, kind in ans.sources[1:]:
        print(f"    also ({kind}, {s:.2f}): {t}")


def chat(agent):
    print(HELP)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line == "/quit":
            break
        elif line == "/save":
            agent.save(); print("  saved")
        elif line == "/stats":
            print(" ", agent.stats())
        elif line == "/ok":
            agent.confirm(); print("  reinforced")
        elif line.startswith("/wrong "):
            err = agent.correct(line[7:].strip()); print(f"  learned (error {err:.2f})")
        elif line.startswith("/learn "):
            arg = line[7:].strip()
            text = open(arg).read() if os.path.isfile(arg) else arg
            print("  stored %d, reinforced %d" % agent.learn(text))
        elif line.endswith("?"):
            show(agent.ask(line))
        else:
            print("  (question must end with ?, or use a /command)")
    agent.save()
    print("saved to", agent.state_dir)


def main(argv=None):
    p = argparse.ArgumentParser(prog="brainlm")
    p.add_argument("--state", default="brain_state")
    p.add_argument("--encoder", choices=["hash", "gru"], default="hash")
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("learn"); l.add_argument("source", help="text or file path")
    a = sub.add_parser("ask"); a.add_argument("question")
    c = sub.add_parser("correct"); c.add_argument("question"); c.add_argument("answer")
    sub.add_parser("stats"); sub.add_parser("chat")
    t = sub.add_parser("train-encoder", help="train the GRU encoder from scratch")
    t.add_argument("file"); t.add_argument("--epochs", type=int, default=3)
    args = p.parse_args(argv)

    agent = BrainAgent(args.state, args.encoder)
    if args.cmd == "learn":
        text = open(args.source).read() if os.path.isfile(args.source) else args.source
        print("stored %d, reinforced %d" % agent.learn(text))
    elif args.cmd == "ask":
        show(agent.ask(args.question))
    elif args.cmd == "correct":
        agent.ask(args.question)
        print(f"error {agent.correct(args.answer):.2f}")
    elif args.cmd == "stats":
        print(agent.stats())
    elif args.cmd == "train-encoder":
        if args.encoder != "gru":
            sys.exit("use --encoder gru")
        lines = [s for s in open(args.file).read().splitlines() if s.strip()]
        agent.encoder.train_lm(lines, args.epochs)
    elif args.cmd == "chat":
        chat(agent); return
    agent.save()
