"""v7 command line: train | quiz | feed | chat | info"""
import argparse
import os
import random
import re
import sys

from config import Config
from curriculum import CLASSES
from learner import Learner, seed_all

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "v7.pt")


def get_learner(args):
    if os.path.exists(args.ckpt):
        L = Learner.load(args.ckpt)
        print(f"loaded {args.ckpt}")
    else:
        L = Learner(Config(d_model=args.d_model))
        print("new model")
    if args.seq_len and args.seq_len != L.cfg.seq_len:
        L.model.set_seq_len(args.seq_len)
        print(f"reading window set to {args.seq_len} bytes")
    print(f"device={L.device} layers={L.cfg.n_layer} d_model={L.cfg.d_model} "
          f"params={L.model.n_params() / 1e6:.2f}M step={L.step} episodes={len(L.episodes)}")
    return L


def cmd_train(L, args):
    while L.class_idx < len(CLASSES):
        c = CLASSES[L.class_idx]
        print(f"\n=== {c['name']} ===")
        for i in range(1, args.max_steps + 1):
            # 70% current class, 30% revision of earlier classes
            batch = [c["lesson"]() if L.class_idx == 0 or random.random() < 0.7
                     else random.choice(CLASSES[:L.class_idx])["lesson"]()
                     for _ in range(L.cfg.batch_size)]
            loss = L.learn(batch)
            if i % L.cfg.eval_every == 0:
                scores = [L.quiz(CLASSES[j]["quiz"], L.cfg.quiz_size) for j in range(L.class_idx + 1)]
                s = " ".join(f"c{j + 1}={x:.0%}" for j, x in enumerate(scores))
                print(f"step {L.step:6d} loss {loss:.3f} quiz {s} layers {L.cfg.n_layer}")
                L.save(args.ckpt)
                # pass only if this class is learned AND earlier classes are still remembered
                if scores[-1] >= L.cfg.pass_score and min(scores) >= L.cfg.keep_score:
                    print(f"passed {c['name']}")
                    break
        else:
            print(f"did not pass {c['name']} within {args.max_steps} steps; run train again to continue")
            L.save(args.ckpt)
            return
        L.class_idx += 1
        L.save(args.ckpt)
    print("\nall classes passed. Use `feed` and `chat` to keep learning.")


def cmd_quiz(L, args):
    for j, c in enumerate(CLASSES):
        print(f"{c['name']}: {L.quiz(c['quiz'], 100):.0%}")
        for _ in range(3):
            p, a = c["quiz"]()
            print(f"   {p!r} -> {L.generate(p, max_new=len(a) + 4, temperature=0)!r} (want {a!r})")


def split_lessons(text, max_len):
    """Blank lines separate lessons; '#' lines are comments.
    Lessons longer than max_len bytes are cut at sentence ends (. ? !) so sentences stay whole."""
    lines = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    paras = [" ".join(p.split()) for p in "\n".join(lines).split("\n\n") if p.strip()]
    out = []
    for p in paras:
        if len(p.encode()) <= max_len:
            out.append(p)
            continue
        cur = ""
        for s in re.split(r"(?<=[.?!])\s+", p):
            if cur and len((cur + " " + s).encode()) > max_len:
                out.append(cur)
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            out.append(cur)
    return out


def cmd_feed(L, args):
    text = open(args.file, encoding="utf-8", errors="replace").read()
    paras = split_lessons(text, L.cfg.seq_len - 2)
    print(f"{len(paras)} lessons from {args.file}")
    bs = L.cfg.batch_size
    # --epochs N reads the file exactly N times; otherwise repeat until the file is learned
    max_epochs = args.epochs or args.max_epochs
    for e in range(1, max_epochs + 1):
        random.shuffle(paras)
        losses = []
        n_steps = (len(paras) + bs - 1) // bs
        for k, i in enumerate(range(0, len(paras), bs), 1):
            losses.append(L.learn(paras[i:i + bs]))
            if n_steps >= 200 and k % 200 == 0:   # big file: show progress inside the epoch
                print(f"  epoch {e} step {k}/{n_steps} loss {sum(losses[-200:]) / 200:.3f} layers {L.cfg.n_layer}")
            if k % 1000 == 0:
                L.save(args.ckpt)
        avg = sum(losses) / len(losses)
        if e == 1 or e % 10 == 0:
            print(f"epoch {e} loss {avg:.3f}")
        if not args.epochs and avg < args.target_loss:
            print(f"learned the file after {e} epochs (loss {avg:.3f} < {args.target_loss})")
            break
    else:
        if not args.epochs:
            print(f"stopped at {max_epochs} epochs (loss {avg:.3f}); feed again to keep learning")
    L.save(args.ckpt)
    print("saved")


def cmd_chat(L, args):
    print("type text. /teach <text>, /sleep, /save, /quit")
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/save":
            L.save(args.ckpt)
            print("saved")
        elif line == "/sleep":
            print(L.sleep())
        elif line.startswith("/teach "):
            t = line[7:]
            for _ in range(args.repeat):
                loss = L.learn([t], lr_scale=L.cfg.chat_lr_scale)
            print(f"learned (loss {loss:.3f})")
        else:
            print(f"model> {line}[{L.answer(line, learn=not args.no_learn)}]")
    L.save(args.ckpt)


def cmd_info(L, args):
    print(L.plast.stats())
    print(f"current class: {L.class_idx + 1}/{len(CLASSES)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "quiz", "feed", "chat", "info"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=0, help="change the reading window (bytes); saved with the model")
    ap.add_argument("--max-steps", type=int, default=6000, help="max steps per class in one run")
    ap.add_argument("--epochs", type=int, default=0, help="read the file exactly N times (0 = until learned)")
    ap.add_argument("--target-loss", type=float, default=0.3, help="feed stops when loss is below this")
    ap.add_argument("--max-epochs", type=int, default=500, help="safety limit for feed")
    ap.add_argument("--repeat", type=int, default=5, help="/teach repetitions")
    ap.add_argument("--no-learn", action="store_true", help="chat without learning from input")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "feed" and not args.file:
        sys.exit("feed needs a file")
    seed_all(args.seed)
    L = get_learner(args)
    {"train": cmd_train, "quiz": cmd_quiz, "feed": cmd_feed, "chat": cmd_chat, "info": cmd_info}[args.cmd](L, args)


if __name__ == "__main__":
    main()
