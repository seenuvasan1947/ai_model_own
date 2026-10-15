"""v9 command line: en-train | feed | chat | quiz | info"""
import argparse
import os
import random
import re
import sys

from config import Config
from curriculum import all_classes
from learner import Learner, seed_all

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "v9.pt")
ENGLISH_DIR = os.path.join(HERE, "lessons", "english")
STRONG_DEFAULT = os.path.join(HERE, "lessons", "strong_words.txt")


def get_learner(args):
    if os.path.exists(args.ckpt):
        L = Learner.load(args.ckpt)
        print(f"loaded {args.ckpt}")
    else:
        L = Learner(Config(d_model=args.d_model))
        print("new model")
    if args.dict:
        L.set_dictionary(args.dict)
    if args.strong:
        L.set_strong_words(args.strong)
    elif not L.cfg.strong_words_path and os.path.exists(STRONG_DEFAULT):
        L.set_strong_words(STRONG_DEFAULT)
    c = L.cfg
    experts = [n for _, n in c.know_layers]
    print(f"device={L.device} English layers={len(c.lang_layers)} knowledge layers={len(c.know_layers)} "
          f"experts={experts} d_model={c.d_model} params={L.model.n_params() / 1e6:.2f}M step={L.step} "
          f"dictionary={len(L.dictionary)} words strong-words={len(L.strong.words)}")
    return L


def split_lessons(text, max_len):
    """Blank lines separate lessons; '#' lines are comments; long lessons are cut at sentence ends."""
    lines = [l for l in text.splitlines() if not l.lstrip().startswith(("#", "??"))]
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
            out.append(cur[:max_len])
    return out


def learn_file(L, args, kind):
    text = open(args.file, encoding="utf-8", errors="replace").read()
    if args.doc:   # whole document in long windows (up to max_ctx)
        flat = " ".join(text.split())
        win = min(args.window, L.cfg.max_ctx - 2)
        paras = [flat[i:i + win] for i in range(0, len(flat), win)]
    else:
        paras = split_lessons(text, L.cfg.train_ctx - 2)
    print(f"{len(paras)} lessons from {args.file} -> {'English (master)' if kind == 'en' else 'knowledge (slave)'}")
    bs = L.cfg.batch_size
    max_epochs = args.epochs or args.max_epochs
    for e in range(1, max_epochs + 1):
        random.shuffle(paras)
        losses, gaps = [], []
        n_steps = (len(paras) + bs - 1) // bs
        for k, i in enumerate(range(0, len(paras), bs), 1):
            losses.append(L.learn(paras[i:i + bs], kind=kind))
            gaps.append(L.last_gate)
            if n_steps >= 200 and k % 200 == 0:
                print(f"  epoch {e} step {k}/{n_steps} loss {sum(losses[-200:]) / 200:.3f} layers {L.cfg.n_layer}")
            if k % 1000 == 0:
                L.save(args.ckpt)
        avg = sum(losses) / len(losses)
        gap = sum(gaps) / len(gaps) if kind == "know" else 0.0
        if e == 1 or e % 10 == 0:
            print(f"epoch {e} loss {avg:.3f}" + (f" English gate {gap:.3f}" if kind == "know" else "")
                  + (f" doorway drift {L.last_drift:.4f}" if L.has_knowledge else ""))
        # done when the file is learned AND the knowledge gate stays closed on plain English
        if not args.epochs and avg < args.target_loss and gap < L.cfg.english_gate_ok:
            print(f"learned the file after {e} epochs (loss {avg:.3f}, English gate {gap:.3f})")
            break
    L.save(args.ckpt)
    print("saved")


def cmd_en_train(L, args):
    if args.file:
        return learn_file(L, args, "en")
    classes = all_classes(ENGLISH_DIR)
    while L.class_idx < len(classes):
        c = classes[L.class_idx]
        print(f"\n=== {c['name']} (English section) ===")
        for i in range(1, args.max_steps + 1):
            batch = [c["lesson"]() if L.class_idx == 0 or random.random() < 0.7
                     else random.choice(classes[:L.class_idx])["lesson"]()
                     for _ in range(L.cfg.batch_size)]
            loss = L.learn(batch, kind="en")
            if i % L.cfg.eval_every == 0:
                scores = [L.quiz(classes[j]["quiz"], L.cfg.quiz_size) for j in range(L.class_idx + 1)]
                s = " ".join(f"c{j + 1}={x:.0%}" for j, x in enumerate(scores))
                print(f"step {L.step:6d} loss {loss:.3f} quiz {s} English layers {len(L.cfg.lang_layers)}")
                L.save(args.ckpt)
                if scores[-1] >= L.cfg.pass_score and min(scores) >= L.cfg.keep_score:
                    print(f"passed {c['name']}")
                    break
        else:
            print(f"did not pass {c['name']} within {args.max_steps} steps; run en-train again to continue")
            L.save(args.ckpt)
            return
        L.class_idx += 1
        L.save(args.ckpt)
    print(f"\nall {len(classes)} English classes passed. Add more class files to lessons/english/ to continue.")


def cmd_feed(L, args):
    learn_file(L, args, "know")


def cmd_chat(L, args):
    print("type text. /teach <text>, /important <text>, /sleep, /save, /quit")
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
        elif line.startswith("/important "):
            t = line[11:]
            for _ in range(args.repeat):
                loss = L.learn([t], kind="know", lr_scale=L.cfg.chat_lr_scale, importance=L.cfg.max_importance)
            print(f"learned as important (loss {loss:.3f})")
        elif line.startswith("/teach "):
            t = line[7:]
            for _ in range(args.repeat):
                loss = L.learn([t], kind="know", lr_scale=L.cfg.chat_lr_scale)
            print(f"learned (loss {loss:.3f})")
        else:
            ctx = L.context_for(line)
            if ctx:
                print("  (dictionary: " + ctx.strip().replace("\n", "; ") + ")")
            print(f"model> {line}[{L.answer(line, learn=not args.no_learn)}]")
    L.save(args.ckpt)


def cmd_quiz(L, args):
    for c in all_classes(ENGLISH_DIR):
        print(f"{c['name']}: {L.quiz(c['quiz'], 100):.0%}")


def cmd_info(L, args):
    print(L.plast.stats())
    print(f"episodes: English {L.episodes.count('en')}, knowledge {L.episodes.count('know')}")
    print(f"known words: {sum(1 for v in L.words.seen.values() if v >= L.cfg.word_known)}")
    print(f"English class: {L.class_idx + 1}/{len(all_classes(ENGLISH_DIR))}")
    print(f"knowledge filed: {L.has_knowledge}, doorway drift now: {L.measure_drift():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["en-train", "feed", "chat", "quiz", "info"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--dict", default="", help="dictionary file (word: definition), remembered in the checkpoint")
    ap.add_argument("--strong", default="", help="strong-words file (one word per line), default lessons/strong_words.txt")
    ap.add_argument("--max-steps", type=int, default=6000, help="max steps per English class in one run")
    ap.add_argument("--epochs", type=int, default=0, help="read the file exactly N times (0 = until learned)")
    ap.add_argument("--target-loss", type=float, default=0.3)
    ap.add_argument("--max-epochs", type=int, default=500)
    ap.add_argument("--doc", action="store_true", help="learn the file as one long document (long context)")
    ap.add_argument("--window", type=int, default=8192, help="bytes per window with --doc (max 30000)")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "feed" and not args.file:
        sys.exit("feed needs a file")
    seed_all(args.seed)
    L = get_learner(args)
    {"en-train": cmd_en_train, "feed": cmd_feed, "chat": cmd_chat, "quiz": cmd_quiz, "info": cmd_info}[args.cmd](L, args)


if __name__ == "__main__":
    main()
