"""Download TinyStories and turn it into a lesson file (one story per lesson, blank line between).

    python3 data_prep/get_tinystories.py                 # validation split, ~19 MB, ~22k stories
    python3 data_prep/get_tinystories.py --split train --max-stories 200000   # part of the 1.9 GB train split
"""
import argparse
import os
import urllib.request

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"valid": "TinyStories-valid.txt", "train": "TinyStories-train.txt"}
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=FILES, default="valid")
    ap.add_argument("--max-stories", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    raw_dir = os.path.join(ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw = os.path.join(raw_dir, FILES[args.split])
    if not os.path.exists(raw):
        print(f"downloading {FILES[args.split]} ...")
        urllib.request.urlretrieve(BASE + FILES[args.split], raw)
    print(f"raw file: {raw} ({os.path.getsize(raw) / 1e6:.1f} MB)")

    stories = []
    with open(raw, encoding="utf-8", errors="replace") as f:
        cur = []
        for line in f:
            if "<|endoftext|>" in line:
                s = " ".join(" ".join(cur).split())
                if s.isascii() and len(s) > 20:   # skip the few broken / non-English stories
                    stories.append(s)
                cur = []
                if args.max_stories and len(stories) >= args.max_stories:
                    break
            else:
                cur.append(line)

    out = os.path.join(ROOT, "lessons", f"tinystories_{args.split}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# TinyStories (Eldan & Li, 2023). One story per lesson.\n\n")
        f.write("\n\n".join(stories) + "\n")
    print(f"wrote {len(stories)} stories -> {out}")


if __name__ == "__main__":
    main()
