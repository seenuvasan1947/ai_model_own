"""Combine synthetic + TinyStories + Simple Wikipedia into one training corpus.

Order matters: shorter/higher-signal sources come first so the very early
optimizer steps see them, then bulk stories/wiki text fills in.

Output: data/foundation_corpus.txt (everything concatenated).
Run:    python data_prep/build_corpus.py

(Run this AFTER the three data-source scripts. Missing sources are noted
but don't stop the build -- you can rebuild after adding one.)
"""

import os

SOURCES = [
    "data/synthetic.txt",       # smallest but highest-density grammar/facts
    "data/simple_wiki.txt",     # factual/expository prose
    "data/tinystories.txt",     # narrative/dialogue prose, bulk
]
OUT_PATH = "data/foundation_corpus.txt"


def main():
    present = []
    missing = []
    for path in SOURCES:
        if os.path.exists(path):
            present.append(path)
        else:
            missing.append(path)

    if missing:
        print("Warning: missing source(s):")
        for m in missing:
            print(f"  {m}")
        print("Run the corresponding data_prep/download_*.py first if you want them included.\n")
    if not present:
        raise SystemExit("No sources present. Nothing to build.")

    total_words = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for path in present:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
            out.write("\n")  # separator between sources
            words = len(text.split())
            total_words += words
            print(f"  {path}: {words:,} words")

    print(f"\nWrote {OUT_PATH}: {total_words:,} words total from {len(present)} source(s).")


if __name__ == "__main__":
    main()
