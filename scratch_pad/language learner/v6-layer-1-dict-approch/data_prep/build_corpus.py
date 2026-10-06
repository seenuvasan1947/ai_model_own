"""Combine synthetic + dict_vocab into one training corpus.

dict-approach: Layer 1 sees ONLY a closed-vocabulary, grammar/numbers/
punctuation/dictionary-definition corpus -- no TinyStories, no Wikipedia,
no narrative/encyclopedic content. Anything outside this closed set is
out-of-vocabulary for Layer 1 by design; real-world content is meant to
be fed later through Layer 2/3 via continual_read.py instead.

Output: data/foundation_corpus.txt (everything concatenated).
Run:    python data_prep/build_corpus.py

(Run this AFTER gen_synthetic.py and gen_dict_vocab.py. Missing sources
are noted but don't stop the build -- you can rebuild after adding one.)
"""

import os

SOURCES = [
    "data/synthetic.txt",       # numbers/arithmetic/calendar/opposites/grammar patterns
    "data/dict_vocab.txt",      # closed-vocabulary definitions + usage + punctuation drills
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
