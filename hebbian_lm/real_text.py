#!/usr/bin/env python3
"""Sanity check on REAL English, outside the synthetic world: children's books from Project Gutenberg.

    python real_text.py                      # downloads ~4 MB (NLTK's copy of the Gutenberg texts) once

It trains the same model on a few simple books, then reports held-out next-word perplexity (against
a plain word n-gram baseline made from the very same synapses), the categories it found among real
words, and some free generation. Expect fluent-looking fragments, not meaning: see README.

The readout settings below (absolute-discount smoothing, category grammar on, topic gate and sentence-frame
memory off) were chosen on a validation split of the training books, not on the test sentences. The topic
and frame mechanisms were designed for the closed synthetic world and make real-text perplexity worse.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import urllib.request
import zipfile
from collections import Counter

import numpy as np

from hebbian_lm import Config, HebbianLM
from hebbian_lm.tokenizer import END_TOKENS, split_sentences, tokenize

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"
BOOKS = ["gutenberg/burgess-busterbrown.txt", "gutenberg/bryant-stories.txt", "gutenberg/carroll-alice.txt"]


def load_books(cache: str) -> str:
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "gutenberg.zip")
    if not os.path.exists(path):
        print("downloading", URL)
        with urllib.request.urlopen(URL, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    out = []
    with zipfile.ZipFile(path) as z:
        for b in BOOKS:
            out.append(z.read(b).decode("utf-8", errors="ignore"))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="data")
    ap.add_argument("--vocab", type=int, default=3000, help="keep the most frequent words, others become 'unk'")
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dictionary", choices=["none", "mini", "wordnet"], default="none",
                    help="give the language layer a dictionary (wordnet needs python fetch_wordnet.py)")
    args = ap.parse_args()

    text = re.sub(r"\s+", " ", load_books(args.cache))
    sents = [s for s in split_sentences(tokenize(text)) if 3 <= len(s) <= args.max_len]
    cut = int(len(sents) * (1 - args.test_frac))
    train, test = sents[:cut], sents[cut:]
    freq = Counter(w for s in train for w in s)
    keep = {w for w, _ in freq.most_common(args.vocab)} | END_TOKENS
    norm = lambda s: [w if w in keep else "unk" for w in s]
    train, test = [norm(s) for s in train], [norm(s) for s in test]
    ntok = sum(map(len, train))
    print(f"{len(train):,} training sentences, {ntok:,} tokens, {len(keep):,} word types; {len(test):,} test sentences")

    from hebbian_lm.lexicon import Lexicon
    lexicon = {"none": None, "mini": Lexicon(), "wordnet": Lexicon(wordnet=os.path.join(args.cache, "wordnet"))}[args.dictionary]
    m = HebbianLM(Config(seed=args.seed, consolidate_every=2000, smoothing="ad", topic_gain=0.0,
                         class_mix=0.6), lexicon=lexicon)
    import time
    t = time.time()
    m.read_lines(" ".join(s) for s in train)
    m.consolidate()
    print(f"trained in {time.time() - t:.0f}s: {m.stats()}")

    def ppl(sents, **flags):
        lp = n = 0
        for s in sents:
            lp += m.logprob(s, **flags)
            n += len(s)
        return math.exp(-lp / n)

    base = dict(use_class=False, use_topic=False, use_frame=False)
    real = dict(use_frame=False)          # category grammar on, topic gate off (cfg), frame memory off
    print(f"\nheld-out perplexity on {sum(map(len, test)):,} unseen tokens (lower is better; 'unk' counts as a word):")
    print(f"  word n-grams only (the same synapses, categories switched off) : {ppl(test, **base):7.1f}")
    print(f"  + Hebbian-learned category grammar                              : {ppl(test, **real):7.1f}")
    m.cfg.topic_gain = 0.75
    print(f"  + topic gate and frame memory (synthetic-world settings)        : {ppl(test):7.1f}   <- hurts on real text")
    m.cfg.topic_gain = 0.0

    print("\ncategories the model found among real words (most frequent members):")
    cats = sorted(m.categories().items(), key=lambda kv: -len(kv[1]))
    for k, ws in cats[:14]:
        print(f"  {len(ws):4d} words: {' '.join(ws[:14])}")

    rng = np.random.default_rng(args.seed)
    print("\nfree generation (temperature 0.7):")
    for _ in range(8):
        print("  ", " ".join(m.generate("", rng=rng, temperature=0.7, max_len=25, use_frame=False)))
    print("\ncontinuations:")
    for prompt in ["the little bear", "once upon a time", "said the", "she was very"]:
        print(f"  {prompt} ->", " ".join(m.generate(prompt, rng=rng, temperature=0.7, max_len=18, use_frame=False)))


if __name__ == "__main__":
    main()
