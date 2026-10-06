# v6-layer-1-dict-approch

Fork of v6 that changes ONLY what Layer 1 trains on. Architecture
(transformer sizes, Layer 2/3 memory mechanism, stack.py wiring) is
unchanged -- see the parent v6/V6_CHANGES.md and V5_DESIGN.md for that.

## What's different from v6

v6 trained Layer 1 on TinyStories + Simple Wikipedia + a small synthetic
grammar set -- i.e. real narrative/encyclopedic *content* got baked into
Layer 1's frozen weights, which fights the point of having Layer 2/3 as
the place content is supposed to live.

This variant trains Layer 1 on ONLY:
- `data_prep/gen_synthetic.py` -> numbers/arithmetic, calendar
  vocabulary, opposites, grammar-pattern sentences (tense, negation,
  questions, plurals). No world-facts.
- `data_prep/gen_dict_vocab.py` -> a small closed vocabulary (see
  `WORDS` dict in that file) with a one-line dictionary-style definition
  per word, 2-3 example sentences per word, and punctuation/special-
  character drill sentences.

No TinyStories, no Wikipedia. `download_tinystories.py` /
`download_wikipedia.py` were deleted from `data_prep/` since this
variant never uses them.

## The out-of-vocabulary boundary (by design)

Anything not in `gen_dict_vocab.WORDS` (or the closed sets in
`gen_synthetic.py`) is, on purpose, something Layer 1 never saw. This
is the direct answer to "what happens with a new word it hasn't seen" --
expect degraded, subword-fragmented, low-quality output for genuinely
novel words, same failure mode the project hit before with Sherlock
Holmes vocabulary (see tokenizer.py's docstring).

The intended fix is NOT to retrain Layer 1 per new word. It's to feed
the new word/content through `continual_read.py` so Layer 2/3 picks it
up as memory -- Layer 1 still parses the *sentence structure* around the
new word using its grammar competence, even if the word itself is an
imperfect embedding.

## To grow this

- Add entries to `WORDS` in `gen_dict_vocab.py` to expand the closed
  vocabulary (currently ~37 words -- intentionally small to start;
  expand freely).
- Re-run in order: `gen_synthetic.py` -> `gen_dict_vocab.py` ->
  `build_corpus.py` -> `expand_qa.py` -> `tokenizer.py` -> `train_layer1.py`.
- `Layer1Config.vocab_size` / `TrainConfig.max_iters` in `config.py` are
  sized for this much smaller corpus (~45K words) -- raise them if you
  grow the vocabulary substantially.
