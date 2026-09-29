# v4: Fixes for the Issues Found in v1-v3

v2/v3 improved the architecture's mechanism and scale. v4 fixes three concrete, already-diagnosed problems instead of changing the architecture further — see the "confirmed problems" list from the conversation this project comes from. All three were verified fixed by actually running the code, not just reasoned about.

---

## Fix #1: Retrieval similarity threshold was too loose

**The bug:** `ask.py`'s memory retrieval used a hardcoded `min_sim=0.3` inside `pair3_longterm.py`'s `retrieve_vecs()`. In v3 testing, asking "What is two plus two?" (completely unrelated to the stored Zeta story) still pulled in Zeta memory vectors at 0.71 and 0.43 similarity — the filter wasn't actually separating relevant from irrelevant content.

**The fix:** raised the threshold to 0.6 and moved it out of a buried magic number into `Pair3Config.retrieval_min_sim` (`config.py`), so it's a visible, tunable setting instead of hidden in a function signature.

**Verified:**
- "What is two plus two?" — no long-term memory injected at all now (correctly filtered out).
- "Who is Zeta?" — Zeta's memory *is* still retrieved (0.62 similarity, just above the new threshold) — the fix doesn't break real retrieval, only irrelevant matches.

---

## Fix #2: The tokenizer bug — trained on the wrong, smaller corpus

**The bug, found while investigating fix #3:** v3's `tokenizer.py` had `__main__` hardcoded to train on `["data/alice_clean.txt", "data/qa_behavior.txt"]` — it never included Sherlock Holmes text, even though `train_pair1.py`'s `TrainConfig.book_path` pointed at `data/combined_corpus.txt` (Alice + Sherlock) for the actual model training. So the tokenizer's vocabulary was built from a smaller subset than the model was ever trained on — Sherlock-specific words never got dedicated BPE merges, forcing them into awkward, garbled subword fragments. This is the most likely direct cause of v3's "crudow"/"loul"-style output artifacts. This was a real bug, not just a vocab-size tuning issue.

**The fix:** `tokenizer.py` now trains on `[TrainConfig.book_path, TrainConfig.qa_path]` — the exact same files the model is trained on — so there's no possibility of the tokenizer and model corpus drifting apart again. Also added `suggest_vocab_size()`, a simple heuristic (scales with the square root of total corpus word count) so vocab size isn't left as a stale fixed number as the corpus grows again in the future; raised the actual default to 6000 (the heuristic suggested 5,007 for this corpus, confirming 6000 is a reasonable choice).

**Verified:** re-running `tokenizer.py` printed `Corpus word count -> suggested vocab_size: 5,007 (config.py currently sets: 6,000)` — confirms it's now genuinely reading and sizing from the full combined corpus, not silently defaulting to a stale file list.

---

## Fix #3: No held-out evaluation existed at all

**The bug:** `estimate_loss()` in `train_pair1.py` sampled from the *same* data used for training, in every version through v3. A dropping loss number could never distinguish real generalization from the model simply memorizing the training text.

**The fix:** `train_pair1.py` now splits both the book and QA token streams into train/val (`TrainConfig.val_split`, default 10%, held out as a contiguous tail rather than randomly interleaved, to avoid leaking near-duplicate overlapping windows into both sets). Both losses are now printed and compared at every checkpoint, with an explicit overfitting flag when the gap exceeds 0.3.

**Verified — and this fix immediately surfaced a real, previously-invisible problem:**

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 8.70 | 8.70 |
| 500 | 4.59 | 5.49 |
| 1500 | 3.78 | 5.13 |
| 3000 | 3.35 | 5.09 |
| 5500 | 2.81 | **5.34** |

**Train loss dropped the whole time (matching what v3 reported and treated as "improvement"). Val loss plateaued around step 1500 and then got *worse*, not better, for the rest of training.** This means v3's reported final loss of 2.87 was overfitting, not generalization — the model was increasingly memorizing the specific training text, not learning transferable language patterns, and nothing in v1-v3 could have detected this because there was no held-out data to check against.

---

## Honest Summary

- Fix #1 and Fix #2 are clean, verified, unambiguous bug fixes — confirmed working as intended by direct testing.
- Fix #3 is the most consequential one: it doesn't just add a nice-to-have metric, it **overturns a conclusion from v3's own `COMPARISON.md`**. The "scale clearly helps" narrative from v3 needs a caveat: scale helped *training* loss, but v4's validation curve shows the model started overfitting well before training finished. The right response is not "scale doesn't help" — it's "this corpus is too small for this much model capacity and this many training steps without regularization (more dropout, early stopping, or more data), and we now have a way to actually see that, instead of being misled by training loss alone."
- **Next natural step, now that overfitting is visible:** either stop training earlier (early stopping based on val loss, roughly around step 1500-2000 where val loss was lowest), add more/stronger regularization, or grow the corpus further so 1.37M parameters has enough data to actually need to generalize rather than memorize.
