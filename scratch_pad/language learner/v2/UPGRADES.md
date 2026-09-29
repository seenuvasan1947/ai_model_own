# v2 Upgrades Over v1

Four changes borrowed from the Titans paper's design philosophy (see `GAP_VS_TITANS.md`), all kept compatible with the original no-backprop-in-Pairs-2/3 constraint.

| # | What changed | File | What it fixes vs. v1 |
|---|---|---|---|
| 1 | Pair 3 promotion uses a **decaying running average of surprise** instead of a hard instantaneous cutoff | `pair3_longterm.py` (`running_surprise`) | A moderately-surprising fact that recurs (but never spikes hard enough alone) can now accumulate and get remembered. v1 would ignore it forever. Verified: in testing, a chunk that v1 promoted instantly took 2 chunks of accumulated surprise to promote in v2 — the intended "sustained pattern" behavior |
| 2 | Pair 2's decay rate is **adaptive** (scales between `decay_min`/`decay_max` based on cosine similarity to the current trace) instead of one fixed constant | `pair2_shortterm.py` (`_adaptive_decay`) | v1 forgot everything at the same rate regardless of relevance. v2 clears genuinely-new topics faster and keeps reinforcing/related content longer |
| 3 | Pair 2's update is split into two explicit terms — **instantaneous surprise** vs **accumulated momentum** — instead of one blended number | `pair2_shortterm.py` (`self.trace`, `self.momentum`) | More expressive short-term state; a sudden spike and a slow-building trend are no longer collapsed into the same signal |
| 4 | Memory is injected into Pair 1 as **raw vectors spliced directly into attention** (`forward_with_memory`, `generate(memory_vecs=...)`), not decoded to text and pasted into the prompt | `pair1_foundation.py`, `ask.py` | The biggest fix. v1's text round-trip (vector → decode to a few words → re-tokenize) threw away most of the vector's information. v2 feeds Pair 3's retrieved vectors and Pair 2's trace straight into Pair 1's attention as extra context positions — the same mechanism Titans calls "Memory as Context" |

## What v2 does NOT fix

- The underlying model is still tiny (358K params, same as v1) and undertrained — these changes improve mechanism fidelity, not model capacity. Generated answers are still largely incoherent.
- A new issue surfaced during testing: the retrieval similarity filter (`min_sim=0.3`) is too loose at this scale — an unrelated question still pulled in irrelevant memories at 0.71 similarity, because the undertrained embedding space makes most things look vaguely similar. Needs tuning or more training to fix.
- Still no gradient-based learned memory (Titans' actual core mechanism) — deliberately deferred, see `GAP_VS_TITANS.md`.

## One-sentence summary

v1 proved the three-tier architecture could work at all; v2 makes each tier behave more like its Titans-inspired counterpart (smarter forgetting, sustained-pattern memory, lossless vector-based context injection) without changing model size or training data — that's what v3 does instead.
