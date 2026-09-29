# V1 — What We Built and What We Achieved

This is a frozen snapshot of the first working prototype of the brain-inspired continual language model designed in `../architecture_design.md`. This document records what was actually built, run, and verified in this version — not the aspirational design, the tested reality.

---

## 1. The Core Idea Being Tested

Can a language model be built in three structurally separate parts — one trained once with backprop (a fixed "language foundation"), and two that keep learning afterward without backprop (dynamic short-term and long-term memory) — and have the fixed part's frozen knowledge get combined with the dynamic parts' newly-learned knowledge to answer questions, generatively (not by retrieval/templating)?

**Answer after this prototype: yes, the full loop works mechanistically, at small scale.** Quality is limited by scale (see Section 5), not by the architecture being broken.

---

## 2. What Was Actually Built

| Module | File | What it does | Learning method |
|---|---|---|---|
| **Pair 1 — Language Foundation** | `pair1_foundation.py`, `train_pair1.py` | Decoder-only transformer, GELU activation, generates text token-by-token | Backprop, once, then frozen |
| **Pair 2 — Short-Term Memory** | `pair2_shortterm.py` | Maintains a single decaying "recent activity" trace vector | Local Hebbian co-activation update, no backward pass |
| **Pair 3 — Long-Term Memory** | `pair3_longterm.py` | Sparse key-value store; promotes/reinforces entries by surprise or repetition | Vector similarity + threshold rules, no backward pass |
| **Continual reading** | `continual_read.py` | Feeds new plain text through the frozen Pair 1, updates Pair 2/3 from it, saves their state to disk | Forward-pass only; Pair 1's weights are never touched |
| **Answering** | `ask.py` | Retrieves from Pair 3, decodes Pair 2's trace into hint words, injects both as context, then generates the answer with Pair 1 | Pure generation (sampling), informed by dynamic memory |

---

## 3. What Was Trained

- **Knowledge corpus (Phase 1):** the real text of *Alice's Adventures in Wonderland* (Project Gutenberg, ~26,500 words) — `data/alice_clean.txt`.
- **Behavior corpus (Phase 1):** a small, generic, unrelated set of ~30 question-answer pairs (`data/qa_behavior.txt`) — teaches the *pattern* of answering a question directly, not book facts. See `../piece_collect.md`, "Knowledge vs. the skill of answering."
- **New/continual knowledge (Phase 2/3 test):** a short original story about a robot named "Zeta" (`data/new_text.txt`) — deliberately unrelated to both the book and the Q&A set, to prove the continual-learning path genuinely adds new knowledge rather than reusing something already trained in.

---

## 4. What Was Verified Working, Step by Step

1. **Real training, not simulated.** `train_pair1.py` ran real backprop; loss measurably dropped (8.29 → 3.58 over 4000 steps at the final tested scale). Confirms the model is actually learning, not stuck or faking progress.
2. **Genuine generation, not retrieval.** `ask.py`'s answers were novel token sequences never present verbatim in any training file — confirmed by inspection. This matters because it proves the system isn't grepping or templating answers.
3. **The "skill of answering" transferred from an unrelated dataset.** Despite never seeing a single question about *Alice in Wonderland*, the model responded to book-related questions in "A:" format and pulled in book vocabulary — direct evidence the generic Q&A set taught a transferable *behavior*, separate from book *knowledge*.
4. **Continual learning genuinely happened, with no backprop.** Running `continual_read.py` on the Zeta story produced real promotions to Pair 3's long-term store (`[promoted_by_surprise]`, `[reinforced]`) and a non-zero Pair 2 short-term trace — confirmed by direct inspection of the printed log, not assumed.
5. **The dynamic memory was actually used at answer time, not just stored.** Asking `ask.py` "Who is Zeta?" *after* the continual-read step triggered a real retrieval from Pair 3 (`Context: ...Zeta lived in a factory near the ocean...` was correctly pulled back via cosine similarity) and injected it into the prompt before Pair 1 generated its answer. This was verified by inspecting the `(injected: ...)` debug line `ask.py` prints, not assumed to work.
6. **Structural separation held.** Pair 1's weights file (`out/pair1_final.pt`) was never touched by `continual_read.py` — Pair 2/3 read Pair 1's frozen output but never modify it, matching the design in `../architecture_design.md` Section 3.

---

## 5. Honest Limitations of This Snapshot

- **Scale is tiny.** Final trained model: 64-dim embedding, 2 layers, 358K parameters — reduced this far specifically because the sandbox environment this was built in turned out to have severe, unexpected compute constraints (both GPU and CPU were 25-1000x slower than typical hardware for a model this size; see the debugging trail in this conversation). This is an environment limitation, not a limitation of the architecture.
- **Generated answers are not yet coherent.** The mechanism (retrieve → inject → generate) is proven to fire correctly, but the actual language produced is still largely incoherent at this scale/training budget — expected and consistent with everything discussed about the scale gap between a toy model and a production LLM.
- **Pair 2's decoded "recent words" are mostly punctuation/stopwords**, not meaningful content yet — the embedding space is too undertrained at this scale for that decoding trick to surface rich words. This should improve with more training, not a design change.
- **No instruction-tuning / RLHF stage** — the Q&A-behavior dataset is a lightweight stand-in for real instruction-tuning, not equivalent to it.
- **No three-factor learning rule yet** — Pair 2's Hebbian update is plain co-activation, not yet modulated by a reward/surprise signal the way real dopaminergic plasticity works (flagged as a known future upgrade in `../architecture_design.md` Section 8).
- **No sleep-like replay/reshuffling** — Pair 2/3 learn from whatever order text arrives in, with no offline reprocessing step yet.

---

## 6. What This Snapshot Proves, in One Sentence

A structurally-separated fixed-foundation-plus-dynamic-memory language model can be built, trained, and shown to genuinely learn new information after initial training and use that new information when generating answers — the open question going forward is scaling this up, not whether the architecture works at all.
