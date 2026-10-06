# v5 Observed Model Behavior

Findings from running the already-trained v5 checkpoint (`out/layer1_final.pt`
+ `out/layer2_state.pt` + `out/layer3_state.pt`), via `continual_read.py` and
`ask.py`, on 2026-10-06. Raw run-by-run output is in `out/session_log.md`;
this file is the interpreted summary.

## Setup

- Layer 1 trained on Alice in Wonderland + Sherlock Holmes (v1-v5 corpus,
  see parent `V5_DESIGN.md`). Small vocab (~1,500), small model.
- Two facts taught via `continual_read.py` after Layer 1 was frozen:
  - "Zeta" — a robot that built boats near the ocean (`data/new_text.txt`).
  - "Milo" — a dog that found a lost lamb named Pebble (`data/new_fact_milo.txt`).
- For each, asked questions about the taught content with memory ON vs
  memory OFF (`ask.py` / `ask.py --no-memory`), which is the designed
  ablation for "does recall change generation."

## What worked

1. **Memory actually grows and persists.** Each `continual_read.py` run
   measurably increased Layer 2/3 unit and connection counts (e.g. Milo:
   L2 +27 units/+1460 connections, L3 +8 units/+86 connections), and the
   state survives across process restarts (`layer2_state.pt`/`layer3_state.pt`
   load back in correctly on the next run).
2. **The memory-feedback splice is wired through and does perturb
   generation.** Memory ON and memory OFF sometimes produce different
   output for the same question, confirming the residual-stream injection
   (`generate_with_memory_callback`) is not a no-op.

## What did NOT work

1. **No semantic recall was observed in any trial.** Across both taught
   facts and multiple phrasings ("Who is Zeta?", "Where does Zeta live?",
   "Who is Milo?", "What did Milo find?"), neither memory-ON nor
   memory-OFF generation ever reproduced any content from the taught
   text (no "robot", "factory", "boat", "ocean", "dog", "lamb", "farm",
   etc.). The model instead emitted unrelated memorized fragments from
   its original training set (e.g. "A cat says meow.", "A dog barks.",
   "A spider has eight legs.", "The opposite of big is blue.").
2. **Memory ON vs OFF differences look like noise, not signal.** The
   difference between ON/OFF outputs is not consistently in the
   direction of the taught fact — it looks like the memory delta nudges
   sampling probabilities enough to change *which* memorized fragment
   gets picked, not enough to introduce genuinely new content. Repeated
   runs of the identical question also disagree with each other (pure
   sampling variance at temperature 0.8), which is similar in magnitude
   to the ON/OFF difference — so ON/OFF divergence alone is weak evidence
   of real recall.
3. **Output is broadly incoherent/off-topic regardless of memory state.**
   Consistent with the known root cause: Layer 1 here is a narrow
   literary-style model (Alice/Sherlock), not a general English speaker,
   so even disregarding memory, answer quality is poor.

## Interpretation

v5 validates that the **architecture executes end-to-end** — Layer 2/3
grow from observation, persist across sessions, and feed back into Layer
1's residual stream in a way that changes output. It does **not**
validate that the **recall is semantically meaningful** — no evidence
across these trials that memory retrieval surfaces the *correct* taught
fact rather than an arbitrary perturbation.

This matches the pre-existing diagnosis in `V6_CHANGES.md`: you cannot
tell whether the recall *mechanism* is broken or whether Layer 1's poor
general-English competence is masking a working mechanism, because v5's
Layer 1 was never good enough at general language to make that
distinction observable. The open question carries forward to v6/
v6-layer-1-dict-approch: once Layer 1 is a competent general-English
model, repeat this same teach-and-ask ablation and check whether memory-ON
answers start containing the taught fact's actual content (not just
different wrong content than memory-OFF).
