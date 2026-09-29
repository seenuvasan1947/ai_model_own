# PLAN_FUTURE: Later Stages (Decide From Scratch)

Nothing here is committed. Each item is re-decided once `PLAN_CURRENT.md` works.

## 1. Generation (decoder)

- Add a small GRU decoder that generates text conditioned on the current state plus retrieved memories.
- Essentially retrieval-augmented generation, but with memory the model writes itself.
- Open question: train the decoder from scratch or reuse a small pretrained one.

## 2. Imagination / lookahead

- A small predictor takes (state, candidate decision) and predicts the outcome.
- Before committing, simulate 2-3 candidates (drafted by the decoder), score predicted outcomes against stored good/bad memories, pick the best.
- Brain analogue: prefrontal mental simulation; related to JEPA and model-based RL.
- Keep to 1-2 steps of lookahead on CPU.
- **Caveat:** imagination is only as good as the predictor, and early on it will be poor. Build it last, after recall and error-driven memory update are solid.

## 3. Richer feedback signals

- Real feedback is scarce in text-only learning. Beyond next-token error and user corrections, consider:
  - self-consistency checks against existing memory (contradiction = error),
  - user thumbs up/down,
  - task success signals if a task environment is added.

## 4. Salience beyond surprise

- Survival/relevance and emotion-like tagging to decide what is worth remembering.
- Repetition-based strengthening and spaced re-exposure for consolidation.

## 5. Backbone upgrades

- Swap GRU for Mamba/SSM or a tiny transformer with a local attention window, and compare.
- Goal: better quality without losing streaming and CPU efficiency.

## 6. Scale and evaluation

- Real text streams of any length (needle-in-haystack style tests, books, summaries).
- Compare against baselines: frozen model, RAG only, fine-tuning.
- Track forgetting, memory size, and recall accuracy over time.

## 7. Research output (optional)

- Write up as an arXiv preprint or workshop paper, building on Titans/fast-weights work.
- Position: small, CPU-scale, architecture-focused continual learning with error-driven memory.

## Decision points

- Does the memory system beat a no-memory baseline? If not, revisit the design before adding generation.
- Is the GRU encoder enough, or does it need Mamba or attention?
- Is there a usable real-feedback source, or do we stay with prediction error and corrections?

---

# STATUS (updated after implementation in `titan_way/`)

| Item | Status | Where / result |
|---|---|---|
| 1. Generation (decoder) | Built | `titan_way/decoder.py`, `train_decoder.py`. Tiny transformer trained from scratch on synthetic data. Held-out exact match 0.74 on unseen random names; 3 of 7 real examples correct (e.g. "The capital of India is New Delhi."). Others garble names, and the self-check falls back to the stored text. |
| 2. Imagination / lookahead | Built (simplified) | `agent.py::answer()`. Drafts candidate answers, scores each by memory support + grounding check - past-mistake penalty, picks the best. It is a scoring step, not a learned outcome predictor. |
| 3. Richer feedback | Built | Contradiction detection on teach, `/wrong` with a mistake memory, `/good`, and the grounding self-check. No automatic next-sentence error signal. |
| 4. Salience beyond surprise | Built | `!` importance tag, repetition strengthening, salience-weighted replay, `/sleep` with decay and forgetting. Not built: emotion / survival tagging. |
| 5. Backbone upgrades | Partly | GRU vs local-attention transformer compared: the GRU failed to learn copying (exact match 0.10), the transformer reached 0.61 -> 0.86 with longer training. Mamba/SSM not built. |
| 6. Scale and evaluation | Done (small) | `titan_way/eval_scale.py`, 3 seeds. See `TITAN_APPROACH.md`. A plain retrieval baseline beats the Titan memory at 100 facts. |
| 7. Research output | Not started | Not code. |

Decision points from above, answered so far:
- **Does the memory beat a baseline?** No. Plain retrieval matched or beat it at every size tested (1.00 vs 0.78 at N=100). The Titan memory's only advantage is fixed size (16,384 params).
- **Is the GRU enough?** No. It could not learn to copy names. The local-attention transformer could.
