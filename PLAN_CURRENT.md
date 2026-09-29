# PLAN_CURRENT: What We Build Now

Goal: a small, CPU-friendly, brain-inspired model that learns continually, remembers across restarts, and learns from its mistakes.

Starting point: `od/full_model.py` (6-layer PyTorch prototype on a toy Task A -> Task B test). It has not been run yet.

## Architecture (memory system, kept from the design)

| Layer | Role |
|---|---|
| 0. Encoder | Slow, stable weights; text -> vector |
| 1. Working buffer | Recent input, temporary |
| 2. Surprise/salience gate | Decides what is worth storing |
| 3. Fast memory | Decaying, quickly updated, not trained by backprop |
| 4. Consolidation | Periodically compress repeated/important items into gists |
| 5. Long-term store | Persistent key-value store on disk |
| 6. Router | Sends input to the right expert module |

Backbone: no full attention. Recurrent, streaming encoder (GRU first). Long-range memory comes from retrieval, not attention over all tokens.

## Build order

1. **Run and debug `full_model.py`.** Check that Task A accuracy survives Task B (forgetting test) and that memory norms, store sizes and expert usage look sane.
2. **Mandatory recall.** Before every decision, query the long-term store and condition the output on the retrieved episodes.
3. **Error-driven memory update.**
   - Prediction error = actual - predicted.
   - Big error: write a strong "mistake" memory with context and correction, with boosted salience.
   - Small error: strengthen the existing memory slightly.
   - Wrong stored gist: overwrite or down-weight it; old entries decay.
4. **Persistence.** Save and load all memory to disk; verify the model resumes after a restart.
5. **Encoder choice.**
   - Phase A: pretrained small embedder (e.g. MiniLM) to test the memory logic in isolation.
   - Phase B: from-scratch GRU encoder on a small corpus (e.g. TinyStories); compare with A.

## Feedback signal (the hard part)

Real "actual outcome" is scarce in text-only learning. Practical start:
- next-token or next-sentence prediction error as the automatic signal,
- plus explicit corrections from the user.

## Success criteria

- Retains Task A after learning Task B, better than a baseline without memory.
- A corrected mistake is not repeated on the next similar input.
- State survives a restart.
- Runs on CPU only.

## Out of scope for now

Generation, imagination/lookahead, survival/emotion signals. See `PLAN_FUTURE.md`.
