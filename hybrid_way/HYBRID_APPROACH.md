# HYBRID_APPROACH: episodic index + neural memory (our own design)

Compared against `titan_way` (pure mini-Titans neural memory) on the same text and questions.

## Design

| Layer | Brain analogue | Implementation |
|---|---|---|
| Working buffer | Short-term memory | Last 8 (question, answer) items seen |
| **Episodic index** (new) | Hippocampus | `episodic.py`: embedding + text per fact, **one-shot write, no training**, nearest-neighbour recall |
| **Neural memory** | Cortex | The same mini-Titans module as `titan_way` (surprise, momentum, decay), fixed 16,384 parameters |
| **Consolidation** (`sleep`) | Sleep | Replays unconsolidated index entries into the neural memory, with rehearsal; optional pruning of entries the neural memory has absorbed |
| Recall | | 1) index (by meaning), 2) neural memory, 3) "I don't know" |
| Feedback | | `/wrong` writes a correction to the index at once; the neural memory relearns it at the next sleep |

Files: `episodic.py`, `agent.py` (HybridAgent), `eval_data.py`, `compare.py`, logs `compare_minilm.log`, `compare_hash.log`.

## How to run the comparison

```bash
cd hybrid_way
../.venv/bin/python compare.py minilm    # about 3 minutes
../.venv/bin/python compare.py hash
```

Use the agent from Python:
```python
from agent import HybridAgent
a = HybridAgent("my_state", encoder="minilm")
a.learn_text(open("book.txt").read())   # one-shot into the index
a.sleep()                                # consolidate into the neural memory
print(a.answer("some question?"))        # text, path (index / neural / refused), score
```

## Test

- Taught: 4 topics in ONE memory (parliament, photosynthesis, solar system, water cycle) = 24 sentences.
- Questions: 14 reusing the text's words, 23 with different words (same meaning), 8 unrelated (should be refused).
- Refusal thresholds are chosen on a separate calibration topic (immune system), not on the test questions.
- Neural systems: mean of 3 seeds (± is the overall standard deviation).

## Results: MiniLM (meaning-based) encoder

| System | same | diff | refuse | overall | ± |
|---|---|---|---|---|---|
| plain lookup (nearest sentence, no memory model) | 0.93 | 0.74 | 0.88 | **0.82** | - |
| titan_way (neural memory only) | 0.83 | 0.64 | 0.12 | 0.61 | 0.05 |
| hybrid: index only | 0.93 | 0.74 | 0.88 | **0.82** | - |
| hybrid: neural only (after sleep) | 0.83 | 0.65 | 0.12 | 0.61 | 0.06 |
| hybrid: index + neural | 0.93 | 0.77 | 0.12 | 0.70 | 0.03 |
| hybrid + prune (index emptied) | 0.86 | 0.65 | 0.12 | 0.62 | 0.05 |

(In the logged size column the two "index only" / "neural only" rows were mislabelled; the script has since been fixed. Index only = 24 vectors, 0 parameters; neural only = 0 vectors, 16,384 parameters.)

## Results: hash (word-overlap) encoder

| System | same | diff | refuse | overall |
|---|---|---|---|---|
| plain lookup | 0.71 | 0.09 | 0.12 | 0.29 |
| titan_way | 0.50 | 0.10 | 0.08 | 0.22 |
| hybrid: index + neural | 0.71 | 0.13 | 0.08 | 0.30 |
| hybrid + prune | 0.52 | 0.12 | 0.08 | 0.24 |

The hash encoder is poor for every system (different-wording questions about 0.1). The comparison that matters is the MiniLM one.

## Verdict: which is better?

- **hybrid beats titan_way**: 0.70 vs 0.61 overall (MiniLM). The index answers 95% of the questions it can.
- **But neither beats the plain embedding lookup (0.82).** The hybrid's index alone equals it exactly, because it is the same lookup.
- **The neural memory has not earned its place on this test.**
  - Alone it scores 0.61, lower than the index on every group.
  - **It almost never says "I don't know"** (refusal 0.12 vs 0.88 for the index). Its recall similarity does not tell known from unknown, so it answers unrelated questions.
  - Added behind the index it gained 1 to 2 questions on "different wording" (0.77 vs 0.74, within noise) but lost the refusals, so overall it is worse than the index alone (0.70 vs 0.82).
- **What the neural memory does give:** compression. With pruning, the 24 sentences are held by 16,384 numbers and an empty index, with same-wording accuracy 0.86. That is the fixed-size memory idea, but at a large cost in accuracy and no ability to refuse.

## Caveats

- One set of 4 topics, 45 questions; a difference of one question is about 0.02. Results within about 0.05 should be read as ties.
- Only 24 sentences. The neural memory's argument (fixed size) matters at scales much larger than this, which we have not tested.
- The neural memory is small (128 dimensions), untuned, and uses random projections. The learned outer-loop projections from the Titans paper are not implemented, and might help.
- Why the neural score fails to separate known from unknown was not investigated. It may be fixable.
- Refusal cutoffs were calibrated on a single 5-sentence topic.

## Suggested next steps

1. Find out why the neural recall score cannot refuse, and fix it (for example a separate "familiarity" signal from surprise).
2. Test at larger scale (hundreds to thousands of sentences), where a growing index costs more and fixed-size memory could matter.
3. Learn the projections in an outer loop and tune the memory, and see whether the neural layer catches up.
4. If it never beats the index, keep the neural memory only for compression or for its continual-learning research value, and be clear about it.

## Chat: give it any text and ask in your own words

```bash
cd hybrid_way
../.venv/bin/python chat.py            # state kept in ./hybrid_state/
```
- Paste any sentence or paragraph, or `/learn path/to/file.txt`. Each sentence is saved at once (no training).
- Ask anything ending with `?`, in your own words. Matching is by meaning (MiniLM), not exact words.
- Below the cutoff (default 0.25 similarity) it says "I don't know". Change it with `--cutoff`.
- `/wrong <right answer>`, `/good`, `/sleep` (consolidate into the neural memory), `/stats`, `/trace on|off`, `/forget`, `/quit`.
- The neural memory is off for answering by default because it cannot refuse (`/neural on` to enable). `/sleep` still trains it.
- The first run downloads the MiniLM model (about 90 MB).

Answers are **extractive**: it returns the stored sentence that best matches the question. It does not compose new sentences (that needs a language model; see `PLAN_FUTURE.md`).

Spot check on a paragraph not used in any tuning (5 sentences about black holes), 6 questions in different wording plus 1 unrelated: all 6 got the right sentence and the unrelated one was refused. One paragraph, so only an indication.
