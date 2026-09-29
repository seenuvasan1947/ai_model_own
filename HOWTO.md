# HOWTO: Build and Use `brainlm` (PLAN_CURRENT.md, steps 1-5)

A small, CPU-only, brain-inspired agent. It reads text, keeps the gist in a persistent memory, recalls before it answers, and learns from corrections.

## 1. Build (set up)

```bash
cd ai_project
python3 -m venv .venv
.venv/bin/pip install numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Only `numpy` is needed for the default `hash` encoder. `torch` is needed for `--encoder gru` and `od/full_model.py`.

## 2. Verify it works

```bash
.venv/bin/python demo_brainlm.py          # text agent: recall, mistake fix, restart
.venv/bin/python od/full_model.py         # toy Task A -> Task B forgetting test
```

`demo_brainlm.py` should show a wrong answer ("...Sydney") before the fix and the right one after, and the same answer after a restart.

## 3. Use it

State is saved in `./brain_state/` (change with `--state DIR`). Memory survives restarts.

### Interactive (recommended)

```bash
.venv/bin/python -m brainlm chat
```

| Command | Meaning |
|---|---|
| `/learn <text or file path>` | Read and learn (any length; files are split into sentences) |
| `anything ending in ?` | Ask. Recall happens before every answer |
| `/wrong <right answer>` | Last answer was wrong: weaken it, store a strong correction |
| `/ok` | Last answer was right: strengthen it |
| `/stats`, `/save`, `/quit` | Info / save / exit (saves automatically on quit) |

Example:

```
> /learn The capital of Italy is Rome.
> What is the capital of Italy?
  The capital of Italy is Rome.   [score 0.71]
> /wrong The capital of Italy is Rome, not Milan.
```

### One-shot commands

```bash
.venv/bin/python -m brainlm learn book.txt
.venv/bin/python -m brainlm ask "Who created Python?"
.venv/bin/python -m brainlm correct "Who created Python?" "Guido van Rossum created Python."
.venv/bin/python -m brainlm stats
```

### From Python

```python
from brainlm import BrainAgent
a = BrainAgent("brain_state")
a.learn(open("book.txt").read())
ans = a.ask("What is X?")       # ans.text, ans.score, ans.sources
a.correct("The right answer.")  # corrects the last question
a.save()
```

## 4. Encoders (plan step 5)

- **Phase A, `--encoder hash` (default):** no training, numpy only. Use it to test the memory logic in isolation.
- **Phase B, `--encoder gru`:** small byte-level GRU trained from scratch on next-byte prediction.

```bash
.venv/bin/python -m brainlm --encoder gru train-encoder corpus.txt --epochs 3
.venv/bin/python -m brainlm --encoder gru learn book.txt
.venv/bin/python -m brainlm --encoder gru chat
```

Use one encoder per state directory. Vectors from different encoders are not compatible.

## 5. How it maps to the plan

| Plan item | Where |
|---|---|
| Layer 0 encoder | `brainlm/encoders.py` |
| Layer 1 working buffer | `BrainAgent.buffer` |
| Layer 2 surprise gate | `BrainAgent.surprise()`: novelty = 1 - similarity to closest memory |
| Layer 4 consolidation | `BrainAgent.consolidate()`: store novel items, reinforce familiar ones, decay everything |
| Layer 5 long-term store | `brainlm/memory.py`, saved as `keys.npy` + `meta.json` |
| Step 2 mandatory recall | `BrainAgent.ask()` always queries the store first |
| Step 3 error-driven update | `BrainAgent.correct()` / `confirm()` |
| Step 4 persistence | `save()` / auto-load on start |
| Layers 3 and 6 (fast memory, router) | Only in `od/full_model.py` (toy test), not yet in the text agent |

## 6. Tuning knobs

`BrainAgent(surprise_gate=0.15, min_score=0.2)`, and in `memory.py` the decay rates, strength weighting and `capacity`. Lower `surprise_gate` stores more; higher `min_score` makes it say "I don't know" more often.

## 7. Honest limitations

- **It retrieves, it doesn't generate.** Answers are stored sentences ranked by similarity. Generation is in `PLAN_FUTURE.md`.
- **Gists are sentences, not real summaries.** Compression is only dedupe, reinforcement and forgetting.
- **The hash encoder matches surface words, not meaning.** A paraphrase with different words may miss. The GRU encoder is only trained on your small corpus, so expect it to be weak too.
- **Feedback is manual.** Only `/wrong` and `/ok` give a real error signal. Next-token/next-sentence prediction error as an automatic signal is not implemented yet.
- **Toy test caveats (`od/full_model.py`).** It scores 0.98 on Task A after learning B, but there is no no-memory baseline yet, so the score doesn't show the memory helps. The fast-memory norms decay to about 0, so that layer is barely used. Both need ablations before any claim.
