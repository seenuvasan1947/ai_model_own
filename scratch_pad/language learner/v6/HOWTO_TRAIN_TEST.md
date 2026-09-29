# v6 How-To: Train, Test, and Use

This is the step-by-step guide for actually running v6 end-to-end. Training takes hours (see time estimates below), so plan accordingly — this doc is written to be resumable at any stage.

**All commands assume you're in the v6 directory:**
```bash
cd "language learner/v6"
```

**All Python commands use the venv:**
```bash
./.venv/bin/python <script>
```

---

## Stage 0: One-Time Setup (5-10 minutes)

Install the extra dependencies v6 needs (the venv already has torch + tokenizers from earlier; v6 also needs the `datasets` library from HuggingFace to pull TinyStories and Simple Wikipedia).

```bash
./.venv/bin/pip install datasets
```

Verify GPU is available (dramatically faster for training):
```bash
./.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available())"
```
If it prints `cuda: True`, training will use GPU. If `False`, it falls back to CPU but expect training to take much longer (see time estimates below).

---

## Stage 1: Build the Training Corpus (30-90 minutes)

The corpus has 4 parts. Run each prep script; they're independent and can be re-run individually if one fails.

### 1a. Synthetic grammar/numbers/facts (fast, ~30 seconds)
Generates itself locally — no download.
```bash
./.venv/bin/python data_prep/gen_synthetic.py
# writes data/synthetic.txt (~100-500K tokens)
```

### 1a'. Expand Q&A behavior set (fast, ~5 seconds)
Also generates locally. Appends ~1000+ generated Q&A pairs on top of the
original hand-written ones in `data/qa_behavior.txt`, deliberately
excluding any that verbatim-match the held-out eval set.
```bash
./.venv/bin/python data_prep/expand_qa.py
# updates data/qa_behavior.txt in place
```

### 1b. TinyStories (download + subset, 15-30 minutes)
Public dataset, downloads via HuggingFace. Default subset is ~10M tokens (adjustable in the script).
```bash
./.venv/bin/python data_prep/download_tinystories.py
# writes data/tinystories.txt (~10M tokens by default)
```
**Disk usage:** ~1-2 GB for the full download cache, then only the subset text file stays.

### 1c. Simple English Wikipedia (download + clean, 15-30 minutes)
Same pattern — HuggingFace download, then cleaned + subsetted.
```bash
./.venv/bin/python data_prep/download_wikipedia.py
# writes data/simple_wiki.txt (~5M tokens by default)
```

### 1d. Combine into the final corpus
```bash
./.venv/bin/python data_prep/build_corpus.py
# writes data/foundation_corpus.txt (all four sources concatenated)
```

Verify the corpus size looks right:
```bash
wc -w data/foundation_corpus.txt
```
Expected: **~15-25 million words** total.

---

## Stage 2: Train the Tokenizer (5-10 minutes)

Trains BPE on the new corpus. Vocab size auto-adjusts based on corpus size (v4 fix — see `tokenizer.py`).

```bash
./.venv/bin/python tokenizer.py
# writes data/tokenizer.json
```

Expected output: `Trained tokenizer with vocab_size=~8000-12000`.

---

## Stage 3: Train Layer 1 (THE LONG STEP — 2-8 hours)

This is the substantive investment. Layer 1 becomes the English master; once done, it never needs training again.

```bash
./.venv/bin/python -u train_layer1.py 2>&1 | tee out/training.log
```

**Time estimates (rough, based on prior runs on this hardware):**
| Hardware | ~10M param model | ~15M param model |
|---|---|---|
| RTX 2050 GPU (when not throttled) | ~1.5-3 hours | ~2-4 hours |
| RTX 2050 GPU (throttled, as seen in v3/v4) | ~3-6 hours | ~4-8 hours |
| CPU only | Don't try. |

**Early stopping is on by default:** if val loss doesn't improve for 5 consecutive eval checkpoints, training stops automatically and saves the best checkpoint. So actual wall time may be less than the max above.

**How to tell it's going well:**
- Train loss dropping steadily (from ~9 at step 0 to ~3-4 by step 1000).
- Val loss dropping alongside train loss (this is the main thing to watch).
- **Val loss / train loss gap staying < 0.5** = generalizing, not memorizing. Big gap opening = overfitting starting — early stopping should kick in.

**How to tell it's going badly:**
- Val loss plateaus while train loss keeps dropping fast → overfitting (early stopping will catch this).
- Val loss actually rising over multiple checkpoints → done, early stopping will save the best earlier checkpoint.
- Loss `nan` or exploding → gradient issue; kill it (Ctrl+C), reduce `learning_rate` in `config.py`, retry.

**If it takes too long or you need to stop:** Ctrl+C is safe. The last checkpoint is saved every eval interval; you can resume by running again (it won't resume mid-training but the last saved checkpoint is intact).

---

## Stage 4: Sanity-Check Layer 1 On Its Own (2 minutes)

Before doing anything else, confirm Layer 1 speaks coherent English:

```bash
./.venv/bin/python ask.py --no-memory "What color is the sky?"
./.venv/bin/python ask.py --no-memory "How many days are in a week?"
./.venv/bin/python ask.py --no-memory "What is 2 plus 2?"
```

**Expected:** grammatical, on-topic answers (even if not always perfectly correct — this is a small model). If you get complete word-salad, something went wrong in training and it's worth rerunning with different hyperparameters before continuing.

`--no-memory` bypasses the memory stack (L2/L3) so this is a pure test of Layer 1 alone.

---

## Stage 5: Test Continual Learning (5 minutes)

The Zeta test, same as v5:

```bash
# Feed the new story into memory (Layer 2/3, no backprop)
./.venv/bin/python continual_read.py data/new_text.txt

# Ask a question that requires the new knowledge, both ways
./.venv/bin/python ask.py "Who is Zeta?"
./.venv/bin/python ask.py --no-memory "Who is Zeta?"
```

**Compare the two answers.** They should differ. With memory on, the answer should be more "Zeta-flavored" (contain relevant vocabulary from the story). Without memory, the answer will be more generic.

---

## Stage 6: Run the Two Eval Harnesses (10-20 minutes)

v6 has **two separate evals** because Layer 1 competence and continual
learning are two separate things and shouldn't share a test set.

### 6a. Language-competence eval — tests Layer 1 (memory OFF vs ON, no facts fed)

```bash
./.venv/bin/python eval_qa.py
```

**What it does:**
1. Loads `data/eval_qa.jsonl` — only questions Layer 1 should be able to
   answer from its training (arithmetic, opposites, day/month sequence, plurals).
   No world-facts here.
2. Runs each question through `ask.py` twice: with memory ON, without.
3. Scores by keyword overlap.

**Expected:** memory-ON and memory-OFF scores should be similar for this eval —
these questions don't require memory. The point of running with memory ON is
to check memory doesn't *hurt* on questions where it shouldn't help.

### 6b. Continual-learning eval — tests Layer 2/3 (feed a fact, then ask)

```bash
./.venv/bin/python eval_qa_continual.py
```

**What it does:**
1. Loads `data/eval_qa_continual.jsonl` — world-fact questions PAIRED with
   the fact needed to answer (e.g., fact: "The capital of France is Paris.",
   question: "What is the capital of France?").
2. For each item: resets memory to empty, feeds the fact through
   `stack.observe()` multiple times so it consolidates, then asks with memory
   ON and OFF. Fresh memory per question — no cross-contamination.
3. Scores + reports the memory delta.

**Expected:** memory-ON score should be measurably HIGHER than memory-OFF.
Since Layer 1 was never trained on these facts, memory-OFF should score near
zero. Any memory-ON gain is direct evidence continual learning works.

- **Big positive delta (>0.15)** = continual learning works well.
- **Small positive delta (0.05-0.15)** = works partially; memory is contributing
  but not dominant. Try tuning `stack_cfg.memory_feedback_strength` up.
- **Zero or negative delta** = something's wrong. Either the fact isn't
  reaching L3 (check L2/L3 unit counts in the output), or memory feedback
  is being drowned out by Layer 1's own generation preferences.

To add your own tests, edit either JSONL file — format is one JSON per line.
`eval_qa.jsonl` uses `{"question", "expected_keywords"}`;
`eval_qa_continual.jsonl` adds `"fact"`.

---

## Stage 7: Optional — Continual Learning Eval

If you want to also measure whether continual learning specifically improves answers about newly-read content:

```bash
# Reset memory
rm out/layer2_state.pt out/layer3_state.pt

# Baseline: eval BEFORE reading anything new
./.venv/bin/python eval_qa.py > out/eval_before.txt

# Feed new content
./.venv/bin/python continual_read.py data/new_text.txt

# Eval AFTER
./.venv/bin/python eval_qa.py > out/eval_after.txt

# Compare
diff out/eval_before.txt out/eval_after.txt
```

Questions about the new content should score higher after continual reading.

---

## Troubleshooting

**"CUDA out of memory" during training:** reduce `batch_size` in `config.py` (e.g., 8 → 4), or reduce `n_embd`/`n_layer`.

**Training is very slow on GPU:** check for thermal/power throttling with `nvidia-smi -q -d PERFORMANCE`. This was a real issue in v3/v4 on the RTX 2050. If throttled, reduce model size, take breaks between training runs, or use CPU (slower but not throttled the same way).

**HuggingFace download fails:** the `datasets` library sometimes has transient issues. Retry, or manually download the corpus to `data/tinystories.txt` / `data/simple_wiki.txt` from other sources — the format expected is one document per newline-separated block, no special formatting needed.

**Out-of-disk during corpus download:** TinyStories and Simple Wikipedia caches can be ~2-4 GB. Set `HF_DATASETS_CACHE=/some/path` before running the download scripts to control cache location. After building `data/foundation_corpus.txt`, the HuggingFace cache can be deleted.

**Model produces nonsense after training:** check `out/training.log` — look for the val loss trajectory. If val loss never dropped meaningfully below 6-7, training didn't work. Common causes: corpus too small (< 5M tokens), model too large for corpus, or `learning_rate` too high (try 1e-4 instead of 3e-4).

---

## What to Do When Layer 1 Is Trained

Once Stage 3 finishes and `out/layer1_final.pt` exists:
- **You never have to retrain Layer 1 again.** It's the frozen substrate for every future experiment.
- Any new domain knowledge → feed via `continual_read.py`, lives in `out/layer2_state.pt` / `out/layer3_state.pt`.
- Iterating on Layer 2/3 mechanics (v7+, if we ever get there) doesn't require retouching Layer 1.
- Back up `out/layer1_final.pt` and `data/tokenizer.json` somewhere safe — those two files together are the substrate.

---

## Total Time Budget (rough)

| Stage | Time |
|---|---|
| 0. Setup | 5-10 min |
| 1. Build corpus | 30-90 min (network-bound) |
| 2. Train tokenizer | 5-10 min |
| 3. Train Layer 1 | **2-8 hours** (main cost) |
| 4. Sanity check | 2 min |
| 5. Continual learning test | 5 min |
| 6. Q&A eval | 5-15 min |
| 7. Optional before/after eval | 15 min |
| **Total** | **3-10 hours** end to end |

The good news: after Stage 3, Layer 1 is done forever. Everything after Stage 3 is minutes, not hours.
