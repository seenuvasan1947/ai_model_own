# TITAN_APPROACH: a Mini-Titans Neural Memory

Reference: Behrouz, Zhong, Mirrokni, *Titans: Learning to Memorize at Test Time* (arXiv 2501.00663), sec. 3.1. Source PDF: `od/2501.00663v1.pdf`.

## Idea

The memory is a small neural network, and its **weights are the memory**. The weights are updated at test time by gradient descent, with no optimizer loop and no training phase. The rules, taken from the paper:

| Rule | Equation | In `neural_memory.py` |
|---|---|---|
| Associative loss | `‖M(k) − v‖²`, `k = xW_K`, `v = xW_V` | `update()` |
| Surprise = gradient of that loss | Eq. 8 | `torch.autograd.grad(...)` |
| Momentum over past surprise | `S_t = η S_{t-1} − θ ∇ℓ` (Eq. 10) | `self.S` |
| Forgetting = weight decay | `M_t = (1−α) M_{t-1} + S_t` (Eq. 13) | `alpha` |
| Recall = forward pass, no update | `y = M(W_Q x)` (Eq. 15) | `retrieve()` |

Persistence is simple: the memory is weights, so `save()` writes them to a `.pt` file (about 179 KB for the default size).

## How it differs from `brainlm` (the sentence-store agent)

| | `brainlm` | `titan_way` |
|---|---|---|
| Memory | Sentences + vectors in a list | MLP weights |
| Write | Append or reinforce if novel | Gradient step (learning) |
| Size | Grows with input | Fixed |
| Recall | Nearest-neighbour lookup | Forward pass |
| Format | `keys.npy` + `meta.json` | `memory.pt` |

## What we did NOT copy from the paper

- The attention branch, persistent-memory tokens, and the MAC/MAG/MAL wiring. Only the neural memory (LMM) is built.
- **The outer loop.** In the paper `W_K`, `W_V`, `W_Q` and the gates `α, θ, η` are learned by large-scale pretraining. Here they are fixed random projections and fixed constants.
- `W_Q` is tied to `W_K`, and the values `v` are L2-normalised as well as `k` and `q`. Without normalising `v` the updates diverged to NaN in our first run.
- The 1-D convolution and the data-dependent gates.

## Run

```bash
cd titan_way
../.venv/bin/python test_small.py    # synthetic vectors: learning, capacity, ablations, forgetting, persistence
../.venv/bin/python test_text.py     # real text: question -> answer facts, paraphrase recall, keep learning
```

Use it from Python:
```python
from neural_memory import NeuralMemory
mem = NeuralMemory(in_dim=512, dim=64, depth=1)
mem.update(x_question, x_answer)     # learn (test time, one gradient step)
mem.surprise(x)                      # how new is this?
mem.retrieve(x_question)             # recall a vector (forward pass)
mem.save("memory.pt"); mem.load("memory.pt")
```
Recall returns a **vector**, not text. To get words back you compare it with the stored answer encodings (as `test_text.py` does). There is no generative decoder yet.

## Results (small scale, CPU, seed 0, one run each)

**Can it learn at test time? Yes.**

| Test | Result |
|---|---|
| 20 random items, before learning | 0.00 recall (chance 0.05) |
| After seeing each once | 0.90 |
| After 5 passes | 1.00 |
| Surprise: already-seen vs new items | 0.05 vs 0.98 |
| Save to disk, load into a fresh memory | 1.00 → 1.00 |
| 10 text facts, exact questions | 1.00 (untrained: 0.20) |
| 10 text facts, paraphrased questions ("what is the …?") | 1.00 |
| Then 10 more facts, no training phase: old / new / all | 0.80 / 1.00 / 0.85 |

**Capacity** (dim 64, 5 passes; recall = closest of all N stored values is the right one):

| N stored | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|
| depth 2 | 1.00 | 1.00 | 0.98 | 0.74 | 0.47 |

**Forgetting** (learn A, then stream B; 20 items each):

| Setting | A after A | A after B | B |
|---|---|---|---|
| no decay | 1.00 | 1.00 | 1.00 |
| α = 0.01 | 1.00 | 0.85 | 1.00 |
| α = 0.05 | 0.55 | 0.15 | 0.15 |

**Ablations (N = 50): mixed evidence**

- Removing momentum lowers recall (0.98 → 0.78), which matches the paper.
- Removing decay did not hurt here (1.00 vs 0.98), because at N = 50 the memory isn't full yet. Decay only shows a cost.
- **The paper's "deeper memory is better" claim did NOT reproduce.** A linear memory (depth 1) beat the MLPs at higher load:

| N | depth 1 | depth 2 | depth 3 (θ = 0.03) |
|---|---|---|---|
| 100 | 0.99 | 0.78 | 0.80 |
| 200 | 0.95 | 0.51 | 0.85 |
| 400 | 0.91 | 0.23 | 0.64 |

  Depth 3 with a smaller step size did much better than depth 2 with θ = 0.1, so this is likely hyperparameter sensitivity rather than a real verdict on depth. We did not tune per depth.

## Honest reading

- The core mechanism works at small scale: a network with no training phase learned associations from single exposures, recalled them from paraphrases, kept learning new facts, and survived save/load.
- These are toy tests: random vectors and 20 text facts. Recall is scored against a list of known answers, which is lenient. It does not show that the memory can produce answers from scratch.
- Stability depends heavily on `θ` and `η`. Values of `θ` of 0.3 or more with momentum diverged in a sweep.
- Decay is a real tradeoff. `α = 0.01` forgot 15% of A after B, and `α = 0.05` forgot most of it, while no decay forgot nothing at this small load. Under overload decay is what makes room for new items.
- One seed, one run. No error bars.

## Next steps

1. Sweep `θ, η, α` per depth, and repeat over several seeds, before drawing conclusions about depth.
2. Harder text: many more facts, paraphrases with different words (needs a better encoder than hashing).
3. Learn the projections `W_K, W_V, W_Q` and the gates in an outer loop (the paper's meta-learning), and compare with fixed random ones.
4. Add a short-term attention or GRU branch, then a small decoder so the model can generate answers.
5. Compare head to head with `brainlm` on the same facts (accuracy, size, forgetting).

## Chat (`chat.py`)

```bash
cd titan_way
../.venv/bin/python chat.py            # state is kept in ./titan_state/
../.venv/bin/python chat.py --state my_brain
```

| Input | What it does |
|---|---|
| `capital of Italy => Rome` | Teach a question => answer pair |
| `Mitochondria produce energy for the cell.` | Teach a plain sentence (stored as sentence -> itself) |
| `What is the capital of Italy?` | Ask (anything ending in `?`) |
| `/wrong <right answer>` | Last answer was wrong: relearn with extra updates |
| `/stats`, `/trace on\|off`, `/forget`, `/quit` | Info, show/hide internals, wipe memory, exit |

**What you can watch** (trace is on by default):
- On teaching: `surprise 1.71 -> 0.05 after 9 updates | weights moved 1.80`. The surprise is how new the fact was, and the weights moved is how much the memory changed.
- On asking: the top 3 candidate answers with cosine similarity. It answers only if the best is at least 0.55, otherwise "I don't know".

**Does it keep what it learned after a restart? Yes.** Every teach saves immediately, so even a crash or Ctrl-C keeps it.
- `titan_state/memory.pt`: the memory weights, plus the fixed projections (about 0.9 MB).
- `titan_state/texts.json`: the answer wordings, plus a small replay buffer of (question, answer) pairs.
- The weights hold no words. `texts.json` is what turns a recalled vector back into text, so it is plain text and readable.

**What we found while building it:**
- The first version answered "capital of Italy" with Tokyo. The questions share "capital of", so learning Japan overwrote Italy.
- Fix: keep updating a new fact until its surprise is under 0.05, and replay up to 16 older facts in the same batch (rehearsal). After that, all 5 test facts were recalled correctly, including after a restart.
- Cost: the replay buffer stores the old questions and answers as plain text, and each teach takes about 10 updates.
- Untested at scale. Five facts is a demo, not a benchmark. The capacity results above still apply.

## Full agent (`agent.py`, `decoder.py`, `chat.py`): PLAN_FUTURE items implemented

Run: `cd titan_way && ../.venv/bin/python chat.py` (default decoder: `decoder_v2.pt`).
Extra commands: `!question => answer` (important), `/sleep`, `/good`, `/wrong <right>`.

Pipeline for a question:
`recall (neural memory) -> candidate sets -> decoder drafts sentences (greedy + sampled) -> score = 0.6*support + 0.4*grounded - mistake penalty -> best wins -> if not grounded, show the stored text as-is`.

Files: `synth.py` (synthetic training data), `decoder.py` (GRU or windowed-transformer byte model), `train_decoder.py`, `agent.py`, `chat.py`, `eval_scale.py`. Logs: `train_*.log`, `eval_scale.log`. Weights: `decoder_v2.pt` (best), `decoder_transformer*.pt`, `decoder_gru.pt`.

### Decoder results (trained from scratch on CPU, synthetic data)

| Run | Backbone | Steps | Held-out exact match (unseen random names) | Real examples right (of 7) |
|---|---|---|---|---|
| 1 | GRU | 1500 | 0.10 | 1 |
| 2 | Transformer (local window) | 1500 | 0.61 | 2 |
| 3 | Transformer | 4500 | 0.86 (easy names) | 2 |
| 4 (v2, harder names) | Transformer | 6000 | 0.74 (harder names) | 3 |

- The GRU never learned to copy names from the prompt. The transformer did.
- Run 3's high score was on syllable-shaped names only, and it garbled real ones (India -> "Melehi"). Run 4 trained on arbitrary letters and two-word names and fixed India, but still garbles some ("Menergy", "Xwater").
- The decoder writes only sentence templates it was trained on (capital, creator, boiling point, "X is Y", copy a sentence, two capitals, "I don't know"). It is not a general language model.

### Evaluation vs baseline (3 seeds; "capital of X" -> Y; paraphrased test question)

| N facts | chance | plain retrieval | Titan memory |
|---|---|---|---|
| 10 | 0.10 | 1.00 | 1.00 |
| 25 | 0.04 | 1.00 | 1.00 |
| 50 | 0.02 | 1.00 | 0.97 |
| 100 | 0.01 | 1.00 | 0.78 (std 0.13) |

- Plain retrieval is at least as accurate at every size, so at this scale the neural memory is not better. Its one advantage is fixed size: 16,384 parameters however many facts it learns.
- Teaching gets slower with more facts (about 32 s for 100), because each fact trains until its surprise is low, replaying older ones.
- Rehearsal did not help measurably here: after 60 facts, 0.99 with replay vs 0.98 without (within noise).

### Verified behaviours (chat tests)

- Contradiction: teaching a different answer for the same question printed "I believed 'Rome' for this; replaced".
- `/wrong Rome` fixed the answer and stored the mistake; the mistake penalty then applies to that answer for similar questions.
- `/sleep` replays by salience and decays; low-salience unimportant pairs drop out of the rehearsal buffer.
- Restart: answers persisted (loaded 3 answers, 22 updates) and "capital of India" was still answered in a full sentence.

### Known problems

- The decoder still garbles some real names; the grounding check catches this and shows the stored text instead. The check is a heuristic (question words + stored answer must appear), not proof of correctness.
- A taught plain sentence is stored as sentence -> itself, so a short question can miss it ("tell me about Mitochondria?" -> "I don't know") because the hash encoder needs word overlap.
- "Imagination" only scores drafted answers; it doesn't predict real outcomes or look several steps ahead.
- Not built: Mamba backbone, emotion/survival salience, an automatic next-sentence error signal.

### Paragraph and file input

- Paste a sentence or a whole paragraph into `chat.py` (or use `/learn path/to/file.txt`, or `/learn <text>`). `agent.learn_text()` splits it into sentences (3+ words each) and teaches each as sentence -> itself, with the same surprise / rehearsal / salience as any fact.
- Questions are answered by returning the closest stored sentence as-is (the decoder is only used for short template answers).
- Tested on a 5-sentence paragraph about parliamentary systems: 3 of 5 paraphrased questions answered correctly and it said "I don't know" for an untaught question. Two questions ("what is a parliamentary system?", "which countries are parliamentary democracies?") missed, because the hash encoder needs word overlap with the stored sentence.
- Cost: about 8 updates per sentence, and it slows as more is learned, so very large files will take a long time.

### Encoder comparison: hash vs pretrained MiniLM (`compare_encoders.py`)

Run: `../.venv/bin/python compare_encoders.py [paragraph.txt]`; use it in chat with `chat.py --encoder minilm` (needs `pip install sentence-transformers`; downloads about 90 MB the first time; a memory saved with one encoder can't be opened with the other).

Test: the 7-sentence parliament paragraph, 5 questions that reuse its words, 7 that use different words, 3 unrelated ones that should be refused. One paragraph, one run, so treat it as an indication only.

| System | Same words (5) | Different words (7) | Refuses unrelated (3) |
|---|---|---|---|
| hash + neural memory | 4 | 1 | 3 |
| MiniLM + neural memory | 4 | 2 | 3 |
| MiniLM + plain nearest-neighbour (diagnostic, no neural memory) | 5 | 5 | needs a cosine threshold (correct matches scored 0.23-0.86, unrelated 0.09-0.20) |

- The pretrained embedder does understand meaning ("lawmakers force a leader out" is closest to the no-confidence sentence, "symbolic roles" to "ceremonial duties").
- The neural memory throws most of that away: it maps each question through a random 384 -> 128 projection and a small learned network, and its recall similarity often lands just under the 0.55 refusal cutoff (0.50-0.53) or on a wrong sentence.
- So with a good encoder, the weak link is the neural memory, not the embedder.
- The 0.23 vs 0.20 margin above is too thin to tune a threshold on 10 questions.

Suggested next step: a hybrid that keeps the rehearsal pairs as an embedding index (fast, exact recall, like the hippocampus) and uses the neural memory as the compact, generalising layer, then compares the two on a larger test.
