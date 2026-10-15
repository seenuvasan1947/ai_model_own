# v7 — Always-Learning, Growing, Brain-Like Language Model

## Goal

One generative model that:

1. Starts knowing nothing and learns **class by class**, like a school student:
   letters → digits → two-letter combos → words → counting → basic math →
   sentences → opposites / synonyms → ... → any topic written in English.
2. **Never stops learning.** No separate "train" and "test" modes and no frozen
   weights. Everything it reads, including questions at "test time", updates
   its weights.
3. **Generates text** from what it has learned (it is not a classifier).
4. **Grows** (adds layers) when it runs out of room.
5. Behaves like a brain:
   - repeated learning makes connections **stronger**
   - unused, weak connections slowly **fade**
   - strong (important) connections **resist fading and overwriting**
   - details fade but **general patterns stay** (abstraction), via a fast
     memory, a slow memory and "sleep" consolidation.

## Architecture

Decoder-only transformer (the same family as GPT and Llama). It predicts the
next byte, so generating text is how it works by default.

| Part | Choice | Why |
|---|---|---|
| Tokenizer | Byte-level (256 symbols) | Learns real letters first; vocabulary never has to change |
| Attention | Causal multi-head self-attention with grouped-query attention (GQA) | Modern standard; generates step by step |
| Positions | RoPE (rotary embeddings) | Modern standard |
| Feed-forward | SwiGLU | Used in Llama-class models |
| Normalization | RMSNorm, before each sub-layer | Stable training |
| Output head | Tied to the input embedding | Fewer weights |
| Sampling | Temperature + top-k | Lets it generate freely |

### Size plan (grows over time)

| Phase | Layers | Hidden size | Heads (kv) | ~Params | Fits RTX 2050 4 GB? |
|---|---|---|---|---|---|
| Classes 1-2 | 4 | 256 | 4 (2) | ~3M | ✅ |
| Classes 3-5 | 8 | 512 | 8 (4) | ~25M | ✅ |
| School English | 12 | 768 | 12 (4) | ~85-125M | ✅ just fits (bf16) |
| Strong English | 24 | 1024 | 16 (4) | ~350M | ❌ needs a bigger GPU |
| "Learn anything" | 16 | 2048 | 32 (8) | ~1B | ❌ needs 24 GB or more |

**v7 status:** depth growth is implemented. It adds a layer that starts as
identity, so nothing the model already knows changes. Width growth (256 → 512
→ ...) is planned for v8. For now, pick `d_model` at the start (`--d-model`).

## The brain-like parts

### 1. Strength per connection (`plasticity.py`)

Every slow weight has a **strength** value of the same shape.

- After each update: `strength = keep * strength + (1 - keep) * |grad × weight|`.
  A weight that keeps being useful (repetition) keeps getting stronger. A weight
  that stops being used slowly loses strength.
- Relative strength `s = strength / (strength + mean_strength)` is in [0, 1).

### 2. Resistance (strong memories are hard to overwrite)

Each update is scaled: `new = old + delta / (1 + resist * s)`.
Strong weights barely move; weak weights learn freely.

### 3. Fading (forgetting unused things)

Every step: `w *= 1 - fade * (1 - s)` (matrices only).
Weak, unused weights shrink toward zero. Strong ones almost never fade.
So things learned once and never again disappear; things learned many times
stay, even if they aren't used for a long time.

### 4. Fast memory + slow memory (`model.py`, `memory.py`)

```
FAST MEMORY (like the hippocampus)          SLOW MEMORY (like the cortex)
- small low-rank "fast weights" per layer   - the main transformer weights
- high learning rate, learns instantly      - low learning rate, with strength/fade
- decays quickly every step                 - lasts long
- episode buffer remembers what was read
   and how many times it was seen
              ↓ SLEEP (consolidation) ↓
   replay only episodes seen several times into slow memory
   episodes seen once and not seen again for a long time are dropped
   fast memory is shrunk
```

The abstraction comes from this: only repeated patterns reach slow memory, so
general patterns survive and one-time details fade.

### 5. Revision (replay)

Each normal learning step mixes in a small share (default 25%) of old episodes
from the buffer. This is the model's "revision" so specific skills (like math
tables) don't fade when the new text doesn't contain them. Letters and grammar
appear in all English text, so they get practice all the time anyway.

## The always-on loop (`learner.py`)

```
text comes in (lesson, file, chat message, question)
  1. LEARN   : predict the text, measure the error, update fast + slow weights
               (with replay, strength, resistance, fade)
  2. ANSWER  : if it was a question, generate a reply with the updated weights
  3. REMEMBER: store the episode and count repeats
  every N steps : SLEEP (consolidate, prune, shrink fast memory)
  when loss stops improving : GROW (add a layer)
repeat forever
```

## Curriculum (`curriculum.py`)

| Class | Learns | Example lessons | Example quiz |
|---|---|---|---|
| 1 | Letters, digits, order, capital/small | `after k comes l`, `big A small a` | `after k comes ` → `l` |
| 2 | Spelling, words, counting, number words | `c a t spells cat`, `7 is seven` | `after five comes ` → `six` |
| 3 | Add/subtract, simple sentences, opposites, synonyms | `4 + 5 = 9`, `the opposite of hot is cold` | `big means ` → `large` |

A class is **passed** when the quiz score is at least 90%. Then the next class
starts. Old classes keep being revised through replay. More classes (4-10)
get added as new generator functions in the same file.

After the classes, use `feed` (learn from any `.txt`) and `chat` (learns from
everything you type) to keep learning forever.

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings in one place |
| `model.py` | Transformer (RoPE, GQA, SwiGLU, RMSNorm) + fast memory + `grow_depth()` |
| `plasticity.py` | Strength, resistance, fading |
| `memory.py` | Episode buffer (hippocampus): repeat counts, sampling, pruning |
| `learner.py` | Always-on learn/answer loop, sleep, growth, checkpoints |
| `curriculum.py` | Classes 1-3 lessons + quizzes |
| `run.py` | Command line: `train`, `quiz`, `feed`, `chat`, `info` |

## How to run

```bash
cd "scratch_pad/language learner/v7"
PY=~/Documents/own_projects/ai_project/venv/bin/python

$PY run.py train                 # go through classes 1-3 (resumes from checkpoint)
$PY run.py quiz                  # test every class
$PY run.py feed mybook.txt       # learn any English text
$PY run.py chat                  # talk; it learns from every message
$PY run.py info                  # size, layers, step, current class
```

Inside `chat`: `/teach <text>` teaches without answering, `/sleep` runs
consolidation, `/save` saves, `/quit` exits.

Checkpoint: `checkpoints/v7.pt` (model, strengths, episode buffer, class, step).

## Honest limits

- Knowledge in a transformer is **spread across many weights**, so strength is
  tracked per weight, not per fact. This is approximate, like in the brain.
- Combining growth, strength/fade, fast/slow memory and sleep in one model is
  **research**, not a ready recipe. The values in `config.py` (fade, resist,
  replay, sleep) will need tuning by experiment.
- Learning from wrong input teaches wrong things. Chat learning uses a smaller
  learning rate than curriculum learning, and you can turn it off
  (`--no-learn`).
- Your RTX 2050 (4 GB) tops out around 12 layers × 768 hidden. Past that, move
  the checkpoint to a bigger GPU and it continues from the same state.

## Roadmap

- v7: base model, depth growth, strength/fade, fast/slow memory, sleep, classes 1-3, chat/feed
- v8: width growth, classes 4-6 (grammar, tenses, reading stories)
- v9: classes 7-10, open-text learning from large corpora
- later: Titans-style neural memory, move to a bigger GPU
