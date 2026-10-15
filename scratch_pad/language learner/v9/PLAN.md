# v9 — English Master, Knowledge Slave

v9 rewrites v8 to follow the master–slave strategy agreed in `v8/PLAN.md` section 11.
Everything else from v8 stays: byte-level input, always learning, curriculum, growth,
strength/resistance/fade, fast memory, sleep, replay, topic experts, 30,000-byte context,
KV cache, QK-norm, warmup, gradient checkpointing.

## Principle

**English is the heart of the model.** The model learns by reading and expresses itself by
writing, both through language. The English section is the **master**; the knowledge
section is the **slave**. Every read, write and update of knowledge goes through the master.

## Architecture

```
bytes ──► emb ──► English LOWER layers (basics)          master, below the doorway
                        │
                   [ DOORWAY ]  (know_tap = 2) ──────────────┐
                        │                          ┌─────────▼──────────┐
                        │                          │ KNOWLEDGE (slave)  │ topic experts,
                        │                          │ reads the master's │ grows to 30 layers
                        │                          │ understanding      │
                        │◄──── + gate × knowledge ─┴────────────────────┘
                        │      gate = tanh(alpha) × sigmoid(w·x), alpha starts at 0
                English UPPER layers (style, vocabulary)  master, above the doorway
                        │      new English layers are added here
                     output (next byte)
```

## What learns when

| Situation | Master (English) | Slave (knowledge) |
|---|---|---|
| `en-train` (classes, grammar files) | ✅ learns, standalone (knowledge skipped) | frozen |
| `feed` / `chat` (any subject) | frozen during the knowledge step | ✅ learns, reading through the master |
| … then, sentences made only of **known words** | ✅ gentle practice (0.3×) | — |
| … then, **unknown words** in the dictionary | ✅ learns `word means …` gently (0.3×) | — |
| New words / unusual patterns in subject text | ❌ never trains English blindly | ✅ stored as knowledge |
| Sleep | revises English lessons | **re-reads old notes** with current English |

## The four linkage tools

1. **One stable doorway** (`know_tap`). Once the slave holds knowledge, the master's basics
   below the doorway learn slowly (`low_lr_scale` 0.2) and resist more (`low_resist_mult` 3).
   Before any knowledge exists they learn freely, like early school.
2. **Learning flows through the master.** The slave reads the master's understanding at the
   doorway and writes back through the master's upper layers, which produce the words. The
   slave is corrected only through the master's output.
3. **Doorway check.** Every English step also checks old English text at the doorway against a
   frozen reference copy of the basics (`doorway_weight`). New words and grammar get learned,
   but old understanding stays steady, so the slave's knowledge stays findable.
4. **Re-reading old notes during sleep.** Notes are kept as text. In sleep the slave re-reads
   them with the current English. If the doorway drifted more than `refile_drift`, it re-reads
   longer (`refile_steps`). Then the reference copy is refreshed: knowledge is now filed under
   today's English. The master never needs to be frozen.

## Gate on plain English

During knowledge steps, 25% of the batch is English lessons (half learned English notes, half
grammar). On them the gate is pushed to 0 (`gate_penalty`), so knowledge only adds where it
has something to say. `feed` stops only when the file is learned **and** the gate on English is
below `english_gate_ok`.

## Dictionary and new words

- Byte-level input: a new word never needs a new slot; any word is letters it can read.
- Unknown word (seen < 5 times) with a dictionary entry → the master learns
  `word means definition`, with 3× English revision and the doorway check.
- Lookups happen for life, only for unknown words, at most 20 times per word.
- In chat, meanings of unknown words are placed in front of the prompt.

## Importance (simple stand-in for emotion)

- `lessons/strong_words.txt`: strong / destructive / emotional words (danger, fire, love,
  fear, …). Each one in a text adds 0.5 importance (max 3).
- `/important <text>` in chat gives maximum importance.
- Importance multiplies how often a note is revised and how long it survives before being
  forgotten. Real emotion is not modelled.

## Exact recall, then own words

Class-file quiz lines:

```
?? Past tense of go is => went                       exact recall (early classes)
?? A very big animal is => an elephant | huge        any listed form is right (later classes)
```

## Commands

```bash
cd "scratch_pad/language learner/v9"
PY=~/Documents/own_projects/ai_project/venv/bin/python

$PY run.py en-train                         # English curriculum (master)
$PY run.py en-train grammar.txt             # master learns from a file
$PY run.py feed lessons/simple_world.txt    # any subject (slave + English practice + dictionary)
$PY run.py chat --dict lessons/dictionary_sample.txt
$PY run.py quiz
$PY run.py info                             # shows doorway drift
```

Chat: `/teach <text>`, `/important <text>`, `/sleep`, `/save`, `/quit`.

## Not in v9 (later)

- Separate short-term and long-term knowledge stores (open idea).
- Real emotion (only a strong-word list).
- Expert offload to CPU/disk, top-k block selection, width growth.
- Question-answer (instruction) stage.

## Status

Code written 2026-10-15. Not yet trained or evaluated. The first test to run:
`en-train` → `quiz` → `feed lessons/simple_world.txt` → `quiz`, and compare English scores
before and after feeding with the v8 results table (`v8/PLAN.md` section 10).
