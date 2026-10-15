# v8 — Two-Section Brain: English Module + Growing Knowledge Module

v8 keeps everything from v7 (always learning, strength/resistance/fade, fast + slow
memory, sleep, replay, growth) and adds:

1. **A separate English (language) section** trained only in `en-train` mode.
2. **A knowledge section** for everything else, with **topic experts (MoE)** that grow.
3. **Dictionary lookup**: new words are looked up and learned, only until they are known.
4. **Long context up to 30,000 bytes**, using cheap attention (local window + compressed
   long-range memory, the idea behind DeepSeek's sparse attention) and a **KV cache**.
5. **Growth up to 30 layers** in total.
6. Modern training details: **QK-norm**, **learning-rate warmup** (also after each growth),
   **gradient checkpointing** (needed for long context on a 4 GB GPU).

## 1. Architecture

**Gated side-branch split** (changed 2026-10-15, after tests showed the stacked version let
knowledge bend English; combines v6's "English makes the output" with v8's learned knowledge):

```
bytes ──► Embedding (shared with the output head)
             │
   ┌─────────▼──────────┐
   │ ENGLISH lower part │  lang.0 .. lang.(know_tap-1)        (know_tap = 2)
   └─────────┬──────────┘
             ├──────────────────────────┐
             │                ┌─────────▼──────────┐
             │                │ KNOWLEDGE BRANCH   │  reads English understanding
             │                │ attention + topic  │  2 layers, grows until total = 30
             │                │ experts (MoE)      │  experts split up to 8 per layer
             │                └─────────┬──────────┘
             │◄── + gate × knowledge ───┘   gate = tanh(alpha) × sigmoid(w·x)
   ┌─────────▼──────────┐                   alpha starts at 0 -> knowledge adds nothing at first
   │ ENGLISH upper part │  lang.(know_tap) .. ; new English layers are added here
   │ (final say on form)│
   └─────────┬──────────┘
          RMSNorm ──► output (256 bytes)
```

- **English makes the output.** Knowledge only adds through the gate.
- **The gate is trained to stay ~0 on plain English lessons** (`gate_penalty`), so knowledge
  only "speaks" where it adds something (e.g. "An apple is" → "a red fruit").
- **English is not frozen.** It keeps learning: fully in `en-train`, gently from dictionary
  lessons (`dict_lr_scale` 0.3), and slowly from normal text in feed/chat
  (`lang_lr_normal` 0.1), always with revision, strength/resistance/fade and sleep.

Every layer: RMSNorm → attention (GQA, RoPE, QK-norm) → RMSNorm → feed-forward
(+ the fast-memory path from v7). Every 2nd layer is a **global** layer that also
sees the whole long context in compressed form.

### Size plan

| Stage | English layers | Knowledge layers | Total | d_model 256 params |
|---|---|---|---|---|
| Start | 4 | 2 | 6 | ~4.6M |
| Mid | 8 | 10 | 18 | ~14M + experts |
| Max | 8 | 22 | **30** | ~23M + experts |

Width stays at `d_model` chosen at start (256 default; 384 or 512 also fit on 4 GB).

## 2. English section and `en-train`

- `python3 run.py en-train` trains **only** the English section (+ the shared embedding).
  The knowledge section is frozen during it.
- Curriculum: built-in classes 1–5 (letters, words, counting, math words, punctuation,
  grammar), then any **class files** in `lessons/english/` in name order
  (`class06_*.txt`, `class07_*.txt`, … up to class 12 and university level).
  You write or collect these files; the code reads them.
- `python3 run.py en-train some_grammar_book.txt` trains the English section on a file.
- Normal learning (`feed`, `chat`) trains the **knowledge branch** fully and the **English
  section slowly** (`lang_lr_normal = 0.1`).
- `en-train` runs the English section **standalone** (knowledge branch skipped), so the
  English section is a complete language model by itself.
- **English anchor** during `feed`/`chat`: 25% of each step is English lessons (half
  already-learned English lessons, half grammar). English practises them, and the knowledge
  gate is pushed to 0 on them (`gate_penalty`). `feed` stops only when the file is learned
  **and** the gate on English lessons is below `english_gap_ok` (0.05).
- (Replaced: the earlier KL "teacher" anchor, see section 10.)

### Class file format

```
# comment
A lesson. Blank line between lessons.

Another lesson.

?? The opposite of early is => late
?? Past tense of go is => went
```

Lines starting with `??` are quiz items (`prompt => answer`). If a class file has no quiz
items, the quiz asks the model to continue random lessons.

## 3. Dictionary (refer only when needed)

- Give a dictionary file once: `--dict lessons/dictionary.txt` (format `word: definition`
  per line). It is remembered in the checkpoint.
- The model keeps a **count of how many times it has seen each word**.
- When learning text with a word it has seen **fewer than 5 times**, and the word is in
  the dictionary, it **looks it up** and learns `word means definition` in the
  **English section**.
- Each word is looked up at most 20 times. Once known, it **stops using the dictionary**
  for it, like a person.
- In `chat`, unknown words in your message get their definition placed in front of the
  prompt, so it can answer using it.
- It never reads the whole dictionary; only the words it meets.

## 4. Topic experts (MoE) in the knowledge section

- Each knowledge layer has a small router and 1–8 expert feed-forward networks.
- The router reads the **running average of the text so far** (a "topic" signal), so a
  story about animals keeps using the same expert and a math text uses another.
  It is causal: it never looks at future letters.
- Top-1 routing: only one expert runs per byte, so compute stays the same as one dense layer.
- **Expert growth**: when knowledge learning stalls, the busiest expert is **split** into
  two copies (output unchanged at the moment of the split); they then specialise.
  When all layers have 8 experts, a new layer is added (until 30 layers).
- A small balance loss stops one expert from taking everything.

## 5. Long context: 30,000 bytes on a 4 GB GPU

Normal attention costs length × length: 30,000² = 900M comparisons per layer. v8 uses:

| Branch | What it sees | Cost |
|---|---|---|
| **Local window** (all layers) | the current 512-byte block + the previous block | length × 1,024 |
| **Compressed memory** (global layers) | one summary per 64-byte chunk of everything older | length × length/64 |

So a byte at position 29,000 sees the last ~1,000 bytes exactly, and the whole earlier
text as ~450 summaries. This is the "compress + local window" part of DeepSeek's Native
Sparse Attention idea. (DeepSeek's fine-grained top-k selection is in the roadmap.)

- **KV cache** for generation: the prompt is processed once; each new byte only
  computes its own keys/values. Memory at 30k bytes, 30 layers: ~0.5 GB.
- **Training window** `train_ctx` defaults to 1,024 bytes for normal lessons. Use
  `feed --doc` to learn whole long documents in windows up to `max_ctx` (30,000).
  Long windows use gradient checkpointing and batch size 1.

## 6. Brain-like learning (from v7, now per section)

- Strength / resistance / fade on every trainable slow weight. A frozen section does not
  fade (it is not being changed at all).
- Fast memory per layer, decays fast; shrinks after sleep.
- Episode buffer tags each lesson as `en` or `know`; replay and sleep revise each section
  with its own lessons.
- Growth: English section adds layers (to 8); knowledge section splits experts, then
  adds layers (to 30 total). After every growth the learning rate warms up again.

## 7. Commands

```bash
cd "scratch_pad/language learner/v8"
PY=~/Documents/own_projects/ai_project/venv/bin/python

$PY run.py en-train                         # English curriculum (classes 1-5 + lessons/english/)
$PY run.py en-train grammar.txt             # train English section on a file
$PY run.py feed lessons/simple_world.txt    # knowledge section, until learned
$PY run.py feed story.txt --doc             # learn a long document with long context
$PY run.py chat                             # always learning; uses dictionary for new words
$PY run.py quiz
$PY run.py info
# add --dict lessons/dictionary.txt once to give it a dictionary
```

## 8. Honest limits

- 30 layers × 256 is still a **small** model (~25M + experts). Long context means it *can*
  see 30,000 bytes; using that well needs much more data and training.
- Training on 30,000-byte windows on a 4 GB laptop GPU is **slow** (seconds per step),
  even with checkpointing. Most learning should use 1,024-byte windows; use `--doc` for
  long-text practice.
- Expert offloading to CPU RAM / disk (to go beyond GPU memory) is **not in v8**: all
  experts stay on the GPU, which is fine up to ~8 experts × 22 layers at d_model 256.
  Planned for v8.1.
- Topic routing is learned; experts may not split into clean human topics.
- Freezing the English section in normal mode protects it completely, but it also means
  English only improves through `en-train` and dictionary lessons. Set
  `lang_lr_normal > 0` in `config.py` to let normal learning improve it slowly.
- Classes 6–12 and university English are **file-based**: the code is ready, the lesson
  files have to be written or collected.

## 9. Comparison with big models (architecture only)

Anthropic does not publish Claude's architecture; this compares against public
frontier-class open models (Llama 3, DeepSeek-V3, Qwen 2.5).

| Item | Big models | v8 | Status |
|---|---|---|---|
| Decoder-only, RoPE, RMSNorm, SwiGLU, GQA | yes | yes | ✅ same |
| KV cache | yes | yes | ✅ done, tested |
| Mixture of experts | many | topic-routed, top-1, split-to-grow | ✅ done / ❌ no CPU-disk offload |
| Long context | 128k–1M tokens | 30,000 bytes (local window + compressed memory) | ✅ done / ❌ no top-k selection (DSA) |
| QK-norm, warmup | yes | yes | ✅ done |
| Gradient checkpointing | yes | yes (texts > 2,048 bytes) | ✅ done |
| LR decay schedule | yes | no — on purpose (never-ending learning has no end point) | — |
| z-loss | some | no | ❌ |
| MLA (KV-cache compression) | DeepSeek | no — cache already small (~0.5 GB at 30k) | ❌ not needed yet |
| Size | 100B+ weights | grows to 30 layers × d_model (256) | ⚠️ depth only, no width growth |
| Tokenizer | BPE 100k–200k | bytes — on purpose (learn letters first, dictionary) | — |
| Instruction / RLHF stage | yes | no | ❌ v9 |
| Data, hardware | trillions of tokens, 1000s of GPUs | small files, one 4 GB laptop GPU | not code |

## 10. Status and test results (2026-10-15)

All tests ran on a fresh test model (d_model 256) on the RTX 2050, saved in a temp folder.

### Achieved ✅

| Feature | Test | Result |
|---|---|---|
| Two sections | params changed by one step | `en` step: only `emb`, `lang.*`, final `norm`. `know` step: only `know.*` (after fixing fast-memory fade leak) |
| English curriculum | `en-train` from zero | classes 1–6 all passed (c1 100%, c2 85–98%, c3 80–92%, c4 100%, c5 85–98%, c6 100%) in ~6 min |
| English growth | during `en-train` | grew by itself 4 → 5–7 English layers when learning stalled |
| Identity growth | output before/after | `grow_lang` 0.0, `grow_know` 0.0, `split_expert` 3.6e-7 difference |
| KV cache | cached vs full forward | max difference 6e-7 (same result) |
| 30,000-byte context | one training step | 1.2 s, 2.1 GB peak GPU memory |
| Long-prompt generation | 50 bytes after a 24k prompt | 0.9 s |
| Speed | short lessons | 68 ms/step (was ~400 ms before the short-text fast path) |
| Class files | `lessons/english/class06_sample.txt` | loaded and passed after quiz items were also taught |
| Dictionary | chat with `--dict` | unknown words (elephant, giraffe) were looked up and shown |
| Knowledge learning | `feed lessons/simple_world.txt` | learns the facts: "An apple is a red fruit", "A cow gives milk", "The sky is blue", "A fish lives in water" |

### Not solved yet ❌ — English drops after `feed`

English quiz scores (c1–c6) after feeding `simple_world.txt`, starting from the same model
(before feed: 100 / 85 / 83 / 100 / 97 / 100):

| Version | After feed | Notes |
|---|---|---|
| first v8 | 4–100 (c2 4%) | knowledge layers + dictionary steps without revision |
| + English lessons mixed in, dictionary revises | 26–49 | still large drop |
| + English teacher (KL, weight 1), ran 500 epochs | **98–100** ✅ | fine — but only after a very long run |
| + teacher, stop when lesson loss < 0.3 (91 epochs) | 26–84 | stopped while English was still damaged |
| + stop also needs English gap < 0.05 (124 epochs) | 26–87 | gap reached 0.05 but English still low |
| + teacher weight 5 (126 epochs) | 16–86 | gap 0.018, English still low |

Diagnosis so far:
- Swapping the pre-feed English section back in did **not** restore scores (first runs) →
  the knowledge layers do bend English output.
- With teacher weight 5, the **English section alone also dropped** (c5 97% → 23%) and its
  weights changed (max change 0.087). During `feed` only **dictionary lookup steps** and
  **sleep** update the English section, so one of them is the suspect.
- The "English gap" measured on the few anchor lessons is low while quizzes are low →
  the gap does not yet measure English quality well.

### Change made after these tests (code written, NOT yet tested)

- Split changed from **stacked** to **gated side branch** (see section 1): English makes the
  output, knowledge adds through a gate that starts closed and is kept closed on plain English.
- KL teacher anchor removed; replaced by the gate penalty.
- English section no longer frozen in feed/chat (`lang_lr_normal` 0.1).
- Dictionary lessons learn gently (`dict_lr_scale` 0.3) with 3× English revision.
- Old v8 test checkpoints are not compatible (new gate weights).

Next steps:
0. Test the gated split: en-train → quiz → feed simple_world → quiz (compare with table above).
1. Feed with the dictionary off to confirm whether dictionary steps cause the English-section drop.
2. If yes: make dictionary lessons smaller and safer (more revision, lower lr, or keep them
   in the fast memory until sleep).
3. Measure the English gap on a fixed set of English lessons instead of random anchor samples.
4. Re-decide `anchor_kl` (currently 5.0, set by the last experiment; 1.0 was the original).

### Not built yet (planned)

- v8.1: expert offload to CPU/disk, DeepSeek-style top-k block selection
- v8.2: width growth
- v9: instruction / question-answer stage
- Classes 7–12 and university English lesson files (code ready, files not written)
- z-loss

## 11. Master–slave linkage: design decisions for the next version (not implemented)

Agreed 2026-10-15. Design notes only, no code yet.

### Principle

**English is the heart of the model.** The model learns by reading and expresses itself by
writing, and both happen through language. So the English section is the **master** and the
knowledge section is the **slave**: every read, write and update of knowledge goes through
the master.

### The problem v6 hit

The slave stores knowledge in the master's internal numbers (representations).

- If the master is **frozen**, the slave's stored knowledge stays findable, but English
  can never improve.
- If the master **learns freely**, English improves, but the slave's knowledge is filed under
  the master's *old* representations and can no longer be found ("the librarian keeps
  renaming the shelves").

The goal: **the master keeps learning without breaking the slave's filing system.**

### Four tools

1. **One stable doorway.** The slave connects to the master at a single point (`know_tap`).
   Master layers *below* the doorway change slowly (high resistance). Upper layers and newly
   grown layers do most of the learning. Human version: basics are firmly set, style and
   vocabulary keep improving on top.
2. **Learning flows through the master** (already in v8's gated split).
   - Reading: the master turns text into understanding; the slave reads that.
   - Writing: the slave's addition goes back through the master's upper layers, which turn
     it into proper words.
   - Learning signal: the slave is corrected only through the master's output, so it only
     learns what the master can express.
3. **Doorway check.** When the master learns, it also checks that its understanding of old
   text *at the doorway* has barely changed (like the KL teacher check, but on the doorway's
   internal numbers instead of the output). New words and grammar get learned, but "apple"
   still points to the same shelf.
4. **Re-reading old notes during sleep.** Episodes are kept as text. When the master has
   changed a lot, sleep makes the slave re-read old notes with the improved English and
   re-file them under the current representations. Human version: re-reading school notes
   years later and understanding them better. This removes any need to freeze the master.

### New words and the dictionary

- Byte-level input means a new word never needs a new slot; any word is letters the model
  can already read.
- Learning a new word = the master learns `word means definition`, with revision, so it
  understands the word in any later sentence.
- New vocabulary keeps appearing in the world (even experts miss words), so lookup happens
  for life, only when a word is unknown, never reading the whole dictionary.
- Because the slave reads the master's understanding, **every word the master learns
  automatically helps the slave**: science, newspapers, any subject.

### Subject reading and English

- While reading other subjects, English learns only from **sentences made of words it
  already knows** (pure language practice, so it gets stronger).
- **New words and unusual patterns** go to the dictionary and the knowledge section, not
  blindly into English.

### Decisions

| Decision | Choice |
|---|---|
| Master (English) | Never frozen. Lower layers change slowly; upper and new layers learn freely |
| Slave (knowledge) | Learns only through the master (gated split, already in v8) |
| Linkage | One doorway (`know_tap`), kept steady by the doorway check |
| Master changed a lot | Sleep re-reads old notes and re-files them |
| New words | Dictionary lookup → master learns → slave benefits automatically |
| Subject reading | English learns only from known-word sentences |
| Emotion | Simple word-strength list (strong / destructive words → higher importance, slower decay). Later |
| Short-term / long-term memory | Later version (still an open idea) |
| Exact recall vs own words | Early classes: exact recall. Later classes: same meaning in different words |

## 12. Roadmap

- v8.1: expert offload to CPU/disk, DeepSeek-style top-k block selection in global layers
- v8.2: width growth (256 → 384 → 512)
- v9: instruction / question-answer stage so it replies instead of only continuing
