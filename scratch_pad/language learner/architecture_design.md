# Architecture Design: Modular Brain-Inspired Continual Language Model

This is the consolidated design after working through the reasoning on dimension-splitting, activation functions, and why structural (not post-hoc) separation is required. This supersedes the single-shared-vector version in `archive/prototype_v1/` and `prototype_spec.md`.

---

## 1. Core Correction This Design Is Built Around

**Wrong approach (rejected):** train one dense transformer using the full width (e.g. 384 dimensions), then slice the *already-trained, already-blended* output vector into 3 pieces (e.g. dims 0-127 / 128-255 / 256-383) and assign each piece to a different memory subsystem.

**Why it fails:** attention and feedforward layers deliberately mix all dimensions together on every forward pass (this is the entire point of attention — letting "river" influence "bank"'s meaning). Learned features are not stored in clean dimension ranges — they live in **overlapping combinations of many dimensions at once** (a real, documented phenomenon called *superposition*). Slicing at a fixed index cuts a distributed feature apart, destroying it rather than isolating it. A fragment of a distributed pattern is not a smaller version of the concept — it's usually meaningless.

**Correct approach (this design):** structural separation, decided *before* training, matching how the brain is actually organized. The optic nerve and auditory nerve never blend into one shared signal before reaching their respective cortices — separation happens from the start, not after the fact. This design builds three genuinely separate sub-networks with no forward-pass connectivity between them except at deliberate, designed integration points (the computational analog of the corpus callosum / arcuate fasciculus).

---

## 2. The Three Modules ("Pairs")

| Module | Role | Learning mechanism | Analogy |
|---|---|---|---|
| **Pair 1 — Language Foundation** | Deep grammar/meaning/structure, built once | Standard backprop (one-time or periodic full retrain) | Critical-period language acquisition (Broca's/Wernicke's) |
| **Pair 2 — Short-Term Memory** | Continuously updating, fast, recent-context-sensitive | Local Hebbian update (forward-pass only, no backward error signal) | Working memory / attention-driven short-term recall |
| **Pair 3 — Long-Term Memory** | Permanent, selective, consolidated knowledge | Sparse quantized key-value store, promotion via salience/repetition | Hippocampus → cortex consolidation |

Each module has **its own weight matrices and its own dedicated dimensions** — not a shared vector split by index. They are structurally separate computational units.

---

## 3. Structural Separation — The Actual Mechanism

- Each module is a **block** in a block-sparse / modular network: its weight matrix operates only on its own dimension range, with **zero connectivity to the other modules' dimensions in the forward pass** by default.
- Because there is no computational path between blocks, a module's dimensions genuinely cannot be influenced by another module's activity — this is what makes "untouched until its own learning phase" actually true, unlike the earlier flawed version (restricting *gradient updates* alone still let information leak through via the *forward* computation).
- **Integration points**: a small number of explicitly designed connecting weights between modules, added deliberately — analogous to the arcuate fasciculus connecting Broca's and Wernicke's, or the corpus callosum connecting hemispheres. These are the only channels through which, e.g., Pair 1's output can inform what Pair 2 stores, or Pair 3 can be queried to influence Pair 1's generation. They are not incidental — they're the one part of the design that must be deliberately specified, since nothing crosses module boundaries by accident.

---

## 4. Dimensions and Layers

| Setting | Value | Reasoning |
|---|---|---|
| Pair 1 embedding dimension | 384 | Matches nanoGPT-small convention; enough capacity for real grammar/meaning learning without requiring heavy compute |
| Pair 1 layers | 6 | Standard small-GPT depth |
| Pair 1 attention heads | 6 (64 dims/head) | Proven ratio; head-splitting is a *valid* form of splitting because it's learned end-to-end, unlike a hand-imposed post-hoc slice |
| Pair 1 context window | 512 tokens | Enough for sentence/paragraph coherence for a prototype |
| Pair 2 dimension | Operates on Pair 1's full 384-dim output vector (read/write), own separate weight structure | Short-term memory needs to interpret the foundation's full representation, not a truncated fragment of it |
| Pair 3 dimension | Same — reads full 384-dim vectors, stores them via quantized keys | Long-term store needs the complete representation to match/reinforce concepts correctly |

Note: Pair 2 and Pair 3 are not smaller slices of the 384 dimensions — they are separate mechanisms (Hebbian update rule; sparse key-value store) that each consume the *complete* 384-dim vector Pair 1 produces, then do their own kind of processing on it. The "split" is about mechanism/substrate, not about dividing the numbers.

---

## 5. Activation Functions

| Module | Activation | Why |
|---|---|---|
| Pair 1 (backprop-trained) | **GELU** (or SwiGLU) | Proven, well-tested; don't introduce a novel, untested variable into the one part of the system that most needs to just work |
| Pair 2 (Hebbian, forward-only) | None in the traditional sense — optionally `tanh` purely to bound update magnitude | Not a weighted-sum-then-nonlinearity computation; magnitude control matters more than a learned nonlinearity |
| Pair 3 (sparse key-value store) | None | Lookup + reinforcement, not a backprop layer — the concept doesn't apply |

Custom activation functions are a legitimate research option (Swish/SiLU was found via automated search, not derived from first principles) but are deliberately **not** used for Pair 1 here — reuse a proven option there and spend original-design effort on the genuinely novel parts (Pairs 2 and 3, and the integration points).

---

## 6. The Sparse Long-Term Store — Handling Continuous Vectors

Discrete-symbol Hebbian models (e.g. `archive/prototype_v1/`'s standalone Hebbian scripts) can use exact-match dictionary keys because words are discrete strings. Pair 1's output vectors are continuous and never exactly repeat, so a naive dictionary would never accumulate strength on anything.

**Fix: vector quantization.** Maintain a codebook of reference vectors. When a new vector arrives, snap it to its nearest codebook entry and use that entry's ID as the sparse key. Repeated close-but-not-identical vectors for the same concept then correctly accumulate on the same key.

**Known tradeoffs (not blockers, but real design work):**
- Codebook size is a new hyperparameter — too small merges distinct concepts, too large recreates the sparsity problem.
- Vectors near a boundary between two codebook entries can snap inconsistently.
- The codebook itself needs a non-backprop way to grow/adjust (e.g. k-means-style updates, or add a new entry when nothing is close enough) to stay consistent with Pair 2/3's no-backprop design.

---

## 7. What Decides Promotion to Long-Term (Pair 3)

Two triggers, both already validated as working code in `files/full_model.py`:
1. **Salience/surprise** — high loss or an emotional-salience score at the moment of exposure (the amygdala-tagging analogy). `consolidate()` already promotes on `loss > surprise_threshold`.
2. **Repetition from different angles** — a new exposure that's cosine-similar to an existing long-term entry reinforces that entry instead of creating a duplicate (`LongTermStore.write()`'s `sims[best] > 0.95` check). This is "the same concept seen again, differently phrased, strengthens the same memory."

---

## 8. What This Design Deliberately Does NOT Include Yet

- No multi-modal "masters" (vision/audio) — text-only, by design, to reduce scope (see `text_processing_brain_regions.md`).
- No sleep-like offline replay/reshuffling mechanism yet (a real brain mechanism for handling live-stream-order sensitivity) — flagged as a future addition, not yet designed.
- No three-factor (reward/surprise-modulated) upgrade to the Hebbian rule yet — currently plain co-activation, which is weaker than the brain's dopamine-modulated version. Flagged as the clearest concrete upgrade path.
- Router/module-selection logic between the 3 pairs is not yet fully specified — the integration points (Section 3) need concrete design before implementation.

---

## 9. Summary

Three structurally separate modules, each with its own weights, connected only through deliberately designed integration points — not one shared vector cut into pieces after training. Pair 1 (384-dim, 6-layer transformer, GELU, backprop) builds the language foundation once. Pairs 2 and 3 read Pair 1's full output and apply their own no-backprop mechanisms (local Hebbian update; sparse quantized long-term store) to keep learning continuously afterward, without needing to touch Pair 1's weights or fragment its representation.
