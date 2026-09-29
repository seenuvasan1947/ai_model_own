# v5 Design: Vertical Stack — Language Substrate + Memory Layers Stacked Below

v1-v4 built the memory system as **parallel** modules (Pair 1 || Pair 2 || Pair 3) that read from Pair 1 and get spliced back in. v4 confirmed that plumbing works, but also surfaced the fundamental integration problem: Pair 1 was never trained knowing memory vectors would appear in its attention positions, so it treats them as unfamiliar input rather than as a re-activation of what it already knows.

v5 restructures this into a **vertical stack**, matching how the brain actually integrates memory: not as an external database consulted alongside language processing, but as a substrate through which language processing itself flows. Memory isn't queried — it's activated by the same computation that language already runs.

---

## The Core Insight (from the user, verbatim reframed)

> The language module is the medium through which memory becomes usable, not a consumer of it. When we memorize English in school days — vocabulary, meaning, grammar — that becomes the substrate. Everything we then read, think, remember, and generate passes through that substrate. It's not one module among three. It's the base everything else builds on.

Practically: memory doesn't store "raw experiences." It stores linguistically-encoded traces, keyed and recalled through the same language machinery that encoded them. Recall isn't fetching — it's re-activating.

---

## The Architecture

```
     ┌─────────────────────────────────────────────┐
     │  Layer 1: Language Substrate (FROZEN)       │  ← Standard transformer, backprop-pretrained
     │  - Standard transformer weights             │     Never changes after pretraining
     │  - Provides embeddings, attention, output    │     (for v5 first cut; future work can allow
     │  - Hidden states are the shared substrate    │      slow adaptation)
     └────────────────────┬────────────────────────┘
                          │ (hidden states flow down; memory activity flows back up)
                          ▼
     ┌─────────────────────────────────────────────┐
     │  Layer 2: Short-Term Memory                 │  ← Sparse graph of connections
     │  - Fast strengthen (per exposure)           │     Fast decay (short retention window)
     │  - Fast decay if not activated              │     Forgets recent-but-uninteresting
     │  - Below-floor connections disappear         │
     └────────────────────┬────────────────────────┘
                          │ (repeatedly reinforced patterns propagate down)
                          ▼
     ┌─────────────────────────────────────────────┐
     │  Layer 3: Long-Term Memory                  │  ← Same sparse graph, slower rules
     │  - Slow strengthen (only from L2 patterns   │     Slow decay (near-permanent for
     │    that already got strong)                  │      strongly-reinforced items)
     │  - Slow decay                                │     Holds things learned deeply
     │  - Below-floor connections disappear         │
     └─────────────────────────────────────────────┘
```

---

## The Unified Mechanism

Every connection anywhere in the memory stack has just **one number: strength**. The three layers differ only in their rate parameters:

| Layer | strengthen rate | decay rate | source of activation |
|---|---|---|---|
| Layer 1 | 0 (frozen) | 0 (frozen) | pretrained backprop, then locked |
| Layer 2 | fast | fast | Layer 1's hidden states |
| Layer 3 | slow | slow | Layer 2's already-strong connections |

## The Update Rule (one rule, applied everywhere)

For every activated connection at each step:
```
strength ← strength × (1 - decay_rate) + strengthen_rate × activation
```

For every connection at each step (whether activated or not):
```
if strength < threshold:
    connection disappears (pruned)
```

That's the entire mechanism. Promotion from short-term to long-term, forgetting, consolidation — all emerge from these two lines applied with different rate constants per layer. Nothing is hand-coded as a separate "promote if X" rule the way Pair 3 did in v1-v4.

---

## Why This Is Better Than v1-v4's Parallel Design

1. **Unifies memory and learning into one mechanism.** Short-term and long-term are the same system with different timescales — matching the neuroscience of short-term potentiation vs. long-term potentiation (both Hebbian, only the time constants differ).
2. **Solves the integration problem structurally.** Memory activity flows back up through the same pathways Pair 1 uses, so there's no "foreign input" problem — recall re-activates the language substrate rather than injecting alien vectors into it.
3. **Promotion is emergent, not rule-based.** In v1-v4 we hand-coded thresholds ("promote if surprise > 3.0"). Here, connections that keep getting reinforced naturally accumulate strength past decay; ones that don't fade. No promotion logic to hand-tune.
4. **Strength IS memory, not a separate object.** No explicit table of vectors like v4's `LongTermStore`. Memory is the graph of connection strengths, and it changes automatically whenever anything is processed.
5. **Matches the user's original intent more faithfully than v1-v4 did.** The vertical stack directly implements "connections established only at learning time, and the language substrate is the base everything runs through."

---

## Concept Representation

**Not a table of vectors** (v4's approach — too static and disconnected). Instead:

- Each "concept" = a **cluster of co-activated units** in a layer.
- The "connection" between two concepts is the strength value between the units that make them up.
- When Layer 1 produces a hidden state, it activates a *pattern* across units in Layer 2. Whichever units co-fire together get their connecting strengths incremented. That's Hebbian, directly.

## Recall / Query

When Layer 1 is generating text and computes a hidden state:
1. That hidden state pattern activates the corresponding units in Layer 2.
2. Layer 2's activated units, through their existing connections, activate related units — this is *recall*.
3. If any of those recalled units have strong connections in Layer 3, those get activated too.
4. Combined activity in Layers 2 and 3 flows back up to Layer 1's residual stream, biasing generation toward what memory suggests.

Critical: **no separate retrieval step.** Recall is just activation spreading through a graph, the same way any neural computation works.

---

## Design Choices for v5

Two design questions with (A/B) options each. **v5 uses (B) for both** — after v1-v4 repeatedly chose the simpler path and failed to produce useful results, this version commits to the more principled design, accepting the added complexity.

### 1. How is a "unit" in Layer 2/3 defined?

- (A) Fixed-count units, dense connections. E.g., 512 pre-allocated slots, every unit can connect to every other. Simpler, but with a hard ceiling: once enough distinct patterns arrive, they overlap and interfere. Every new memory writes over pieces of old ones — catastrophic forgetting sneaking in through the back door. Rejected for that reason.
- **(B) Sparse, growing set of units.** Layer 2 starts empty. A genuinely novel pattern from Layer 1 creates a new unit. A pattern that's close to an existing unit's preference reinforces that existing unit instead. Units whose own activation has been zero for long enough eventually fade and free capacity. **← chosen for v5**

### 2. How does Layer 1's hidden state map to Layer 2's units?

- (A) Quantize (snap-to-one). Snap Layer 1's continuous vector to the nearest single codebook entry; activate only that one unit. Clean bookkeeping, but collapses "storm-with-fear" and "storm-with-awe" into the same slot — destroys the contextual nuance that Layer 1's hidden vector actually carries. Rejected for that reason.
- **(B) Distributed activation.** Layer 1's hidden vector activates many units in Layer 2 partially, each proportional to how well its preferred-vector matches Layer 1's output. Similar things share overlapping activation patterns naturally. Preserves the richness of what Layer 1 computed. **← chosen for v5**

## What (B)+(B) actually does, step by step

- Layer 2 starts with **zero units**. There's no fixed slot count.
- When a hidden vector `h` arrives from Layer 1:
  - For every existing unit `u` in Layer 2, compute `activation(u) = similarity(h, u.preferred_vector)`.
  - If the maximum activation across all existing units is **below `novelty_threshold`** → this is a genuinely new pattern → **create a new unit** whose preferred vector is `h`. Its activation is 1.0.
  - Otherwise → activate every existing unit `u` where `activation(u) > activity_floor`, proportional to `activation(u)`. Don't create anything new; existing units already cover this input.
- For every pair of units `(u_i, u_j)` that both activated on this step:
  - Increment their connection strength: `strength(i,j) ← strength(i,j) * (1 - L2_decay) + L2_strengthen * activation(i) * activation(j)`.
- Every connection in Layer 2 decays each step, whether it fired or not. Connections below `L2_floor` disappear (pruned).
- Every unit tracks its own recent activity. Units with zero activity for longer than `L2_unit_lifetime` fade and are removed — freeing capacity.

**Layer 3 works identically**, with slower rate constants (`L3_decay << L2_decay`, `L3_strengthen << L2_strengthen`, `L3_unit_lifetime >> L2_unit_lifetime`), and receives input **only from Layer 2's already-strong connections** — not from Layer 1 directly. This is what makes L3 "consolidated": only patterns that already got reinforced enough to survive in L2 ever propagate down to L3.

**Recall back up to Layer 1** works the same way in reverse. When Layer 1 is generating and produces a hidden state, that state activates units in Layer 2 (same distributed rule). Those activated units, through their existing connections, activate related units — this is associative recall. If any of those related units have strong connections in Layer 3, those get activated too. The combined activation pattern in Layers 2+3 is projected back into Layer 1's embedding space (each activated unit contributes a scaled version of its preferred vector) and added to Layer 1's residual stream — biasing generation toward what memory suggests, using Layer 1's own representation.

## Honest tradeoffs of (B)+(B) — named upfront, not after building

1. **More hyperparameters.** `novelty_threshold`, `activity_floor`, `L2_strengthen`, `L2_decay`, `L2_floor`, `L2_unit_lifetime`, and their L3 counterparts. Each needs tuning; wrong values could make the system either never form memories (threshold too high) or grow units indefinitely (threshold too low).
2. **Debug complexity.** Instead of "which unit fired?" the state is a distribution across many partial activations. Harder to reason about when things go wrong.
3. **Bigger implementation.** ~2-3x more code than the fixed-count/quantize version.
4. **No guarantee it works.** More principled than (A)+(A), but still an unvalidated design. Could still hit a problem we haven't foreseen. What (B)+(B) gets us is a better *chance* of working, not a certainty.

---

## File Structure

```
v5/
  V5_DESIGN.md         ← this doc
  config.py            ← Pair1Config (unchanged), StackConfig (new)
  layer1_language.py   ← reuses v4's pair1_foundation.py, kept frozen
  layer2_shortterm.py  ← sparse-strength graph, fast rates
  layer3_longterm.py   ← same graph structure, slow rates
  stack.py              ← wires L1 → L2 → L3 and back up to L1's residual
  train_layer1.py      ← copy of v4's train_pair1.py (unchanged)
  continual_read.py    ← feed text through the whole stack; strengths update automatically
  ask.py                ← generation with memory activity biasing the residual stream
  data/, out/            ← same as v4
```

---

## What v5 Explicitly Does NOT Do (Yet)

- **Layer 1 stays fully frozen** for v5. Future work can allow slow adaptation of Layer 1's own connection strengths, but that's out of scope for the first prototype — it would immediately raise stability risks (language ability could drift). Keep it locked until the stack itself is proven.
- **No gradient-based learning anywhere below Layer 1.** Same constraint as v1-v4. If v5 works, this is a stronger validation than v1-v4's parallel design of the "no-backprop-in-memory" bet.
- **No automatic feedback signal.** Reinforcement is driven purely by co-activation, not by whether generation was "good." Adding a reward-modulated three-factor rule is future work.
- **No sleep-like offline consolidation.** Consolidation from L2 to L3 happens continuously as connections cross thresholds, not in a periodic batch pass. Future work if the continuous version proves unstable.

---

## Success Criteria for v5

The first cut of v5 is a success if it can demonstrate:

1. **The stack runs end-to-end.** Language substrate trains, memory layers activate during reading, connections form and decay, generation runs with memory feedback. (This is the "does it work at all" bar — matches v1's success bar.)
2. **Layer 3 accumulates connections only for repeatedly-reinforced content.** Reading the same passage twice creates stronger L3 connections than reading two different passages once each. This tests whether the strengthen-decay dynamic actually implements the intended short→long promotion.
3. **Recall from memory measurably changes generation.** Compare generation with the memory stack active vs. bypassed. If bypass and active produce identical output, the memory isn't affecting anything and the integration failed.
4. **Layer 1 stays stable.** Since Layer 1 is frozen, this should be trivial — but worth measuring, since any bug in the "back-up to residual" path could effectively corrupt it via activation drift.

Not required for v5 success (but tracked): output *quality* improvement over v4. That's a scale/training issue, not an architecture issue, and v5's job is to prove the architecture, not beat v4 on generation quality with the same tiny model.
