# Prototype Spec: Brain-Inspired Continual Language Learner

This is the buildable v0 spec for the design discussed: a decoder-only transformer, trained deeply on one language first (Phase 1, backprop), then extended with routed/localized continual learning (Phase 2, research-stage, not built in v0).

**Scope of this prototype:** Phase 1 only — a working, small, decoder-only generative language model trained on one language, with the architecture already shaped (via a lightweight Mixture-of-Experts layer) to support Phase 2 routing later. Phase 2's local/forward-only update mechanism is *not* implemented yet — that's a follow-up milestone once Phase 1 is validated.

---

## 1. Architecture Decisions

| Choice | Decision | Why |
|---|---|---|
| Encoder vs Decoder | **Decoder-only** | Generation task (predict next token), not seq-to-seq mapping |
| Context size | **512 tokens** (v0) | Small enough to train fast on modest hardware; not testing long-context yet |
| Model size | **~10-25M params** | Fast iteration; big enough to show real generative behavior on a narrow single-language corpus |
| Layers | 6 | Standard small-GPT depth |
| Embedding dim | 384 | Matches nanoGPT-small conventions |
| Attention heads | 6 | 64 dims/head |
| Tokenizer | **Byte-Pair Encoding (BPE)**, trained only on the target language corpus | Keeps vocabulary tight and language-specific, matching the "deep single-language foundation" design |
| Expert layer | Simple **top-1 MoE feedforward block** (4 experts) in place of the standard dense FFN | Establishes routable, separable "sub-spaces" now, so Phase 2 can later attach localized updates to individual experts without re-architecting |
| Training method | Standard **backprop** (Phase 1 only, per the agreed design) | Bootstraps the language foundation; Phase 2 will later swap the update rule, not the weights |

---

## 2. Why MoE Now, Even Though Phase 2 Isn't Built Yet

Putting a Mixture-of-Experts feedforward block in from the start — even while still using plain backprop — means:
- The "separate space per concept" structure exists in the architecture from day one, so Phase 2 doesn't require redesigning the model, only replacing *how* individual experts get updated.
- We can already inspect *which* expert activates for which kind of input during Phase 1 training, as an early signal of whether meaningful specialization is emerging (this doubles as an experiment, not just infrastructure).

---

## 3. Milestones

1. **M0 (this milestone):** decoder-only transformer + MoE-FFN + BPE tokenizer, training loop, overfit-check on a tiny sample (sanity test).
2. **M1:** train on a real single-language corpus (few MB–few hundred MB of clean text), validate it generates coherent text and correctly handles a context-dependent case like "bank" (river vs. money) as discussed.
3. **M2 (future, not in this prototype):** implement Phase 2 — freeze the router, replace backprop with a local Forward-Forward-style update rule scoped to individual experts, test continual learning without forgetting.

---

## 4. File Layout

```
language learner/
  prototype_spec.md      <- this file
  prototype/
    model.py             <- decoder-only transformer with MoE-FFN
    tokenizer.py          <- BPE tokenizer training/loading
    train.py              <- training loop
    config.py              <- hyperparameters
    data/                  <- put your language corpus .txt here
```

---

## 5. What This Prototype Does NOT Do Yet

- No continual/online learning (Phase 2) — still standard backprop, weights are frozen after training like any normal model, for now.
- No usage-based decay/pruning of expert weights.
- No cross-expert integration mechanism beyond the router's natural mixing.
- No multi-language support — single language only, by design.

This is intentionally scoped to validate the *base generative model + expert-routable structure* before attempting the harder, unvalidated continual-learning mechanism.
