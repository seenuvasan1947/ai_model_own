# v3: Scale Up — Dimensions, Layers, and Training Data

> **⚠️ CORRECTION (see Section 6):** the loss numbers below (Section 2) were **training loss only** — v3 originally had no held-out validation split, so they could not distinguish real generalization from memorization. Section 6 re-runs v3's *exact* original setup with a validation split added and shows the model was significantly overfitting the whole time. The comparisons in Sections 3-5 (generated output quality) are unaffected by this — those are still real, honest qualitative improvements — but the "scale helps, measurably" framing in Section 5 needs the correction in Section 6 read alongside it.

v1 and v2 kept the model deliberately tiny (64-dim, 2 layers, 358K params, one ~26.5k-word book) to prove the *architecture* works before spending compute on *quality*. v3 keeps the same architecture (all of v2's Titans-inspired upgrades carry over unchanged) and scales the two things that actually determine output quality: model capacity and training data volume.

---

## 1. What Changed

| Setting | v1 / v2 | v3 | Change |
|---|---|---|---|
| Embedding dimension | 64 | 128 | 2x |
| Layers | 2 | 3 | 1.5x |
| Attention heads | 2 | 4 | 2x |
| Context window | 32 tokens | 64 tokens | 2x |
| Parameters | 358,144 | 1,115,264 | ~3.1x |
| Knowledge corpus | *Alice's Adventures in Wonderland* alone (~26,525 words) | *Alice's Adventures in Wonderland* + *The Adventures of Sherlock Holmes* combined (~97,497 words in the two source files; ~150,380 tokens once tokenized with the QA set) | ~3.7x more words, 2 distinct authors/styles instead of 1 |
| Training steps | 4,000 | 6,000 | 1.5x |
| Architecture (Pairs 1/2/3 mechanism) | v2's design | **Unchanged** — same momentum-surprise, adaptive decay, surprise/momentum split, and vector-based memory injection as v2 | — |

Nothing about *how* the system learns changed in v3 — only *how much* it was given to learn from and *how much capacity* it has to learn with. This isolates the effect of scale from the effect of the v1→v2 mechanism upgrades.

---

## 2. Loss Curve Comparison

| Step | v1/v2 (358K params, Alice only) | v3 (1.1M params, Alice + Sherlock) |
|---|---|---|
| 0 | 8.29 – 8.31 | 8.31 |
| 400/500 | 5.44 – 5.51 | 4.52 |
| 2000 | (not run this far in v1/v2's 4000-step budget at this granularity) | 3.46 |
| 4000 | 3.58 (v1) / 3.56 (v2) | 3.03 |
| 6000 | *(v1/v2 stopped at 4000)* | **2.87** |

v3 reaches a lower loss than v1/v2 ever did, and does so while modeling a harder, more diverse corpus (two authors' styles instead of one) — consistent with the expected effect of more capacity plus more data.

---

## 3. Generated Output Comparison

**v1/v2 (358K params), asked "What is two plus two?":**
> "The opposite of the opposite of grass is small."

**v3 (1.1M params), same question:**
> "Two plus two is four."

A fully correct, coherent answer — a qualitative jump, not just a lower loss number.

---

**v1/v2, asked "Who did Alice meet?":**
> "You use to be at the Hatter. "A: The year."

**v3, same question:**
> "The Knave of Hearts, and then she did not in an offended tone, the moral of the Queen's eyes, "and, I'd have you wonder how would be beheaded!" She said the Gryphon, "they were very much!" The Mouse went on with its feet and Alice, "and she"

Not fully grammatical, but now uses the *correct* characters (Knave of Hearts, Queen, Gryphon, Mouse) in Wonderland-appropriate context and tone — a substantial coherence improvement over v1/v2's near-random word salad.

---

**v3, asked "What did Sherlock Holmes do?" (tests the new second book):**
> "You wear shoes on your feet."

Wrong/irrelevant (QA-set bleed-through, same failure mode as v1/v2 had with Alice content) — confirms the Sherlock content is present in the corpus but not yet reliably retrievable by the model at this scale/training budget. A real, honest remaining gap.

---

## 4. Continual Learning + Memory Injection (same test as v1/v2: the "Zeta" story)

v3, asked "Who is Zeta?" after reading the Zeta story via `continual_read.py`:
> "The largest ocean on Earth is the Pacspand-son, Sherlock Holmes, however, I was a man; and was no doubt of relord-sume, in the crudow of the loul and there was a small"

Notable: the answer starts by correctly reciting a fact from the generic Q&A set verbatim-correct ("The largest ocean on Earth is the Pac[ific]...") before drifting into a blend of Sherlock-style prose and garbled subword artifacts (e.g. "crudow", "loul" — likely BPE token-boundary errors at this vocab/training size). The injected Zeta memory vector (retrieved at only 0.55 similarity, down from 1.0-entry store) did not strongly steer the output this time — a smaller Pair 3 store (only 1 entry promoted from this short story) gives weaker retrieval signal than v1/v2's runs, which is expected given the same short story was used but v3's larger/different embedding space changed what counted as "surprising" enough to promote.

---

## 5. Honest Takeaways

- **Scale helps, measurably and visibly** — not just a lower loss number on a graph, but qualitatively better, more correct, more in-character output. This directly confirms the "scale gap" point made throughout this whole project: architecture alone was never going to produce fluent output; capacity and data were always the missing ingredient.
- **The QA-behavior/book-knowledge separation still works at this scale** — "two plus two is four" being answered perfectly while Alice/Sherlock questions still partially misfire confirms the *behavior* (answering directly) generalizes well even when the *knowledge* retrieval is still weak.
- **Sherlock Holmes content is present but not yet reliably surfaced** — a real limitation to address with either more training steps, a larger model, or better-targeted retrieval, not a sign the corpus mixing failed.
- **Subword-level garbling ("crudow", "loul") is new at this scale** and wasn't visible in v1/v2's more repetitive, smaller-vocabulary-usage output — likely an artifact of a larger, more diverse corpus outrunning the model's current capacity to fully resolve token boundaries. Worth watching as training continues.
- **v2's mechanism upgrades and v3's scale upgrades are independent and additive** — nothing here required changing Pair 2/3's logic; the next natural experiment is combining more scale with the mechanism fixes flagged as still open in v2 (looser retrieval threshold, no learned memory).

---

## 6. Corrected Results — v3's Original Setup, Re-run With a Validation Split

v4 (`v4/UPDATES.md`, fix #3) added a held-out train/val split and discovered overfitting invisible in v1-v3. To check whether that was specific to v4's other changes (fixed tokenizer, bigger vocab) or was already present in v3 as originally run, v3's *exact* original setup — same buggy tokenizer (trained on Alice-only text, not the full combined corpus — see `v4/UPDATES.md` fix #2 for why this was a bug), same `vocab_size=4000`, same 128-dim/3-layer model, same 6,000 steps — was re-run with only the val-split measurement added, nothing else changed.

**Result: the same overfitting pattern v4 found is confirmed present in v3's original configuration too.**

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 8.32 | 8.32 |
| 500 | 4.41 | 5.11 |
| 1500 | 3.60 | 4.77 |
| 3000 | 3.22 | 4.64 |
| 4500 | 2.95 | 4.46 |
| 5500 | 2.84 | 4.54 |

Train loss (2.84) closely matches the originally-reported final loss (2.87 — the small difference is just normal run-to-run variance). But val loss plateaued in the 4.4-4.6 range from roughly step 2000 onward and never approached the train loss — a persistent ~1.6-1.7 gap by the end of training.

**What this means for Sections 2-5 above:**
- The **loss numbers in Section 2** were real but incomplete — they measured memorization capability, not generalization, and should not be read as "the model got objectively better at language modeling" without this caveat.
- The **generated-output comparisons in Sections 3-4** are not undermined by this — a model can overfit its training loss while still producing more coherent output than a much smaller, undertrained model, because the *output quality* comparisons were direct, qualitative, human-read comparisons, not loss-derived. v3's answers genuinely were more coherent than v1/v2's, independent of the overfitting finding.
- The **"scale helps" conclusion in Section 5** needs a qualifier: **scale helped *this specific corpus size become memorizable* by a 1.1M-parameter model, not necessarily *generalizable*.** A larger corpus, more regularization (dropout, weight decay tuning), or fewer training steps (early stopping around where val loss was lowest) would be needed to know whether scale genuinely improves generalization here, or only gives the model enough capacity to memorize a bigger dataset.

This correction doesn't invalidate v3's real, observed output-quality improvements — but it does mean "v3 proves scale straightforwardly helps" was an overclaim. The honest version: **v3 proves scale improves observed output quality on this size of corpus, while also being large enough to overfit it — both things are true at once, and future work should track val loss, not just train loss, to tell which effect is doing the work.**
