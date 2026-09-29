# Gap Analysis: Our Design vs. Titans (Google Research, 2024)

Reference: `../../files/2501.00663v1.pdf` ("Titans: Learning to Memorize at Test Time") and its independent critique `../../files/2510.09551v1.pdf` ("Titans Revisited"). This documents what Titans does differently from v1, what we're borrowing for v2, and what we're deliberately still not adopting.

---

## What's the same (shared insight)

Both split memory into a short-term tier (attention/fast, bounded by context) and a long-term tier (persistent, survives beyond the context window). Both are motivated by the same limitation: attention alone forgets everything once it falls out of the context window.

---

## Where the mechanisms genuinely diverge

| | Titans | Our v1 |
|---|---|---|
| What the long-term memory *is* | A neural network (an MLP) — the memory *is* a set of weights | An explicit table — a growing list of (vector key, text label, strength) entries |
| How it updates | Online **gradient-based** rule at test time — a "surprise" signal (gradient of an associative-memory loss), momentum, and a learned forgetting gate | Cosine-similarity threshold rules only — no gradient anywhere |
| How the update rule was designed | **Learned** during the main training phase (meta-trained via backprop to discover a good online-update rule) | **Hand-designed** by us — threshold, promotion rule, decay all manually chosen |
| Retrieval | Implicit — a forward pass through the memory MLP | Explicit — discrete nearest-neighbor lookup, spliced into the prompt as text |
| Integration with the base model | Tightly fused architecturally (Memory-as-Context / Memory-as-Gate / Memory-as-Layer, all differentiable, trained jointly) | Loosely bolted on — Pair 1 frozen, injection done by decoding vectors to text and prepending to the prompt |
| Validation | Tested at real scale (language modeling, genomics, time-series; benchmarked vs. transformers and linear RNNs) | Tested qualitatively at toy scale (358K params); no benchmark comparison |

**The core distinction:** Titans' long-term memory is learned and differentiable, even at test time. Ours is hand-engineered and non-differentiable — closer to RAG than to Titans' actual mechanism. This was a deliberate tradeoff: our original design goal was "no backprop anywhere except once, for Pair 1," which Titans does not share (it uses gradients throughout, including online at inference).

---

## What v2 borrows from Titans (compatible with our no-gradient rule)

1. **Momentum on surprise** — instead of a hard instantaneous threshold, keep a decaying running average of surprise so a sustained pattern builds influence even if no single moment is extreme.
2. **Adaptive, input-dependent decay** — instead of one fixed decay constant, scale decay by how dissimilar new input is to what's currently held (dissimilar → forget faster, similar/reinforcing → forget slower). A heuristic stand-in for Titans' learned forgetting gate.
3. **Memory-as-context integration** — instead of decoding memory vectors to text and pasting them into the prompt (lossy round-trip), splice the retrieved vectors directly into the hidden-state sequence Pair 1's later layers process, matching the spirit of Titans' "Memory as Context" variant.
4. **Explicit surprise/momentum separation** — split Pair 2's single trace update into two named components (instantaneous surprise vs. accumulated momentum) instead of blending both into one Hebbian update.

## What v2 deliberately does NOT adopt

- **A real learned, gradient-updated memory module.** This is Titans' actual core mechanism — but adopting it means giving Pairs 2/3 gradients, which breaks the original design constraint. Left as an explicit, separate future decision (see `../architecture_design.md`), not silently absorbed into v2. **Important update below: this is not simply a "future upgrade we haven't gotten to" — there is documented evidence it may not even work as hoped. Read the correction section before treating gradient-based memory as an obvious next step.**
- **Architectural fusion (MAC/MAG/MAL as trained-in variants).** We approximate the *spirit* of Memory-as-Context (item 3 above) as a forward-pass-only splice into a frozen Pair 1, not as something Pair 1 was trained to expect — a real, honest limitation versus Titans' jointly-trained integration.
- **Benchmark validation.** v2 is still evaluated qualitatively, not against baselines.

---

## Correction: don't assume adopting more of Titans' mechanism would help ours

An earlier draft of this document implied that closing the gaps above (especially "add a real learned, gradient-updated memory") would be a clear path to better performance, on the logic of "Titans does this, so doing it too should help." **That implication was not backed by evidence and was directly challenged and corrected during this project** (see the conversation this document comes from). The correction, for the record:

**"Titans Revisited" (`../../files/2510.09551v1.pdf`) already tested almost exactly our situation, and found a warning sign, not a green light.** From that paper's own reproduction: when they froze the Transformer backbone and updated *only* the neural memory weights at test time over many epochs, performance stayed flat or slowly degraded — their own words: "memory updates alone are insufficient for meaningful test-time learning." They attribute this to a mismatch between the frozen backbone's key/value projections and how the memory evolves, and propose *jointly* adapting memory and backbone together as necessary future work — which nobody has published a working answer for yet, as of these two papers.

**Why this matters directly for us:** our architecture — Pair 1 frozen, Pairs 2/3 updating separately — is structurally the same "frozen backbone + memory-only updates" setup Titans' own reproduction found doesn't reliably work, *even when the memory update uses real gradients*. So adding gradients to Pairs 2/3 would not automatically fix anything; it would put us in a configuration with a documented, unresolved failure mode, not a proven improvement.

**What the evidence we actually have (not a paper's, ours) points at instead:** every real, measured quality improvement across v1 → v2 → v3 came from scaling Pair 1 (bigger model, more training data) — see `../v3/COMPARISON.md`. Nothing about Pairs 2/3's mechanism has yet been shown, empirically, in our own testing, to move output quality. That's a real, in-hand result in favor of continuing to prioritize Pair 1 scale and fixing concrete observed bugs (vocabulary size not scaling with corpus, retrieval similarity threshold too loose) over chasing architectural parity with Titans for its own sake.

**The takeaway:** the split, non-gradient design for Pairs 2/3 is not shown to be inferior to Titans' approach — both have open, unresolved problems at this frontier. Treat the "what v2 deliberately does NOT adopt" list above as an honest inventory of differences, not a to-do list of proven upgrades.
