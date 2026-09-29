# Activation Functions in Modern LLMs

| Model | Activation function |
|---|---|
| GPT-2 / GPT-3 | GELU |
| BERT | GELU |
| LLaMA / LLaMA 2 / LLaMA 3 | SwiGLU |
| PaLM | SwiGLU |
| Mistral / Mixtral | SwiGLU |
| Our prototype (`model.py`) | GELU |

**Why nonlinearity is required at all:** stacking layers using only plain weighted sums (no nonlinear function) mathematically collapses into a single layer, no matter how many layers are stacked — matrix multiplication composed with matrix multiplication is still just a matrix multiplication. The activation function is what makes depth actually useful.

**GELU vs SwiGLU:**
- GELU: squashes one value nonlinearly.
- SwiGLU (gated): computes two linear projections from the input, applies Swish/SiLU to one, multiplies them together elementwise — a learned, per-dimension gate controlling how much information flows through. Measurably improves quality for the same parameter budget (per PaLM/LLaMA papers), which is why it's become close to a de facto standard in recent open frontier models.

**Note:** Anthropic has not publicly disclosed Claude's internal activation function — not confirmed, only industry-general inference.

---

## Can we invent our own activation function?

Activation functions aren't handed down by law — they're just functions that satisfy a few requirements, and researchers invent new ones regularly:
- Must be nonlinear (otherwise, as we covered, depth collapses into one layer).
- Must be differentiable (or differentiable almost everywhere) if it's going to sit inside a backprop-trained part — pair 1 in our design.
- Beyond that, it's genuinely a design choice. Swish/SiLU (which SwiGLU is built on) was literally discovered by Google researchers running an automated search over candidate functions — it's not some deep mathematical law, it was found empirically to work better, then adopted. GELU was also a specific, deliberate design choice (a smooth approximation combining properties of ReLU and dropout-like stochastic regularization), not an inevitability.

Given the rest of this project is already genuinely novel (sparse quantized Hebbian pairs 2/3, freeze-after-Phase-1 routing), the recommendation is to **not** invent a new activation function for Pair 1 — reuse a proven option there and spend original-design effort on the parts that are actually novel and unavoidable.

---

## Recommended architecture split (6-dim / 4-layer example, 3 pairs)

| Component | Recommendation | Why |
|---|---|---|
| **Pair 1 — language foundation (backprop-trained)** | Standard decoder-only transformer block, **GELU** (or SwiGLU) | Proven, well-tested, removes a variable — this part most needs to just work, since everything else depends on it |
| **Pair 2 — short-term / continuously updating memory** | No traditional activation — Hebbian outer-product local update, optionally squashed with `tanh` only to bound update magnitude | Not a weighted-sum-then-nonlinearity computation; magnitude control matters more than a learned nonlinearity here |
| **Pair 3 — long-term / permanent memory** | No activation function at all — sparse quantized key-value / associative store | This is lookup + reinforcement, not a backprop layer, so the concept doesn't apply the same way |

**Bottom line:** use a standard, proven activation (GELU) only where backprop is actually happening (Pair 1). Don't over-engineer novelty into a part of the system that doesn't need it.

---

## Knowledge vs. the skill of answering — where Q&A ability actually comes from

Raw text (a newspaper article, a novel, any plain document) is never pre-formatted as question-answer pairs — so how does a model learn to *answer questions* about content that was never phrased as Q&A? Knowledge and the skill of answering are learned from two different, separate sources, not the same one.

**Source 1 — raw text (any book, article, corpus) → teaches knowledge, not answering behavior.**
When the model reads "Alice fell down the rabbit hole and found a bottle labeled 'Drink Me,'" plain next-token prediction absorbs that fact into its weights — it now "knows" this happened. No Q&A format needed for this part; this is just Step 1 pretraining — knowledge is a side effect of getting good at prediction on any text, question-formatted or not.

**Source 2 — a small, separate, generic Q&A example set → teaches the *behavior* of answering directly, not new facts.**
This set does not need to be about the same book/topic at all. Simple generic examples like:
```
Q: What color is the sky?
A: The sky is blue.

Q: What is 2 plus 2?
A: 2 plus 2 is 4.
```
teach the *pattern*: "when input looks like a question, generate a direct, on-topic answer instead of continuing with more narrative text." This is a behavioral/structural skill, not a factual one — and once learned, it **generalizes** to whatever knowledge the model has, regardless of where that knowledge came from.

**Why this isn't a contradiction:** this is literally how real LLMs work — pretrained on mostly non-Q&A internet text (articles, code, forums, books), then instruction-tuned on a comparatively tiny set of Q&A examples. That small tuning step makes the model start answering questions about *everything* it absorbed during the much larger raw-text phase, not just topics that happened to appear in Q&A format during pretraining.

**Concretely, for a one-book training scenario:**
1. Train Phase 1 on the raw book text (narrative, exactly as written, no reformatting) — this is where it learns everything *about* the book.
2. Separately, add a small, generic, reusable Q&A-behavior dataset (a few dozen to a few hundred simple question→answer examples, about anything, not necessarily the book) — this teaches *how to respond to a question at all*.
3. At inference, ask it a question about the book. It applies the *learned skill* (respond directly to a question) to the *learned knowledge* (facts about the book) — even though those two things were never seen together during training. This combination, not hardcoding, is what produces a genuine, generative, non-templated answer.
