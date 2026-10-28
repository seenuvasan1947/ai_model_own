# 03 - Titans: Learning to Memorize at Test Time - Core Ideas

Paper: Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research), arXiv 2501.00663.

## 1. The problem in plain words
Imagine reading a 1,000-page novel and being asked about page 3 on page 990.
- **Transformer** = a person with a *perfect photographic view of the last N pages only* (accurate but short window, expensive to widen).
- **Linear RNN / "linear Transformer" / Mamba-type models** = a person who keeps a *small fixed-size notebook* and rewrites it as they read. Cheap, but the notebook is tiny and old facts get overwritten/compressed away.

Goal of Titans: give models a **real long-term memory** that is cheap, large-capacity, and that **keeps learning while the model is running**.

## 2. Human memory analogy (the paper's motivation)
| Human | Titans component |
|-------|------------------|
| Short-term memory (accurate, limited, right now) | **Attention** over the current window |
| Long-term memory (stores important/surprising things over a lifetime) | **Neural long-term memory module** (new!) |
| Knowledge/skills built in over evolution (task-knowledge) | **Persistent memory** (learned, input-independent tokens) |

Key quote-level idea (paraphrased): memory is not one block - it is a *system* of interconnected modules (short-term, long-term, meta/persistent) that can work independently.

## 3. Memory = learning (key philosophical idea)
The paper views *a memory as a thing that learns to store a mapping from keys to values*. So instead of a memory being a matrix or vector just written to, make the memory a **small neural network** `M` (e.g., a multi-layer MLP) whose **weights are updated at test time** to *memorize* what it sees.

- "Memorize" = fit the network so that `M(key) ≈ value`.
- "Retrieve" = ask the network: `M(query)`.
- "Write" = one step of gradient descent on that network, **while the model is processing the input** (not during the original training). That's "learning at test time".

Note the two levels:
- **Outer loop (normal training)**: learns W_K, W_V, W_Q, how to set learning rate/forgetting, the attention layers, etc. (meta-learning: *learn how to memorize*).
- **Inner loop (test time)**: updates `M`'s weights on the fly. No labels or human supervision needed.

## 4. What should be memorized? → **Surprise**
Humans remember surprising events (a car crash on your way to work) and forget routine ones (usual route). Titans measures surprise as the **gradient of the memory's loss** w.r.t. the new input:

- Associative memory loss for the current token `x_t`:
  ```
  k_t = x_t W_K       v_t = x_t W_V
  loss(M; x_t) = || M(k_t) − v_t ||²
  ```
- Surprise = `∇ loss(M_{t−1}; x_t)`. A big gradient = memory predicted badly = "surprising" = needs a big update. If memory already knows it, gradient ≈ 0 and nothing changes.

### Tiny numeric example
`M` = one 2×2 linear map, starts at zeros. Token: key `k=[1,0]`, value `v=[3,0]`. Learning rate θ = 0.1.
- Step 1: `M k = [0,0]`, error = `[−3,0]`, gradient = `2·(Mk − v)·kᵀ = [[−6,0],[0,0]]`. Update `M = M − 0.1·grad = [[0.6,0],[0,0]]`. Retrieval `M k = [0.6, 0]`.
- Same token again: error `[−2.4,0]`, smaller gradient (−4.8) → `M` becomes `[[1.08,0]...]`. Surprise shrinks as the fact is learned. ✔
(Real Titans uses a deeper non-linear MLP, so it can store far richer patterns.)

## 5. The problem with "pure momentary surprise" → **Past + momentary surprise (momentum)**
If a big surprise happens, the *following* tokens are often also relevant, but their individual gradients may be small, so the model would miss them (it'd fall into a flat "plateau"). Fix: add **momentum**, as in optimizers.

```
S_t = η_t · S_{t−1}  −  θ_t · ∇ loss(M_{t−1}; x_t)
        ^ past surprise      ^ momentary surprise
```
- `η_t` (data-dependent) decides how much past surprise carries over.
- `θ_t` (data-dependent) decides how strongly the current surprise counts.
Both are **functions of the input**, so the model can say "this token begins a new context, forget past momentum" (η→0) or "this is a big event, keep going".

## 6. Forgetting (memory management)
Memory has finite size; with millions of tokens it overflows. Add a **forget gate (weight decay)**:
```
M_t = (1 − α_t) · M_{t−1} + S_t          α_t ∈ [0,1], data-dependent
```
- α → 1: wipe memory (new document).
- α → 0: keep everything.
This is the same math as the forget gate in LSTMs / decay in modern RNNs, and also equals **weight decay** in gradient descent. The paper shows many modern recurrent models (Mamba, DeltaNet, gated variants) are special cases of this framework with a *shallow (linear)* memory, no momentum, etc.

## 7. Deep memory
A matrix-valued memory (a single linear layer) can only store *linear* relations - limited capacity. Titans uses an `L_M`-layer **MLP** as memory (L_M ≥ 2). Paper results: deeper memory → lower perplexity, especially on long sequences, at some speed cost.

## 8. Retrieval
To read, project the input into a query and **forward-pass** without updating:
```
q_t = x_t W_Q
y_t = M*(q_t)      // M* = apply the network, no weight change
```
(Queries and keys are normalized with ℓ2-norm; a 1D depthwise convolution + SiLU non-linearity is applied to q,k,v projections - stability/efficiency tricks as in recent recurrent models.)

## 9. Summary of the "Neural Long-Term Memory" (LMM) module
1. Make k, v, q from the token.
2. Compute surprise (gradient of `||M(k)−v||²`).
3. Combine with momentum (past surprise).
4. Apply forgetting.
5. Update `M`'s weights.
6. Retrieve with `M(q)` when needed.

Next → `04_titans_architecture.md` (how to plug this memory into a full model, speed tricks, results)
