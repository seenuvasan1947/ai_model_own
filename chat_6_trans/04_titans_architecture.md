# 04 - Titans: Architecture, Training Speed, Experiments, Limits

## 1. Three memory branches
| Branch | What it is | Learned when | Analogy |
|--------|-----------|--------------|---------|
| **Core (short-term)** | Attention (full or sliding-window) over current chunk | Training | Working memory |
| **Long-term memory** | Neural memory `M` (file 03) | Weights change at **test time** | Life experience |
| **Persistent memory** | `N_p` learnable tokens `P = [p_1…p_Np]`, **independent of input**, prepended to the sequence | Training only (fixed after) | Built-in task knowledge / skills |

### Why persistent memory?
- It stores *task knowledge* (what kind of problem this is).
- Acts like a fix for attention's bias toward the **first tokens** (a known "attention sink" effect): these always-present tokens give attention stable anchors.
- Equivalent view: replaces/extends the Transformer's FFN as data-independent key-value store (FFN weights act like persistent keys/values).

## 2. Three ways to combine them (Titans variants)

### 2.1 MAC - Memory as a Context
1. Split the long input into **segments/chunks** (e.g., 512 tokens).
2. For the current chunk, use it to **query the long-term memory** → get a summary `h_t = M*(q_t)` of relevant past.
3. Build attention input = `[persistent tokens] ‖ [retrieved memory h_t] ‖ [current chunk]`.
4. Attention decides what is useful from long-term memory *and* the current chunk.
5. Output of attention is used to **update** the long-term memory (store what mattered).
6. Final output: gate between attention output and memory output.
✔ Attention acts as a smart filter ("is the retrieved info relevant right now?"). Best in long-context tasks.

### 2.2 MAG - Memory as a Gate
Two parallel branches on the same input (with persistent tokens prepended):
- Branch A: **sliding-window attention** (precise short-term).
- Branch B: **neural memory** (compressed long-term).
Combine with a **gating (non-linear weighted) mix**. Good efficiency; branches complement each other.

### 2.3 MAL - Memory as a Layer
Stack: `input → neural memory layer → sliding-window attention → output` (like a hybrid RNN + attention stack, as in some Mamba/Samba-style models). Simple, but the attention can only see what the memory layer passes on, so slightly weaker.

### 2.4 Titans (LMM) alone
Using only the neural memory (no attention) already beats many modern recurrent models → shows long-term memory is strong on its own.

## 3. Making it fast - parallel training
Problem: updating `M` token by token (online gradient descent) is sequential → slow.
Solution:
- Split the sequence into **mini-batches/chunks of size b**. Within a chunk, compute all gradients **w.r.t. the same starting weights** (like mini-batch gradient descent) - this becomes **matrix multiplications** (GPU-friendly).
- Data-dependent `η, θ, α` are computed in parallel; the momentum and decay recurrences are linear → solved with a **parallel associative scan**.
- Result: training speed competitive with other efficient recurrent models; complexity linear in sequence length (for the memory branch). Attention part stays quadratic **within its small window only**.

## 4. Experiments (what the paper reports)
Model sizes: ~170M, 340M, 400M, 760M parameters; trained on FineWeb-Edu-style data.

1. **Language modeling & commonsense reasoning**: Titans variants beat comparable Transformer++ and modern recurrent baselines (Mamba/Mamba2, Gated DeltaNet, TTT, etc.) on perplexity and accuracy; hybrid variants (MAC, MAG) best.
2. **Needle-in-a-Haystack (S-NIAH)**: find one fact in a huge text. Titans keep accuracy high as length grows to 16K tokens while many baselines collapse.
3. **BABILong benchmark** (reasoning over facts spread through very long documents): Titans (MAC) outperformed far larger models (including big LLMs with retrieval augmentation) in the paper's setup; the paper claims **scaling to context windows larger than 2 million tokens** with higher accuracy.
4. **Time-series forecasting** and **DNA modeling**: competitive/state-of-the-art, showing the idea isn't text-only.
5. **Ablations**: removing momentum, forgetting (weight decay), convolution, or persistent memory each hurts; **deeper memory** helps; **MAC and MAG** best overall.
(Always read the paper's tables for exact numbers; they are benchmark- and setup-dependent, and independent replications may differ.)

## 5. Limitations / critical view
- Test-time learning adds **compute and complexity** (inner-loop gradients, deeper memory slower).
- Sensitive to hyper-parameters (chunk size, memory depth, learning-rate parameterization).
- Surprise (gradient of a reconstruction loss) is a *proxy* for "importance" - not always what matters for a downstream task.
- Original public code was not released with the paper at first; community re-implementations exist; claims at 2M+ tokens were reported by authors and are harder to verify.
- Memory is a lossy compression (unlike attention which keeps everything exactly inside the window).
- Safety/stability: a model that changes its own weights while running needs care (drift, poisoning with adversarial inputs).

## 6. Follow-up: "Nested Learning" / HOPE (the *continual learning* paper)
Behrouz et al. (Google, NeurIPS 2025) generalize Titans' idea:
- A model is not one learning process but **nested optimization problems**, each updating at its own **frequency** (like brain waves: fast ones update often, slow ones rarely).
- **Continuum Memory System (CMS)**: instead of "short-term vs long-term" two buckets, a *spectrum* of memory modules updated at different rates (replaces the FFN block).
- **HOPE**: a self-modifying, Titans-based recurrent architecture combined with CMS, aimed at **continual learning** - learning new things without catastrophically forgetting old ones.
- Insight: even optimizers (momentum, Adam) and attention are *associative memories* compressing their context. So "architecture" and "optimizer" are the same idea at different levels.
Treat this as the next chapter after Titans.

Next → `05_connection.md`
