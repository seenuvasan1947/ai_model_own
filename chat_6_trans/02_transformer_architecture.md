# 02 - Transformer Architecture, Training, Results, Limits

## 1. Big picture
```
Input tokens → Embedding + Positional Encoding → [ENCODER × 6] ─┐
                                                                 ├→ memory (K, V)
Output tokens (shifted right) → Embedding + PE → [DECODER × 6] ←─┘
                                                      ↓
                                               Linear → Softmax → next-token probabilities
```
Base model: `N = 6` layers each side, `d_model = 512`, `d_ff = 2048`, `h = 8`, dropout 0.1, ~65M parameters. "Big" model: d_model 1024, d_ff 4096, h 16, ~213M.

## 2. One Encoder layer (repeated 6×)
Two sub-layers:
1. **Multi-head self-attention** (every word looks at every word in the input)
2. **Position-wise Feed-Forward Network (FFN)**

Each sub-layer is wrapped as: `output = LayerNorm( x + Sublayer(x) )`

### 2.1 Residual connection (`x + Sublayer(x)`)
A shortcut that adds the input back to the output. Like editing a draft rather than rewriting from scratch: keeps original information and lets gradients flow in deep networks (6+ layers train stably).

### 2.2 Layer Normalization
Re-centers and re-scales each token's vector (mean 0, variance 1, then learned scale/shift). Keeps numbers from exploding or vanishing; speeds up and stabilizes training. (Paper uses "post-norm"; many modern models use "pre-norm".)

### 2.3 Feed-Forward Network
```
FFN(x) = max(0, x W1 + b1) W2 + b2      (512 → 2048 → 512, ReLU)
```
Applied to **each position independently** (same weights for all positions, different across layers). Attention *mixes information between tokens*; the FFN *processes each token individually* (think "thinking" step). Later research found FFNs act like key-value **memories** storing facts - a thread that connects to Titans.

## 3. One Decoder layer (repeated 6×)
Three sub-layers:
1. **Masked multi-head self-attention** - decoder looks at previously generated words only.
2. **Encoder-decoder (cross) attention** - Q from decoder, K,V from encoder output. Lets the decoder "look back at the source sentence" (like glancing at the French text while writing English).
3. **FFN**
(each with residual + LayerNorm)

### 3.1 Masking (no cheating)
When predicting word 4, the model must not see word 4 or later. Implementation: before softmax, set scores for future positions to `-∞` → weight becomes 0.

```
          I   love  tea
I        [✓    ✗     ✗ ]
love     [✓    ✓     ✗ ]
tea      [✓    ✓     ✓ ]
```
This **causal mask** preserves the *auto-regressive* property: generate one token at a time, feed it back as input. (Titans and GPT-style LLMs use this same causal setting.)

Also: **padding masks** hide padding tokens in batches.

## 4. Output layer
Decoder output → Linear (512 → vocab size) → Softmax → probability for each possible next token. Pick the best (paper uses **beam search**, beam size 4, length penalty 0.6).

## 5. End-to-end example (translation)
Input: "I love tea" → German "Ich liebe Tee"
1. Tokenize, embed, add positions.
2. Encoder 6 layers: each word's vector gets enriched with context ("love" learns its subject is "I").
3. Decoder starts with `<start>`. Predict "Ich". Feed `<start> Ich` → predict "liebe" (using masked self-attention + cross-attention to encoder). Continue until `<end>`.

During **training** all positions are predicted *at once* in parallel (teacher forcing with the mask) - the speed win over RNNs.

## 6. Training details (paper)
- Data: WMT 2014 English-German (4.5M sentence pairs), English-French (36M).
- Hardware: 8 NVIDIA P100 GPUs. Base: 12 hours; big: 3.5 days.
- Optimizer: **Adam** (β1=0.9, β2=0.98, ε=1e-9) with a **warm-up** schedule:
  `lrate = d_model^-0.5 · min(step^-0.5, step · warmup^-1.5)`, warmup = 4000 steps (learning rate rises then decays).
- Regularization: dropout 0.1 (on sub-layer outputs and embeddings), **label smoothing** 0.1 (don't be 100% sure → better BLEU).

## 7. Results
| Task | Transformer (big) | Previous best |
|------|------|------|
| WMT14 En→De | **28.4 BLEU** | ~26 (ensembles) |
| WMT14 En→Fr | **41.8 BLEU** (single model) | ~41 (at much higher cost) |
Plus it generalized to English constituency parsing. Training cost was a small fraction of earlier SOTA models.

## 8. What happened after (impact)
- Encoder only → **BERT** (understanding). Decoder only → **GPT** family (generation). Both → T5, original translation models.
- Extended to images (ViT), audio, proteins (AlphaFold), code, video.
- Scaling laws: more data/params/compute → predictably better.

## 9. Limitations (important for Titans)
1. **Quadratic cost**: attention matrix is n×n. 1k tokens = 1M scores; 1M tokens = 1 trillion. Memory & time explode.
2. **Limited context window**: only sees what fits in the window. Beyond it, information is simply gone.
3. **KV-cache grows linearly** during generation (stores K,V for all past tokens).
4. **Frozen after training**: weights don't change when you use the model. It can't truly *learn* from a new document it reads - it only "looks" at it in context (short-term, working-memory-like).
5. **Poor length generalization** beyond trained context length.

These five points are the doorway to the Titans paper → `03_titans_ideas.md`
