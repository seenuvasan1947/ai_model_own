# 01 - Transformer Basics ("Attention Is All You Need")

## 1. The problem the paper solved
Task: **sequence-to-sequence** (e.g., translate English → German). Before 2017 the best models were **RNNs / LSTMs / GRUs** (plus CNNs) with an encoder-decoder design and a little "attention" bolted on.

Problems with RNNs:
1. **Sequential**: to read word 5 you must first process words 1-4 one by one. GPUs love parallel work, so training was slow.
2. **Long-distance forgetting**: information must travel through many steps; it fades. In "The cat, which ate the fish that was caught yesterday, **was** happy", linking *cat* to *was* is hard.
3. **Fixed-size bottleneck**: the whole sentence is squeezed into one hidden state.

**The paper's idea**: drop recurrence and convolution entirely. Use only **attention**, so every word can look at every other word *directly, in parallel*.

Path length between any two words: RNN = O(n) steps; Transformer = **1 step**.

## 2. From text to numbers

### 2.1 Tokenization
The paper uses **byte-pair encoding (BPE)** (about 37k shared tokens for En-De; word-piece 32k for En-Fr). "Transformers" → `Transform` + `ers`.

### 2.2 Embeddings
Each token ID looks up a row in an **embedding table** → vector of size `d_model = 512` (base model). The paper multiplies embeddings by `√d_model` and **shares** the same weights for input embeddings, output embeddings, and the final pre-softmax layer ("weight tying").

Example: `cat → [0.12, -0.8, 0.33, ... 512 numbers]`. Similar words get similar vectors after training.

## 3. Positional encoding (word order)
Attention by itself treats a sentence like a *bag of words*: "dog bites man" = "man bites dog". So we **add** a position signal to each embedding.

Paper's choice: fixed **sinusoidal** formulas

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Each dimension is a wave with a different wavelength (from 2π to 10000·2π).
- Think of a **clock with many hands** (seconds, minutes, hours...): the combination uniquely identifies the position.
- Neat property: `PE(pos+k)` is a linear function of `PE(pos)`, so relative offsets are easy to learn.
- They also tried *learned* position embeddings: nearly identical results. Sinusoids chosen because they may extrapolate to longer sequences.

Final input to layer 1: `embedding + PE`.

## 4. Self-attention - the core idea

### 4.1 Intuition
Sentence: *"The animal didn't cross the street because **it** was too tired."*
What does "it" refer to? Attention lets the word "it" **look at all other words and decide how much each matters**, then build a new representation of "it" mixing in info mostly from "animal".

### 4.2 Query, Key, Value (library analogy)
Every token produces three vectors via three learned matrices (`W_Q, W_K, W_V`):
- **Query (Q)** - "what am I looking for?" (your search phrase)
- **Key (K)** - "what do I contain / advertise?" (book's label)
- **Value (V)** - "the actual content I give if chosen" (book's content)

Steps for one token:
1. Compare its **Q** with every token's **K** (dot product = similarity score).
2. Scale, softmax → **attention weights** (probabilities).
3. Output = weighted sum of all **V**s.

### 4.3 Formula (Scaled Dot-Product Attention)
```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V
```
Why `/√d_k`? With big `d_k`, dot products get large, softmax saturates (one weight ≈1, others ≈0) and gradients vanish. Dividing keeps values in a healthy range.

### 4.4 Worked numeric example (tiny, d=2)
Query for "it": `q = [1, 0]`

| Token | Key | Value | q·k | ÷√2 | exp | weight |
|-------|-----|-------|-----|-----|-----|--------|
| animal | [1,0] | [2,0] | 1.0 | 0.707 | 2.03 | **0.455** |
| street | [0,1] | [0,2] | 0.0 | 0.000 | 1.00 | 0.225 |
| tired | [.5,.5] | [1,1] | 0.5 | 0.354 | 1.43 | 0.320 |

Output = 0.455·[2,0] + 0.225·[0,2] + 0.320·[1,1] = **[1.23, 0.77]**
→ new vector for "it" is mostly "animal"-flavored. 

All tokens do this simultaneously as **matrix multiplication** - this is why it's fast on GPUs.

### 4.5 Where the Q, K, V come from
In **self-attention**, Q, K, V all come from the *same* sequence. In **encoder-decoder attention** (file 02), Q comes from the decoder, K and V from the encoder.

## 5. Multi-Head Attention
One attention = one "viewpoint". Language needs many (grammar, coreference, nearby words, topic...).

- Split into `h = 8` heads; each head has its own `W_Q, W_K, W_V` projecting 512 → `d_k = d_v = 64`.
- Run attention in each head in parallel; **concatenate** (8×64 = 512) and multiply by `W_O`.
```
MultiHead = Concat(head_1..head_h) · W_O
head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)
```
- Same total compute as one 512-dim head, but richer. Paper's ablation: 1 head loses ~0.9 BLEU; too many heads (32) also hurts a little.
- Papers' visualizations show some heads track long-range dependencies, some track syntax (e.g., "its" ↔ noun).

## 6. Why self-attention (paper's Table 1)
| Layer type | Cost per layer | Sequential ops | Max path length |
|------------|----------------|----------------|-----------------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k n) |

Attention is cheaper than recurrence when `n < d` (typical sentences). Its weakness: **O(n²) in sequence length** - becomes the big problem for long documents (this is exactly what Titans tackles; see file 05).

Next → `02_transformer_architecture.md`
