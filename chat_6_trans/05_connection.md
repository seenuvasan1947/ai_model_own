# 05 - How the Transformer and Titans Are Linked

## 1. One-line story
**Transformer (2017)** gave AI a *perfect but short and frozen working memory* (attention).
**Titans (2024/25)** keeps that attention but adds a **long-term memory that learns while running**, fixing the Transformer's biggest limits.

## 2. The chain of ideas
1. Transformer: attention = `softmax(QKᵀ/√d)V`. Q, K, V are exactly the *query / key / value* concepts of a **memory lookup**.
2. Researchers showed attention is an **associative memory**: store (key→value) pairs, retrieve by query. The KV-cache *is* the memory (exact, but grows with length, O(n²) compute).
3. **Linear attention / RNN revival** (Linear Transformer, RetNet, Mamba, DeltaNet...) compress that memory into a fixed-size matrix → O(n) cost, but small capacity, old info overwritten.
4. **Titans** says: memory should be *learned*. Make the memory a neural net updated by gradient descent on `||M(k) − v||²` - with surprise, momentum, forgetting - and combine it with attention.
5. **Nested Learning/HOPE**: generalize to many memories updating at different speeds → continual learning.

## 3. Component-by-component mapping
| Transformer | In Titans |
|-------------|-----------|
| Q, K, V projections | Same `W_Q, W_K, W_V`, but K,V *write* into memory, Q *reads* from it |
| Self-attention over the window | **Short-term memory** (core), kept (full or sliding-window) |
| KV-cache (grows forever) | Replaced for old tokens by the compact **neural memory** |
| FFN (stores knowledge in frozen weights) | **Persistent memory** tokens (frozen, task knowledge) + deep memory MLP (but this one is *updated at test time*) |
| Causal mask / autoregression | Same autoregressive setting preserved |
| Residual + LayerNorm | Still used around blocks |
| Context limited by window & O(n²) | Attention limited to small chunk; memory handles very long range in ~linear time |
| Weights frozen after training | Memory weights **change at inference** (learning at test time) |
| Position encoding | Order is implicitly handled by sequential memory updates + attention positions in the window |

## 4. Side-by-side comparison
| Aspect | Transformer | Linear RNN (e.g., Mamba) | Titans |
|--------|-------------|---------------------------|--------|
| Memory type | Exact KV-cache | Fixed small vector/matrix state | Deep neural net (MLP) + attention + persistent tokens |
| Cost vs length | O(n²) | O(n) | ≈ O(n) memory + small-window attention |
| Long-range recall | Great inside window, none beyond | Weak, lossy | Strong (reported to 2M+ tokens) |
| Learns at test time? | No (in-context only) | No (state update only, not gradient learning) | **Yes** (gradient-based surprise updates) |
| Forgetting | Hard cutoff by window | Decay | Learned, data-dependent gate |
| Best at | Precise short context | Speed | Long context with accuracy |

## 5. Everyday analogy to remember both
- **Transformer** = you in an exam with *open book, but only 10 pages visible at once*, and the book never changes you.
- **Titans** = same exam, but you also keep **notes in your head that automatically update with whatever surprised you**, you forget unimportant things, and you have built-in knowledge of exam format.

## 6. Quick revision checklist
Transformer: tokens → embeddings → +positions → [self-attention (Q,K,V, heads) → add&norm → FFN → add&norm]×N → (decoder: masked attn + cross attn) → softmax. Limits: O(n²), finite window, frozen weights.
Titans: attention (short) + neural memory (long, test-time learning via surprise + momentum + forgetting) + persistent tokens; combined as MAC / MAG / MAL; trained in parallel via chunked gradients + associative scan; strong on long context. Next: Nested Learning/HOPE for continual learning.

## 7. Sources to read yourself
- Vaswani et al., "Attention Is All You Need", arXiv:1706.03762
- Behrouz, Zhong, Mirrokni, "Titans: Learning to Memorize at Test Time", arXiv:2501.00663
- Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025)
- Helpful background: Jay Alammar's "The Illustrated Transformer"
