# Roadmap: Transformer → Titans (A to Z)

## Papers covered
| # | Paper | Authors | Year | arXiv |
|---|-------|---------|------|-------|
| 1 | **Attention Is All You Need** (the Transformer) | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google) | 2017 | 1706.03762 |
| 2 | **Titans: Learning to Memorize at Test Time** | Behrouz, Zhong, Mirrokni (Google Research) | 2024/25 | 2501.00663 |

> Note: "continue learning" is interpreted here as the Titans paper (learning continues *while the model is being used*, i.e. at test time). Its follow-up, **Nested Learning / HOPE** (Behrouz et al., 2025), is the paper literally about *continual learning*. It is summarised at the end of file 04 and in file 05. If you meant only that one, tell me and I will expand it.

## Reading order
1. `01_transformer_basics.md` - why Transformers exist, tokens, embeddings, attention (Q/K/V), multi-head, positions
2. `02_transformer_architecture.md` - encoder, decoder, masking, residuals, LayerNorm, FFN, training, results, limits
3. `03_titans_ideas.md` - the memory problem, neural long-term memory, surprise, momentum, forgetting
4. `04_titans_architecture.md` - persistent memory, MAC / MAG / MAL, parallel training, experiments, limits, Nested Learning
5. `05_connection.md` - how the two papers connect, side-by-side comparison, big picture

## Mini glossary (for freshers)
- **Token**: a small piece of text (word or word-part). "unhappy" might be `un` + `happy`.
- **Embedding**: a list of numbers (a vector) that represents a token's meaning.
- **Vector / matrix**: a list of numbers / a table of numbers.
- **Parameters (weights)**: the numbers the model learns during training.
- **Training vs inference (test time)**: training = learning from data; inference = using the finished model.
- **Context window**: how many tokens the model can look at at once.
- **Gradient**: tells which direction to change weights to reduce an error (loss).
- **Loss**: a number measuring how wrong the model is.
- **Softmax**: turns any list of numbers into probabilities that sum to 1.
- **Quadratic cost O(n²)**: if the text gets 10x longer, the work gets 100x bigger.
