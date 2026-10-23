# ai_model_own

Experiments with small language models. Each approach lives on its own branch.

* **`hebbian-approach`** — [`hebbian_lm/`](hebbian_lm/): a "Hebbian Semantic Workspace", not a transformer. A
  dictionary and grammar rule engine, a comprehension engine, an exact semantic memory and a language layer that learns
  with local Hebbian-family rules only (no backpropagation, no GPU, NumPy only). It reads with understanding, looks up
  unknown words, answers as grammatical sentences, says "I do not know" when it doesn't, and keeps learning.
  Overview: [`hebbian_lm/README.md`](hebbian_lm/README.md). **How to run and use it: [`hebbian_lm/USAGE.md`](hebbian_lm/USAGE.md).**
