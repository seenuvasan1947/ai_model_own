# ai_model_own

Experiments with small language models. Each approach lives on its own branch.

* **`hebbian-approach`** — [`hebbian_lm/`](hebbian_lm/): a language learner that uses only local, Hebbian-family
  learning rules (no backpropagation, no GPU, NumPy only). Learns words, word categories and grammar from text, keeps
  learning from what you tell it, and answers from what it has learned or says it doesn't know.
  Start with [`hebbian_lm/README.md`](hebbian_lm/README.md).
