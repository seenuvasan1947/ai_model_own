# Release 1.0.0

First release of the Hebbian Semantic Workspace: a language system that is not a transformer, learns without backpropagation,
and reads, remembers, forgets and answers.

**Get it:** branch `release-v1.0.0` (frozen at the release; `hebbian-v2` may move on). **Trained model:** `release/v1.0/workspace/` (12 MB, checksums in
`release/v1.0/SHA256SUMS`). **Run it:** `python chat.py --workspace release/v1.0/workspace` (needs only Python 3.9+ and numpy).
**Rebuild it yourself:** `python train.py` (about a minute on a CPU). Full guide: [USAGE.md](USAGE.md).

## What is in this release

| Part | What it does | How it learns |
|---|---|---|
| Hebbian language layer (`model.py`) | grammar, word categories, word order, fluency, generates grammatical sentences | local Hebbian-family rules, no gradients; keeps learning from every sentence |
| Vocabulary and grammar rule engine (`lexicon.py`, `comprehension.py`, `reader.py`) | looks up unknown words (bundled 390-word dictionary, optional WordNet), tags, splits long sentences into clauses, resolves pronouns and names | hand-written rules |
| Semantic memory (`semantic.py`, `entities.py`) | exact facts, separate things of the same kind (`tom's ball`), corrections, "is a" inference | strengthening on re-reading and on use; rivals inhibit each other; forgetting with a half-life that grows with use |
| Note keeper (`notes.py`) | keeps only the important points as notes (new facts, topics, what you asked about, flagged sentences), as gists or key terms; finds them again by cue | Hebbian word-to-note links strengthened by use; notes fade slowly |
| Answers (`workspace.py`) | grammatical answers built by morphology rules, worded by how sure the memory is; summaries ("what did X do ?") | – |

## What it can and cannot do (measured; see V2.md for the tables)

* Plain statements and the question shapes in USAGE.md: 100% on the synthetic benchmarks (store/retrieve 10–100 facts, 157/160 pronoun
  questions, 10/10 reasoning through "is a", 98.5% grammar repair, separate entities).
* Forgetting and notes: unused facts fade on a graded curve, used ones stay, a faded fact is recovered by reading the matching note
  again, and it does not invent answers for things never told.
* **Real prose is only partly understood.** On 51 hand-labelled Gutenberg sentences the reader finds 81–94% of the labelled
  propositions with WordNet (63–68% with the bundled dictionary), but its output contains mistakes. In a hand test on a
  12-sentence paragraph about the Sepoy Mutiny, 12 of 27 questions asked from different angles were answered usefully (5 of them
  only by finding the original sentence in the notes); 6 were answered "i do not know" although the fact was stored, and 3
  question types ("what happened after ...", "what happened in 1858") are not understood.
* Arithmetic is a built-in rule, not something it learns by repetition; teaching the language layer a new operation repeatedly did not make it generalise.
* Questions about time and cause ("when", "why", "what happened") are answered only by finding the original sentence in the notes.
* It generates grammatical but not coherent sentences.

## Checking the release

```bash
cd release/v1.0 && sha256sum -c SHA256SUMS && cd ../..    # the model files are as released
python -m unittest discover -s tests                       # 82 tests, about 30 s (WordNet tests are skipped without data/wordnet)
python train.py                                            # reproduces the benchmark tables
```

## Versions

* `hebbian-approach`: the earlier working version (language layer + workspace + separate entities), unchanged.
* `release-v1.0.0` (= `hebbian-v2` at release time): this release (real-text reader, forgetting, selective notes, own-words answers).
