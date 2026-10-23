# Design

Goal: a model that learns basic English words and grammar, **keeps learning after "training" ends**, and answers
from what it has learned, using only local, Hebbian-family learning. (Background and sources: [RESEARCH.md](RESEARCH.md).)

## Principles

1. **Local plasticity only.** A synapse changes using the activity of the unit on each side of it, plus at most a
   global scalar (competition, normalisation, a surprise/neuromodulator signal). No gradients, no loss, no labels
   except the next word that actually arrives (the "post-synaptic" activity of the word it must predict).
2. **Always learning.** There is no train/test switch inside the model. `read(..., learn=True)` updates it; a new
   word recruits a new unit; "sleep" (`consolidate()`) replays what is stored. Training is just reading a lot.
3. **Two speeds.** Word-level cells learn in one shot (fast, specific). Categories and category grammar are slow and
   statistical (general). Fast cells let it use a new word in an old sentence; the slow layer lets it use that word in
   sentences it has never seen. This is the complementary-learning-systems idea from the literature.
4. **Know the difference between evidence and guess.** Every readout is a confidence-weighted blend; the model can
   decline to answer.

## What exists, and which rule writes it

| Structure | Where | Rule (all local) |
|---|---|---|
| **Word units** | `words`, `idx` | one localist unit per word, recruited on first sight ("fast mapping") |
| **Spelling code** | `FV` | fixed hashed character n-gram / suffix code, so a new word has *some* signature before it has any context |
| **Neighbour codes** | `U[w, 0..3]` | Hebbian accumulation: word `w`'s unit accumulates the sparse random "index" code of the words 1 and 2 positions left and right of it (pre = neighbour code, post = `w`). This is the Random Indexing / distributional input to the category layer |
| **Word→next-word synapses** `S1` | dense matrix | asymmetric (STDP-like) Hebb: `S1[prev, w] += g` when `w` follows `prev` |
| **Conjunction cells** `P[n]` (n=2,3,4) | dict of cells | on first co-occurrence a cell recruits for the last n words and Hebbian-links to the next word (sequence detector of rising order). `PF[n]` is the same refined by the sentence's first word (so a sentence that began as a question stays one); `PM` is the sentence-start cell per reading mode (plain / question / answer) |
| **Topic synapses** `C[cue, w]` | dense matrix | covariance-style Hebb: `C[cue, w] += trace(cue) × surprise-gain` where `trace` is a decaying (0.9/token) activity of recently seen *content* words. Read out as an association log-ratio `log((C+ε)/(E+ε))`, `E` = expected co-activity, shrunk by `C/(C+2)`: the sliding-threshold (BCM/covariance) idea that frequent cues carry little information |
| **Surprise gain** | scalar | three-factor rule: the update of the lexical cells and `C` is scaled by `clip(−log p_est / 2, …, 4)` where `p_est` comes from the word-level synapses *before* the update. Surprising events are written harder |
| **Pointer synapses** `echo_*` | two dicts | counts of how often the next word repeated a recently active word that sat in a given *slot* (the word before it), in a given context (mode, prev2, prev1). Learned rate = P(copy \| candidate with that slot present); this is how an answer re-states the question's subject, including subjects it has never copied before |
| **Category units** `Wc` | 96 prototypes | competitive Hebbian learning on each word's context signature: winner takes all, instar/Oja-style update `Wc[k] += η(x − Wc[k])`, ART-style *recruitment* of a fresh unit when the best match is below a vigilance threshold (so new kinds of word can create new categories) |
| **Sleep** `consolidate()` | every 1,500 sentences | replay the stored word signatures through the same rule, settle the prototypes to the mean of their members (the fixed point of the instar rule), merge near-duplicates, free empty units, reassign every word |
| **Episodic buffer** | `episodes` | the last distinct sentences (bounded). Replayed in sleep to rebuild the category-level grammar, so it always matches the current categories |

### Reading out a prediction (`predict`)

1. **Word-level** distribution: unigram → bigram (`S1`) → `P[2]`, `PF[2]`, `P[3]`, `PF[3]`, `P[4]` … each cell blended
   with the shorter context by Witten-Bell weights (`n / (n + wb·types)`) or absolute discounting (`smoothing="ad"`,
   used for real text).
2. **Category-level** distribution: the same back-off chain over *category* n-grams (orders 1-4, plus first-word frame),
   then emitted back to words through the category's membership. This is what generalises to unseen combinations.
3. The two are combined **geometrically** (product of experts) with weight `class_mix` (0.7), **per candidate word**:
   a context word or a candidate word that has been seen only a few times has an unreliable category, so its weight
   shrinks (`class_trust`, `cand_trust`).
4. **Topic gate**: multiply by `exp(gain × evidence)` for *content* words only (rare words; grammar words are never
   gated), where evidence is the summed association of the question/context cues, positive part only.
5. **Pointer mixture**: with the learned copy probability, move that much mass onto recently seen words in the
   right slot.

### Continual learning

* **New word**: recruited on sight; provisional category from the first context; settled in the next sleep.
* **New fact**: lexical cells + topic synapses written in one shot (surprise-boosted); answered immediately.
* **Correction** (`tell`, `teach_qa`): a confident cell contradicted by *told* input loses a share of its old synapses
  (heterosynaptic depression), both in the lexical cells and, among words of the same category, in the topic synapses.
  Bulk `read`/`read_lines` never does this, so ordinary training statistics are untouched.
* **Don't know** (`ask(abstain=True)`): refuse if the question has a word it never met, or if a content word of its
  own answer (not copied from the question) has topic evidence below `min_support`.
* **Forgetting**: none by default. `recency > 0` makes newer updates count more (not used in the reported results).

## What is Hebbian and what is engineering

Honest split, so the label is not oversold:

* **Hebbian / local**: word units, `S1`, conjunction cells, `C`, neighbour accumulators, category units (competitive
  instar rule, recruitment, merging), pointer counts, surprise gain (a three-factor rule), heterosynaptic depression.
* **Engineering / not a neural rule**: the episodic sentence buffer and its replay into class n-gram tables; Witten-Bell
  and absolute-discount back-off; the geometric mixing of experts; the topic gate's functional form and cap; the
  abstention test; the `question/answer/:` protocol tokens being transparent; the synthetic curriculum itself.

The point of the project is that every piece of *knowledge* (words, categories, order, facts) is written by local
rules; the readout formulas are hand-built.

## Evaluation design

`curriculum.py` generates a closed world whose grammar is enumerated exactly (121k sentences), so every sentence the
model produces can be classed as *seen in training*, *new but grammatical*, or *outside the world*. Held-out items are
chosen by hashing **(subject, verb)** and **(verb, object phrase)** pairs, so a held-out sentence needs a word
combination the model never saw; the nouns and verbs themselves still appear elsewhere. Minimal pairs
(agreement, number, selection, subject-type, word order) are built from held-out sentences and verified to be
outside the language. Facts split into taught facts and held-out facts (told only after training).

Two experiments in the history of this project were wrong and are worth remembering: holding out a (verb, noun) pair
for `eat` removed *all* evidence that a food noun is edible (a test no model could pass), and a held-out sentence
that was also a taught fact leaked into training (found by a unit test).

## Main parameters (`Config`)

| | default | meaning |
|---|---|---|
| `max_classes` | 96 | category units available (46 used on the synthetic world, 95 on real text) |
| `vigilance` / `merge_threshold` | 0.3 / 0.85 | when to recruit a new category / merge near-duplicates |
| `class_order` | 5 | longest context (4 previous words) |
| `class_mix` | 0.7 | weight of the category expert |
| `class_trust`, `cand_trust` | 1, 8 | discount of the category expert for rarely seen words |
| `topic_decay`, `topic_gain`, `topic_cap` | 0.9, 0.75, 12 | content-trace decay; strength and cap of the topic gate |
| `surprise_ref`, `surprise_max` | 2, 4 | surprise-gain scale and ceiling |
| `revise_decay` | 0.7 | share of contradicted synapses removed by a correction |
| `min_support` | 2.5 | abstention threshold |
