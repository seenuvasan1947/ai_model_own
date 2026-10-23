# Hebbian Semantic Workspace

> **Branch `hebbian-v3`: [V3.md](V3.md)** — learned counting and arithmetic, word meaning (events with roles, named things, core vocabulary), grammar as knowledge, retelling in its own words, and the wrong-relation fix.
>
> **Release 1.0.0 — [RELEASE.md](RELEASE.md)** (trained model in `release/v1.0/`, run `python chat.py --workspace release/v1.0/workspace`).
>
> **This is v2 (branch `hebbian-v2`).** What changed from the working v1 and the new numbers are in **[V2.md](V2.md)**; v1 is unchanged on branch `hebbian-approach`.

A language system that is **not a transformer** and learns without backpropagation. It reads text, looks up words it
has never met in a dictionary, turns sentences into stored facts, answers questions from those facts as grammatical
sentences, says "I do not know" when it does not, and keeps learning from whatever you tell it.

It combines four parts, each doing the job it is good at:

```
 text ──► vocabulary & grammar rule engine ──► comprehension ──► propositions ──► SEMANTIC MEMORY
          (dictionary, morphology, tagger)     (who/what/where)   (subject, relation, value)   exact storage, Hebbian
               │ unknown word?                                                                 strengthening, rivals,
               │ look it up, learn it                                                          recency, "is a" inheritance
               ▼                                                                                        │
        HEBBIAN LANGUAGE LAYER  ◄──────────────── realisation (agreement, plurals, a/an) ◄────────────────┘ retrieval
        grammar, word categories,
        fluency; learns from everything
```

* **Vocabulary and grammar rule engine** (`lexicon.py`, `comprehension.py`): morphology rules (plurals, tenses,
  irregulars), a part-of-speech tagger that settles ambiguous words from context, noun-phrase and verb-group
  chunking, question patterns, pronoun resolution. A bundled dictionary works offline; optional WordNet gives ~150,000
  words with definitions, kinds, "is a" chains and opposites.
* **Semantic memory** (`semantic.py`): each fact is an explicit edge (subject, relation, value). Reading it again
  strengthens it; values that cannot both be true compete (two colours, two homes); told corrections beat old beliefs;
  if nothing is stored about a word, it inherits through "is a" links.
* **Hebbian language layer** (`model.py`): learns grammar, word categories and word order with local Hebbian-family rules
  only (no gradients). It gives every new word a starting category from the dictionary, scores fluency, and keeps
  learning from what is read.
* **Realisation**: answers are built with the morphology rules, so agreement, plurals and a/an are right by construction.

> Branch `hebbian-approach`. **How to run and use it: [USAGE.md](USAGE.md).** Design: [DESIGN.md](DESIGN.md).
> Literature notes: [RESEARCH.md](RESEARCH.md).

## Try it

```bash
pip install -r requirements.txt     # numpy
python train.py                     # ~20 s; trains and prints a full report
python chat.py                      # talk to it
python -m unittest discover -s tests   # 99 tests
```

## A real session (offline dictionary)

```
you> what color is the sky ?                        ai > the sky is blue .
you> is a zebra an animal ?                         ai > yes , the zebra is an animal .          <- the dictionary knows zebra -> equine -> animal
you> what is a zebra ?                              ai > a zebra is a kind of equine . zebra means : a striped horselike animal of Africa .
you> the zebra is brown . it eats the grass .       ai > ok , learned it . ( looked up zebra, eat )    <- "it" = the zebra
you> what does the zebra eat ?                      ai > the zebra eats the grass .
you> what color is the sparrow ?                    ai > i do not know .                          <- never told: it does not guess
you> a flib is a bird .                             ai > ok , learned it . ( learned the word flib )   <- a word no dictionary has
you> is a flib an animal ?                          ai > yes , the flib is an animal .            <- flib -> bird -> animal
you> what color is the glorp ?                      ai > i do not know the word glorp . what is it ?
you> the sky is gray .                              ai > ok , learned it . ( looked up gray )
you> is the sky blue ?                              ai > no , the sky is gray .                   <- the correction replaced the old belief
you> what is seven plus eight ?                     ai > seven plus eight is fifteen .
you> tom has a red ball . anna has a blue ball .   <- two separate balls (ball#1, ball#2)
you> what color is tom's ball ?                     ai > tom's ball is red .
you> what color is the ball ?                       ai > which ball ? anna's ball or tom's ball ?
you> the blue one                                   ai > anna's ball is blue .
/check the dogs runs .    ->  agreement: plural subject 'the dogs' needs 'run'
/fix does the cat runs ?  ->  does the cat run ?          /fix a dogs run .  ->  a dog runs .
```

## Results

Everything below is measured by `python train.py --ablations` (30,000 training sentences, seed 0, bundled dictionary) unless
stated. The test world is a **closed, synthetic child-English world** with an exact grammar (121,046 grammatical
sentences: singular/plural, a/an, pronouns, agreement, three tenses, negation, questions, modals, adjectives,
prepositions, numbers) plus 89 facts. 24,324 sentences, and everything built from held-out (subject, verb) or
(verb, object) pairs, were **never seen in training**.

### New in the workspace

| Test | Result |
|---|---|
| **A. Storage and retrieval.** 10 → 100 facts (colour, sound, home, legs, food) about animals the language layer never saw, told once, then every fact asked | **workspace 100% at 10, 30, 60 and 100 facts**; language layer alone 30% / 40% / 47% / 47% (told once or three times, with or without a dictionary). By relation at 100 facts, language layer alone: sound 100%, home 75%, legs 55%, colour 0%, food 0% (it has no practised "eats" relation, and colours of unseen animals clash with the random colours in its training sentences) |
| **B. Words never met.** 10 dictionary words absent from training (zebra, sparrow, wolf …) | with the dictionary a new word lands in the right category **before any sentence: 10/10** (0/10 without). With the full WordNet the same idea works on 64% of a harder 42-word set (bundled dictionary 93%): WordNet's first-sense heuristics are noisier. "is a X an animal?" and "what is a X?" correct 10/10 right after one sentence |
| **C. Pronouns across sentences.** "the zebra is brown . it eats the grass . the sparrow is small . it lives in the tree ." then 4 questions, 40 stories | 157/160 = **98.1%** |
| **D. Reasoning through "is a".** 10 yes/no questions incl. a word defined only by the user and chains of two links | 10/10 |
| **F. Separate entities.** "tom has a red ball . anna has a blue ball ." (60 random stories: 8 things, 6 names, 10 colours), then possessive questions, an ambiguous "the ball", a follow-up answer, "what does X have", counts, and telling the same thing twice | instances on: **100%** on all six checks; instances off (the old behaviour): 0% on the first four (it collapses to one ball), 100% on the two counts. Synthetic stories written for this feature, so 100% shows the mechanism works, not that it copes with free English |
| **E. Grammar repair.** one agreement/number error per sentence | exact repair: agreement **98.5%** (406 sentences), number **100%** (77); false alarms on correct sentences 1/400 |

How to read A honestly: the semantic memory is exact by construction, so 100% is expected; the point is the comparison with
the language layer, which stores facts diffusely, needs practised relations and cannot cope with a relation it has not seen.
The two layers do different jobs, which is why the system uses both.

### The Hebbian language layer (unchanged jobs: grammar, categories, fluency, continual learning)

| What | Result |
|---|---|
| Word categories it invented with no labels | 49 units; agreement with true parts of speech NMI **0.86** (fine tags) / 0.74 (coarse) |
| Grammar: correct sentence more probable than a version with one error (800 pairs built from *unseen* combinations) | **99.0%**: agreement 98.7%, number 100%, word order 100%, subject-type 100%, verb-object fit 93.9% |
| Free generation, 400 samples | 85.0% training sentences, **14.5% new grammatical sentences it made up**, 0.5% outside the world |
| Facts taught during training, asked back (whole answer must match) | **95.6%**; facts told **once** as a plain statement afterwards: **20/20** |
| "I do not know" (language layer) | never-told facts 20/20 and 9/9 refused, 0 confident wrong answers; false refusals of taught facts 1.5% |
| A made-up word taught with 3 sentences | right category 100%; then used correctly in 4 sentence frames 87%; old skills 99.3% → 99.3% |
| Correcting an old fact | 1 statement |

What each mechanism of the language layer contributes (switched off one at a time):

| | minimal pairs | selection | perplexity | ungrammatical | taught facts | told-once |
|---|---|---|---|---|---|---|
| full | 0.990 | 0.939 | 18.5 | 0.3% | 0.956 | 1.00 |
| no sentence-frame memory | 0.993 | 0.970 | 16.8 | **6.0%** | 0.897 | 1.00 |
| no category grammar | 0.900 | 0.712 | **86.6** | 0.7% | 0.721 | 1.00 |
| no topic memory | 0.975 | 0.773 | 15.0 | 1.0% | **0.485** | 0.80 |
| no copy/pointer gate | 0.990 | 0.939 | 18.6 | 0.3% | **0.500** | **0.00** |
| word n-grams only | 0.909 | 0.667 | 42.6 | 4.3% | 0.000 | 0.00 |

### Real English

* **Perplexity** on three children's books from Project Gutenberg (72k training tokens, 6,467 unseen test tokens): word
  n-grams 245 → with the Hebbian-learned categories **124** (with WordNet attached: 122). The categories it finds among real words are
  recognisably parts of speech (`come go make get eat look take …` / `came went thought looked saw ran heard took …`).
  Generation is locally fluent and globally incoherent (*"it was the first one that was ever made in america ."* but also
  *"she was very pretty , and ever ever ."*). Run `python real_text.py`.
* **Comprehension on real fiction** (v1 numbers; v2 improves them, see [V2.md](V2.md)): on 300 random short sentences of one
  Gutenberg book the v1 rule engine extracted at least one proposition from **58%**. On 51 hand-labelled sentences v1 found 28–35%
  of the labelled propositions, v2's clause reader 81–94% (with WordNet; 63–68% with the bundled dictionary). Still a rule
  system: ellipsis, nested relatives, idioms and long-distance dependencies are not handled.

## What did not work, and what is still weak

* A **staged curriculum** (simple patterns first) made grammar slightly worse at equal data; the default is one mixed pass.
* The language layer's **frame memory and topic gate**, tuned on the synthetic world, roughly double perplexity on real books.
* **More data** than ~30k synthetic sentences stopped helping on held-out combinations.
* In development the dictionary's *first-sense* guesses hurt: with the full WordNet, words like "legs", "eats" and "blue" had
  odd readings until readings were ranked by usage frequency; WordNet remains noisier than the small curated dictionary.
* **The workspace is a rule-based comprehension engine plus an exact graph; only the language layer is Hebbian
  learning.** The semantic memory uses Hebbian-style strengthening and lateral inhibition on graph edges, not a neural net.
* **Entities are created by simple rules only.** A particular thing gets its own instance when an indefinite noun is the
  object of a verb whose subject is a name or an instance ("tom has a ball", "anna sees a dog"), or is possessed
  ("tom's ball", "his ball"). Generic statements ("the bird has a beak", "a dog is an animal", "the zebra is brown") stay at
  the level of the kind. "the X" with several candidates means the latest one mentioned when reading, and triggers a
  clarifying question when asked. Telling the same owned thing again (same owner, noun, adjectives) strengthens it instead
  of making a second one, so "tom has a ball . tom has a ball ." is one ball. Not handled: plural instances ("two balls"
  are the kind), "whose ball is red ?", `my`/`your`, possessives of plurals, instances of people ("a boy").
* **Wrong relation, right subject** is still possible in the language layer alone ("what color is the cow ?" → "the cow says moo .");
  the workspace avoids it for patterns it parses, but it has nothing equivalent for questions it cannot parse (it then falls
  back to the language layer).
* Answers are one fixed sentence shape per relation (not paraphrased); there is no dialogue memory beyond pronoun resolution.
* Memory of the language layer grows as vocabulary² (about 4,000 words is comfortable); pure Python/NumPy runs at roughly
  10k tokens per second.
* The fluency score is reported with each answer but not used to choose among alternatives.

## Files

| | |
|---|---|
| `hebbian_lm/model.py` | the Hebbian language layer |
| `hebbian_lm/lexicon.py`, `resources/mini_lexicon.json` | dictionary backends, morphology rules, closed-class words |
| `hebbian_lm/comprehension.py`, `semantic.py`, `workspace.py` | comprehension, semantic memory, the workspace |
| `hebbian_lm/curriculum.py`, `evaluate.py`, `evaluate_workspace.py` | the synthetic world and every benchmark above |
| `train.py`, `chat.py`, `real_text.py`, `fetch_wordnet.py` | train + report, chat, real-English check, optional WordNet download |
| `hebbian_lm/entities.py` | the instance registry (`ball#1`, owner, qualifiers) |
| `hebbian_lm/reader.py`, `notes.py`, `evaluate_real.py` | v2: clause reader for real text, note keeper, real-text evaluation |
| `tests/` | 99 tests (tokenizer, curriculum integrity, locality of learning, grammar, categories, facts, correction, abstention, morphology, parsing, memory, workspace scenarios, grammar repair, persistence) |

No GitHub Actions or separate build branch was needed: training takes about 20 s and the tests about 25 s on a plain CPU.
