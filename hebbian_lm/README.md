# Hebbian language learner

A small language model that learns **only with local, Hebbian-family plasticity rules**: no backpropagation, no loss
function, no GPU, nothing but NumPy. It starts empty, learns words, word categories and grammar from text, keeps
learning from whatever you tell it afterwards, and answers questions from what it has learned (or says it doesn't know).

> Branch `hebbian-approach`. Everything lives in this folder. Research notes: [RESEARCH.md](RESEARCH.md).
> How it works, rule by rule: [DESIGN.md](DESIGN.md).

## Try it

```bash
pip install -r requirements.txt        # numpy only
python train.py                        # ~16 s on a laptop CPU; prints a full report, saves models/hebbian_lm.pkl
python chat.py                         # talk to it; it keeps learning from what you say
python -m unittest discover -s tests   # 23 tests, ~15 s
python real_text.py                    # sanity check on real children's books (downloads ~4 MB once)
```

## What it does (a real session, output of the trained model)

```
you> what color is the sky ?            ai > the sky is blue .              <- taught during training
you> how many legs does the bird have ? ai > the bird has two legs .
you> what color is the sand ?           ai > i do not know .                <- never heard of "sand"
you> the sand is brown .                ai > learned it (1 new word).       <- told once
you> what color is the sand ?           ai > the sand is brown .
you> how many legs does the spider have ?   ai > i do not know .
you> the spider has eight legs .        ai > learned it (1 new word).
you> how many legs does the spider have ?   ai > the spider has eight legs .
you> the flib runs .  /  a flib sleeps .  /  the flib is big .       <- a made-up word, three sentences
you> /say does the flib                 ai > does the flib eat eggs ?       <- uses it in a frame it never saw
you> /say can the flib                  ai > can the flib run ?
you> the sky is gray .                  ai > learned it (1 new word).       <- a correction
you> what color is the sky ?            ai > the sky is gray .
```

## Results

Everything below is measured by `python train.py --ablations` (30,000 training sentences, seed 0) unless stated.
The model lives in a **closed, synthetic child-English world** (121,046 grammatical sentences: singular/plural, a/an,
pronouns, agreement, present/past/progressive, negation, questions, modals, adjectives, prepositions, numbers, plus 89
facts). The grammar is exact, so "grammatical" is not a guess. 24,324 sentences, and every sentence built from a
held-out (subject, verb) or (verb, object phrase) pair, **never appear in training**.

| What | Result |
|---|---|
| Words learned | 314 (one unit each, created on first sight) |
| Word categories it invented, with no labels | 46 units; agreement with true parts of speech NMI **0.86** (fine tags) / 0.71 (coarse) |
| Grammar: is the correct sentence more probable than a version with one error? (800 pairs built from *unseen* combinations) | **98.6%** overall: agreement 97.7%, number 100%, word order 100%, subject-type 100%, verb-object fit 93.9% |
| Free generation, 400 samples from nothing | 82.5% exact training sentences, **15.8% new grammatical sentences it made up**, 1.8% outside the world (often sensible extrapolation, e.g. "my doctors walk in the park") |
| Facts taught during training, asked back (68 questions, whole answer sentence must match) | **95.6%** (colours 81%; legs/home/opposites/sounds 100%) |
| Facts told **once** as a plain statement after training, then asked | **20/20** |
| Facts it was never told (20 + 9 questions) | **29/29** answered "i do not know ." (with abstention on; 0 confident wrong answers) |
| New made-up word, 3 sentences | lands in the animal category (100%); then used correctly in 4 sentence frames 90%; old skills unchanged (99.5% → 99.5%) |
| Correcting an old fact | 1 statement flips the answer |
| Arithmetic ("two plus three is five") | 85.7% on sums it saw: this is rote memory, not arithmetic |

### What each mechanism contributes (switched off one at a time, nothing retrained)

| | minimal pairs | selection | perplexity | ungrammatical output | taught facts | told-once facts |
|---|---|---|---|---|---|---|
| full model | 0.986 | 0.939 | 18.0 | 1.0% | 0.956 | 1.00 |
| no sentence-frame memory | 0.991 | 0.955 | 16.4 | **6.3%** | 0.838 | 1.00 |
| no category grammar | 0.901 | 0.712 | **86.5** | 0.7% | 0.765 | 0.95 |
| no topic memory | 0.964 | 0.652 | 14.4 | 0.3% | **0.485** | 0.80 |
| no copy/pointer gate | 0.986 | 0.939 | 18.0 | 1.0% | **0.515** | **0.00** |
| word n-grams only | 0.909 | 0.667 | 42.6 | 4.3% | 0.000 | 0.00 |

The Hebbian-learned **categories** are what lets it handle word combinations it has never seen (perplexity 86 → 18).
The **topic memory + copy gate** are what make facts work. The **sentence frame** mostly keeps generated sentences
consistent (a sentence that began as a question ends as one); it does not help the discriminative scores.

### Stability

| training sentences | seed | train time | minimal pairs | perplexity | ungrammatical | taught facts | told-once |
|---|---|---|---|---|---|---|---|
| 5,000 | 0 | 3 s | 0.91 | 22.1 | 5.6% | 0.93 | 1.0 |
| 15,000 | 0 | 8 s | 0.97 | 18.7 | 3.2% | 0.94 | 1.0 |
| 30,000 | 0 / 1 / 2 | 16 s | 0.996 / 0.982 / 0.95 | 18.4 / 18.4 / 20.9 | 0.4% / 3.2% / 1.2% | 0.96 | 1.0 |
| 100,000 | 0 | 51 s | 0.986 | 21.5 | 0% | 0.94 | 1.0 |

Seeds change results by a few points; the verb-object "selection" score is the noisiest (0.56 to 1.0 across seeds).
More data does not keep helping (perplexity on *held-out combinations* gets slightly worse: more confidence in the
combinations it has seen).

### Real English (not the synthetic world): `python real_text.py`

Three children's books from Project Gutenberg (72k training tokens, 3,000-word vocabulary), 6,467 unseen test tokens:

| | held-out perplexity |
|---|---|
| word n-grams only (same synapses, categories off) | 245.0 |
| + Hebbian-learned category grammar | **124.1** |
| + topic gate and frame memory (the synthetic-world settings) | 238.7 (worse) |

The categories it finds among real words are recognisably parts of speech, from raw text and no labels, for example
`come go make get eat look take help find keep catch sing hear give` / `came went thought looked saw ran heard took
put asked stood felt found knew` / `little old green other next poor same city best whole tiny`. Free generation is
locally fluent and globally incoherent. Fluent: *"it was the first one that was ever made in america ."*,
*"he said ; this life is too quiet for me ."*, and a prompt of *"once upon a time"* continued as *"there was a wee
little boy who slept in a tiny trundle bed near his mother's great bed"*. Not fluent: *"then he came lower , and then
she wrote stories for the unk ."*, *"she was very pretty , and ever ever ."* (`unk` = a word outside the 3,000-word
vocabulary). Settings for this run were picked on a validation split of the training books, not on the test tokens.

## Things that did **not** work (kept in the record)

* **A staged curriculum** ("memorise simple patterns first, then richer ones"). I expected it to help; at equal data
  it made agreement worse (about 0.96-0.99 → 0.87-0.94). `python train.py --stages` is still there; the default is one mixed pass.
* **Frame memory and topic gate on real text.** Tuned on the synthetic world, they roughly double perplexity on real
  books (124 → 239). They are mechanisms for a small closed world.
* **More data.** Nothing improved past ~30k sentences on the held-out-combination tests.
* Several first versions of the question answering failed for instructive reasons (the question word "color" predicted
  the *subject nouns* of colour questions; the protocol word "answer" boosted every answer word; an additive PMI score
  saturated). DESIGN.md says what replaced them.

## Honest limitations

* **It is a statistical learner, not a reasoner.** Architecturally it is a class-based n-gram model whose classes,
  sequence cells and associations are all learned by Hebbian rules, plus a topic/copy memory. It has no long-range
  structure, no logic, no arithmetic (sums are memorised), and no understanding in any deep sense.
* **The strong numbers are in a tiny closed world I designed.** On real text it learns useful categories and beats a plain
  n-gram, but its generations are not coherent. Don't expect general English mastery from 72k tokens, or from this
  architecture at all.
* **Wrong relation, right subject.** "what color is the cow ?" can produce "the cow says moo ." because it has a
  taught fact about cows (abstention cannot catch this).
* **Counts, not replacement.** Told input is accumulated, so older evidence fades only by the belief-revision rule
  (heterosynaptic depression in contradicted cells); it flips one fact in one shot, but I did not test long
  chains of conflicting updates or many interacting facts.
* **Scale.** Word-to-word synapses are dense matrices, so memory grows as vocabulary² (about 4,000 words is comfortable;
  the real-text run uses 3,000). Pure Python/NumPy runs at roughly 10-11k tokens per second.
* **Not all of it is "Hebbian".** The core synapses and category units are. The episodic sentence buffer, the
  product-of-experts mixing, back-off smoothing, the pointer mixture and the abstention rule are engineering
  (DESIGN.md has the exact split).

## Files

| | |
|---|---|
| `hebbian_lm/model.py` | the model and all learning rules |
| `hebbian_lm/curriculum.py` | the synthetic world: exact grammar, facts, held-out splits, minimal pairs |
| `hebbian_lm/evaluate.py` | every metric above (categories, minimal pairs, generation, facts, abstention, continual, ablations) |
| `train.py`, `chat.py`, `real_text.py` | train + report, interactive chat, real-English check |
| `tests/test_hebbian_lm.py` | 23 tests: tokenizer, curriculum integrity, locality of learning, grammar, categories, facts, correction, abstention, new words, save/load, determinism |
| `RESEARCH.md`, `DESIGN.md` | literature notes; the design |

## No GitHub Actions

Training takes about 16 s and the tests about 15 s on a plain CPU, so nothing needed a remote build and no separate
build branch was created.
