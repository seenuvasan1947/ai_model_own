# How to run and use it

Everything here runs on a plain CPU with Python 3.9+ and NumPy. No GPU, no PyTorch, no internet needed (one optional
download makes the dictionary much bigger).

## 1. Install

```bash
git clone https://github.com/seenuvasan1947/ai_model_own
cd ai_model_own
git checkout hebbian-approach
cd hebbian_lm
pip install -r requirements.txt          # numpy
```

## 2. Build and talk (two commands)

```bash
python train.py        # ~20 s. Trains the language layer, builds models/workspace/, prints a full report
python chat.py         # talk to it
```

`train.py` writes `models/hebbian_lm.pkl` (language layer alone) and `models/workspace/` (what `chat.py` opens:
semantic memory, dictionary additions and the language layer). To see only the quick version of the training run:
`python train.py --quick`.

### Optional: the full English dictionary (WordNet, ~150,000 words)

```bash
python fetch_wordnet.py                        # downloads ~10 MB into data/wordnet/ (Princeton WordNet 3.0)
python train.py --dictionary wordnet           # train with it
python chat.py                                 # picks WordNet up automatically (or --dictionary mini to ignore it)
```

Without it the system uses a bundled 390-word dictionary that covers the demo world (animals, people, foods,
colours, sounds, places, common verbs and adjectives with their opposites).

## 3. Using the chat

Type a sentence ending in a full stop to **teach** it, or a question ending in `?` to **ask**.

```
you> the zebra is brown . it eats the grass .          # teach (it understands "it" = the zebra)
ai > ok , learned it . ( looked up zebra, eat )
you> what does the zebra eat ?
ai > the zebra eats the grass .
you> what color is the sparrow ?                        # never told: it says so instead of guessing
ai > i do not know .
you> a flib is a bird .                                 # a word the dictionary does not know: explain it in a sentence
ai > ok , learned it . ( learned the word flib )
you> is a flib an animal ?                              # it reasons through "is a": flib -> bird -> animal
ai > yes , the flib is an animal .
you> what color is the glorp ?                          # unknown word in a question: it asks
ai > i do not know the word glorp . what is it ?
you> the sky is gray .                                  # a correction: new told fact replaces the old one
you> what color is the sky ?
ai > the sky is gray .
```

| Type | What happens |
|---|---|
| a statement (`the dogs eat bones .`) | understood, turned into facts, stored; unknown words are looked up in the dictionary; the language layer also learns the grammar of the sentence |
| a question (`what does the dog say ?`) | answered from memory, the dictionary and "is a" inference, as a grammatical sentence; `i do not know .` if nothing supports an answer |
| `/word zebra` | what the dictionary and memory know: meaning, kind, "is a kind of" chain, synonyms, opposites, stored facts |
| `/facts zebra` | everything stored about a noun, with strengths |
| `/check the dogs runs .` | grammar report: agreement, a/an, singular/plural, unknown words |
| `/fix a dogs run .` | the sentence with the rule-based fixes applied (`a dog runs .`) |
| `/lm what does the cow say ?` | ask the language layer alone (its diffuse knowledge from training) |
| `/say the zebra` / `/free 5` | the language layer continues a sentence / makes up sentences |
| `/classes` | the word categories the language layer invented for itself |
| `/stats`, `/save [dir]`, `/help`, `/quit` | housekeeping. Save before you leave if you want the new knowledge kept |

### What it understands

Statements: `X is ADJ` · `X is a/an Y` · `X VERB (Y)` incl. negation, past and plural · `X lives/walks/sleeps in/on/to the Y` ·
`X has N legs` · `X says SOUND` · `there are N Xs` · `the opposite of X is Y` · `X can VERB` · coordination
(`the cat and the dog play`) · `he/she/it/they` referring to the previous sentence · contractions (`didn't`).

Questions: `what color is X` · `what does X say/eat/<verb>` · `where does X live` / `where is X` · `how many N does X have` ·
`how many Xs are there` · `who <verb>s Y` · `what is a X` / `what does X mean` · `what is the opposite of X` ·
yes/no (`is X ADJ`, `is X a Y`, `does X VERB Y`, `can X VERB`) · arithmetic (`what is five plus eight`).

Not understood (it says so): relative clauses, passives, "if/because", questions about time, metaphor.

## 4. Using it from Python

```python
from hebbian_lm.lexicon import Lexicon
from hebbian_lm.workspace import Workspace

ws = Workspace.open("models/workspace", lexicon=Lexicon())   # or Lexicon.default() to use WordNet if downloaded
ws.read("the zebra is brown . it eats the grass .")          # understand + store (+ the language layer learns grammar)
print(ws.ask("what color is the zebra ?").text)               # -> the zebra is brown .
r = ws.ask("is a zebra an animal ?")                          # a Reply with .text .kind .confidence .inferred .note .fluency
print(r.text, r.inferred, r.note)
print(ws.describe("zebra"))                                   # meaning, kind, hypernyms, opposites, known facts
print(ws.check("the dogs runs ."))                            # [Issue(kind='agreement', ...)]
print(ws.correct("does the cat runs ?"))                      # does the cat run ?
ws.save("models/workspace")                                   # keep what it learned
```

Without any trained model (dictionary + rules + memory only): `Workspace(model=None, lexicon=Lexicon())`.

The language layer on its own (`hebbian_lm.HebbianLM`):

```python
from hebbian_lm import HebbianLM
lm = HebbianLM.load("models/hebbian_lm.pkl")          # trained by train.py
lm.tell("the flib runs .")                             # learn from a sentence (a correction overturns old beliefs)
lm.ask("what color is the sky ?", abstain=True)        # 'i do not know .' when unsupported
lm.logprob("the dogs run .") > lm.logprob("the dogs runs .")      # grammar as probability
" ".join(lm.generate("the flib"))                      # continue a sentence
lm.seed_category("zebra")                              # place a new word using the dictionary + spelling only
lm.categories()                                        # the word categories it invented
```

The dictionary and rules (`hebbian_lm.lexicon`): `Lexicon().analyze("mice")` → lemma `mouse`, noun, plural ·
`.is_a("zebra","animal")` · `.antonyms("big")` · `.define("spider")` · `.learn("flib","n",kind="animal",hypernyms=["animal"])` ·
`inflect_noun("wolf","pl")` → `wolves` · `inflect_verb("fly","3sg")` → `flies`.

## 5. Training options

```bash
python train.py --docs 100000          # more training sentences (≈50 s)
python train.py --dictionary none      # no dictionary (language layer only)
python train.py --stages               # staged curriculum (simple → rich); in my tests it did not help
python train.py --ablations            # also switch each mechanism off in turn and re-measure
python train.py --report report.json   # write all numbers as JSON
```

## 6. Check it works

```bash
python -m unittest discover -s tests   # 47 tests, ~25 s (WordNet tests are skipped if it is not downloaded)
python real_text.py                    # the same language layer on real children's books (downloads ~4 MB once)
python real_text.py --dictionary wordnet
```

## 7. Where things are

| | |
|---|---|
| `hebbian_lm/model.py` | the Hebbian language layer: all learning rules |
| `hebbian_lm/lexicon.py` | dictionary (bundled + WordNet), morphology rules, closed-class words |
| `hebbian_lm/comprehension.py` | tagger, chunker, sentence → facts, question → query, pronoun resolution |
| `hebbian_lm/semantic.py` | the semantic memory (graph, Hebbian strengthening, rivals, inheritance) |
| `hebbian_lm/workspace.py` | ties it together: reading, asking, answers as grammatical sentences, grammar check/fix |
| `hebbian_lm/curriculum.py`, `evaluate*.py` | the synthetic test world and every benchmark |
| `tools/build_mini_lexicon.py` | regenerates the bundled dictionary |

## 8. Troubleshooting

* `models/workspace/ not found` → run `python train.py` first.
* It answers `i do not know .` to something you taught → check `/facts <noun>`; it matches nouns by their dictionary
  form (`dogs` and `dog` are the same fact), and a word it cannot parse as a noun phrase is skipped.
* `i did not understand the question .` → the question is outside the patterns in section 3; rephrase simply.
* Everything is lowercase internally; capital letters only help it recognise names in the middle of a sentence.
