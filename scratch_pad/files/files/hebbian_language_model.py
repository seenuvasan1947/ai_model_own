"""
Hebbian / Co-occurrence Based Toy Language Model
=================================================

This is a small research/education prototype exploring an alternative to
backprop-trained embeddings: a hand-structured vector space (words are
placed into semantic/grammar "regions" up front) that is then strengthened
using a local Hebbian-style update rule based purely on co-occurrence
statistics from a small corpus. No gradients, no backpropagation.

It supports:
  1. Manually defined regions (POS-like categories: NOUN, VERB, ADJ, DET, etc.)
  2. Local Hebbian learning: "neurons that fire together, wire together"
     -> if two words co-occur in a sentence, their connection strengthens.
  3. Continuous/incremental learning: you can feed new sentences at any time
     and only the relevant weights update (no global retraining).
  4. Simple generative sampling: given a start word, walk the strengthened
     graph probabilistically to produce a new sentence, using the region
     structure to keep grammar roughly sane.

WHAT THIS IS:   A transparent, inspectable toy that demonstrates the
                mechanism you're describing, so you can see its strengths
                and its ceiling firsthand.
WHAT THIS ISN'T: A competitor to transformer LLMs. Its generation quality
                is limited by design -- that limitation is itself the
                point, and is discussed at the bottom of this file.

Run it directly:
    python hebbian_language_model.py
"""

import random
import json
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. MANUALLY DEFINED VECTOR SPACE REGIONS (this replaces random init)
# ---------------------------------------------------------------------------
# Instead of random embedding vectors refined by backprop, we hand-assign
# every word to a grammatical/semantic region. Words in the same region
# share a "type" that constrains how they can be used during generation.

REGIONS = {
    "DET":  ["the", "a", "this", "that"],
    "NOUN": ["cat", "dog", "man", "woman", "city", "book", "river", "sun",
             "car", "tree", "child", "market", "mountain", "story"],
    "VERB": ["runs", "sees", "reads", "loves", "chases", "finds", "builds",
             "writes", "watches", "crosses"],
    "ADJ":  ["big", "small", "old", "young", "quiet", "bright", "quick",
             "gentle", "ancient", "hidden"],
    "PREP": ["in", "near", "under", "over", "beside"],
}

WORD_TO_REGION = {w: region for region, words in REGIONS.items() for w in words}

# A simple hand-designed grammar template library: sequences of regions
# that are considered valid sentence "shapes". This is the scaffold the
# model is allowed to generate within -- analogous to feeding it grammar
# rules directly instead of hoping backprop discovers them from scratch.
SENTENCE_TEMPLATES = [
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "PREP", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "DET", "NOUN"],
]


# ---------------------------------------------------------------------------
# 2. THE HEBBIAN MODEL
# ---------------------------------------------------------------------------
class HebbianLanguageModel:
    def __init__(self, learning_rate=0.1, decay=0.001):
        """
        learning_rate: how much a co-occurring pair strengthens per exposure
        decay: small forgetting factor applied globally each update, so
               unused connections slowly fade (prevents unbounded growth
               and mimics biological synaptic decay)
        """
        self.lr = learning_rate
        self.decay = decay
        # weights[(word_a, word_b)] = connection strength (float)
        self.weights = defaultdict(float)
        # region_strength[region][word] = how "activated" this word's
        # region-membership is, based on how often it's been seen
        self.word_freq = defaultdict(int)
        self.vocab = set(WORD_TO_REGION.keys())

    # --- Hebbian update rule: strengthen co-occurring words -------------
    def hebbian_update(self, sentence_words):
        """
        For every pair of words that co-occur within a sentence (within a
        small window), increase their connection weight. This is purely
        local: each update only touches the two words involved, there is
        no global error signal and no backward pass.
        """
        window = 3
        n = len(sentence_words)
        for i, w1 in enumerate(sentence_words):
            self.word_freq[w1] += 1
            self.vocab.add(w1)
            for j in range(i + 1, min(i + 1 + window, n)):
                w2 = sentence_words[j]
                pair = (w1, w2)
                # Hebbian strengthening: fire-together-wire-together
                self.weights[pair] += self.lr
                # Symmetric reverse connection too (bidirectional association)
                self.weights[(w2, w1)] += self.lr * 0.5

        # Global mild decay -- "use it or lose it" (synaptic pruning)
        for pair in list(self.weights.keys()):
            self.weights[pair] *= (1 - self.decay)
            if self.weights[pair] < 1e-4:
                del self.weights[pair]

    def learn_sentence(self, sentence: str):
        """Feed one new sentence into the model. Fully incremental --
        this can be called at any time, in any order, forever. There is
        no 'training phase' vs 'inference phase' split like backprop
        models have."""
        words = sentence.lower().rstrip(".").split()
        self.hebbian_update(words)

    def learn_corpus(self, sentences):
        for s in sentences:
            self.learn_sentence(s)

    # --- Generation: walk the strengthened graph within a grammar shape --
    def next_word_candidates(self, current_word, region):
        """Return candidate words for `region`, ranked by strength of
        their learned connection to current_word (falls back to region
        frequency if no connection learned yet)."""
        candidates = REGIONS[region]
        scored = []
        for w in candidates:
            strength = self.weights.get((current_word, w), 0.0)
            freq_bonus = self.word_freq.get(w, 0) * 0.01
            scored.append((w, strength + freq_bonus + random.random() * 0.05))
        scored.sort(key=lambda x: -x[1])
        return scored

    def next_word_candidates_wide(self, context_words, region, used_words, recency_penalty=0.6):
        """Wider-context version: instead of only the immediately previous
        word, sum association strength from ALL words generated so far in
        this sentence (with more recent words weighted higher). Also
        applies an inhibition penalty to words already used, so the model
        doesn't immediately loop back onto the same strong pair."""
        candidates = REGIONS[region]
        scored = []
        for w in candidates:
            strength = 0.0
            for i, cw in enumerate(reversed(context_words)):  # most recent first
                recency_weight = 1.0 / (1 + i)  # closer words matter more
                strength += self.weights.get((cw, w), 0.0) * recency_weight
            freq_bonus = self.word_freq.get(w, 0) * 0.01
            inhibition = recency_penalty if w in used_words else 0.0
            noise = random.random() * 0.05
            scored.append((w, strength + freq_bonus + noise - inhibition))
        scored.sort(key=lambda x: -x[1])
        return scored

    def generate(self, seed_word=None, template=None, wide_context=True):
        """Generate one sentence by walking a grammar template.

        wide_context=False -> original behaviour (only looks at previous
                               word; kept for comparison).
        wide_context=True  -> looks at the whole sentence generated so
                               far, weighted by recency, and inhibits
                               words already used to avoid repetition
                               loops. Still NOT full transformer attention
                               (no learned attention weights, no query/key/
                               value projections) -- just a wider, hand-
                               weighted window. Useful for seeing how much
                               of the "looping" problem is fixable with a
                               wider window alone vs. needing real attention."""
        template = template or random.choice(SENTENCE_TEMPLATES)
        sentence = []
        used_words = set()
        current = seed_word

        for region in template:
            if current is None:
                choices = REGIONS[region]
                current = max(choices, key=lambda w: self.word_freq.get(w, 0) + random.random())
            elif wide_context:
                ranked = self.next_word_candidates_wide(sentence + ([seed_word] if seed_word else []),
                                                         region, used_words)
                current = ranked[0][0]
            else:
                ranked = self.next_word_candidates(current, region)
                current = ranked[0][0]
            sentence.append(current)
            used_words.add(current)

        return " ".join(sentence).capitalize() + "."

    # --- Catastrophic forgetting probe -------------------------------------
    def association_strength(self, word_pair):
        return self.weights.get(word_pair, 0.0)

    # --- Inspection helpers ------------------------------------------------
    def top_associations(self, word, k=5):
        pairs = [(w2, s) for (w1, w2), s in self.weights.items() if w1 == word]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def save(self, path):
        data = {
            "weights": {f"{a}|{b}": s for (a, b), s in self.weights.items()},
            "word_freq": dict(self.word_freq),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.weights = defaultdict(float)
        for k, v in data["weights"].items():
            a, b = k.split("|")
            self.weights[(a, b)] = v
        self.word_freq = defaultdict(int, data["word_freq"])


# ---------------------------------------------------------------------------
# 3. DEMO: incremental learning + generation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = HebbianLanguageModel(learning_rate=0.15, decay=0.002)

    corpus_batch_1 = [
        "the quick dog chases the small cat",
        "the old man reads a hidden book",
        "a bright sun watches the quiet city",
        "the gentle woman finds the ancient river",
    ]

    corpus_batch_2 = [  # simulate learning MORE, later, incrementally
        "the young child builds a big story",
        "the quick cat runs near the tree",
        "the ancient book writes a hidden story",
    ]

    print("=== Learning batch 1 ===")
    model.learn_corpus(corpus_batch_1)
    print("-- narrow context (original) --")
    for _ in range(3):
        print(" ", model.generate(wide_context=False))
    print("-- wide context + inhibition (new) --")
    for _ in range(3):
        print(" ", model.generate(wide_context=True))

    # Snapshot a few key associations from batch 1 BEFORE learning batch 2,
    # so we can check afterwards whether they degraded (catastrophic
    # forgetting) or survived (since Hebbian updates are local/additive,
    # not global gradient rewrites).
    probe_pairs = [("dog", "chases"), ("man", "reads"), ("woman", "finds")]
    print("\n=== Associations BEFORE batch 2 ===")
    before = {}
    for pair in probe_pairs:
        before[pair] = model.association_strength(pair)
        print(f"  {pair}: {before[pair]:.4f}")

    print("\n=== Learning batch 2 (incremental, no retraining from scratch) ===")
    model.learn_corpus(corpus_batch_2)
    print("-- wide context + inhibition, after batch 2 --")
    for _ in range(3):
        print(" ", model.generate(wide_context=True))

    print("\n=== Associations AFTER batch 2 (forgetting check) ===")
    for pair in probe_pairs:
        after = model.association_strength(pair)
        delta = after - before[pair]
        pct = (delta / before[pair] * 100) if before[pair] else float("nan")
        print(f"  {pair}: {after:.4f}  (change: {delta:+.4f}, {pct:+.1f}%)")
    print("  -> Small negative changes here come only from the global decay")
    print("     term, NOT from a gradient computed for an unrelated task.")
    print("     This is the key structural difference from backprop: batch 2")
    print("     never computes an error signal that reaches back and rewrites")
    print("     batch-1 weights on purpose. Compare this to a real neural net")
    print("     fine-tuned by backprop on new data, where unrelated weights")
    print("     can shift a lot because gradients flow globally through every")
    print("     shared parameter.")

    print("\n=== Word association check ('dog') ===")
    print(" top associations:", model.top_associations("dog"))

    print("\n=== Generate seeded from a specific word ===")
    print(" ", model.generate(seed_word="the",
                               template=["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
                               wide_context=True))
