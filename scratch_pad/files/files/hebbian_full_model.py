"""
Two-Layer Hebbian Language Model (no backprop)
===============================================

Architecture:
  LAYER 1 - "General language base"
      Trained once on Brown Corpus (NLTK). Auto POS-tags every word so
      generation is grammatically constrained without any hand-labeling.
      After training, frozen forever.

  LAYER 2 - "Domain knowledge" (your choice at runtime)
      Starts empty. You feed it any plain text file or type sentences live.
      Updates continuously, forever, no retraining needed.

  GENERATION
      Uses auto-learned POS classes to pick grammatically valid next words,
      scored by Hebbian associations from both layers combined.
      No backpropagation anywhere.

Usage:
    python hebbian_full_model.py
"""

import re
import random
import json
import os
from collections import defaultdict


# =============================================================================
# 0. CORPUS LOADING + AUTO POS TAGGING
# =============================================================================

def try_load_brown_tagged(max_sentences=3000):
    """Load Brown Corpus with POS tags already attached (NLTK provides these).
    Returns (sentences_as_word_lists, word->POS dict) or (None, None)."""
    try:
        import nltk
        from nltk.corpus import brown
        try:
            tagged = brown.tagged_sents(tagset='universal')
        except LookupError:
            nltk.download('brown', quiet=True)
            nltk.download('universal_tagset', quiet=True)
            tagged = brown.tagged_sents(tagset='universal')

        sentences = []
        word_pos = {}  # word -> most common POS tag

        pos_counts = defaultdict(lambda: defaultdict(int))

        for sent in list(tagged)[:max_sentences]:
            words = []
            for word, tag in sent:
                w = word.lower()
                words.append(w)
                pos_counts[w][tag] += 1
            sentences.append(" ".join(words))

        # Assign each word its most frequent POS tag
        for word, counts in pos_counts.items():
            word_pos[word] = max(counts, key=counts.get)

        return sentences, word_pos
    except Exception:
        return None, None


BUNDLED_GENERAL_CORPUS = [
    "The dog runs quickly across the quiet park.",
    "A man reads an old book near the window.",
    "The bright sun rises over the distant mountain.",
    "Children play happily in the crowded market.",
    "The river flows gently past the ancient city.",
    "She writes a long letter to her old friend.",
    "The quick cat chases a small mouse under the table.",
    "A gentle woman finds a hidden path through the forest.",
    "The train arrives at the station every morning.",
    "He builds a small house near the tall trees.",
    "The teacher explains the lesson to the young students.",
    "Birds fly south when the winter grows cold.",
    "The old car stops suddenly on the empty road.",
    "A young girl sings a soft song in the garden.",
    "The farmer plants seeds early in the spring.",
    "Rain falls softly on the sleeping town at night.",
    "The chef prepares a delicious meal for the guests.",
    "A wise king rules the peaceful kingdom for years.",
    "The scientist studies the stars through a large telescope.",
    "Workers build a new bridge across the wide river.",
    "The dog barks loudly when a stranger walks by.",
    "A child draws a picture of a bright yellow sun.",
    "The old man tells a story about his youth.",
    "Waves crash against the rocks near the shore.",
    "The doctor examines the patient with great care.",
    "A small boat sails slowly across the calm lake.",
    "The library is quiet during the early morning hours.",
    "Leaves fall gently from the trees in autumn.",
    "The baker sells fresh bread every single day.",
    "A curious cat explores the empty house alone.",
]

# Fallback POS map for bundled corpus (hand-labeled, used only when NLTK unavailable)
FALLBACK_POS = {
    "the": "DET", "a": "DET", "an": "DET", "this": "DET", "that": "DET",
    "dog": "NOUN", "cat": "NOUN", "man": "NOUN", "woman": "NOUN", "child": "NOUN",
    "sun": "NOUN", "river": "NOUN", "tree": "NOUN", "house": "NOUN", "book": "NOUN",
    "city": "NOUN", "market": "NOUN", "mountain": "NOUN", "park": "NOUN",
    "station": "NOUN", "garden": "NOUN", "story": "NOUN", "king": "NOUN",
    "teacher": "NOUN", "scientist": "NOUN", "baker": "NOUN", "doctor": "NOUN",
    "runs": "VERB", "reads": "VERB", "rises": "VERB", "plays": "VERB",
    "flows": "VERB", "writes": "VERB", "chases": "VERB", "finds": "VERB",
    "arrives": "VERB", "builds": "VERB", "explains": "VERB", "fly": "VERB",
    "stops": "VERB", "sings": "VERB", "plants": "VERB", "falls": "VERB",
    "prepares": "VERB", "rules": "VERB", "studies": "VERB", "barks": "VERB",
    "draws": "VERB", "tells": "VERB", "crash": "VERB", "examines": "VERB",
    "sails": "VERB", "sells": "VERB", "explores": "VERB",
    "quick": "ADJ", "bright": "ADJ", "old": "ADJ", "gentle": "ADJ",
    "young": "ADJ", "quiet": "ADJ", "small": "ADJ", "large": "ADJ",
    "tall": "ADJ", "ancient": "ADJ", "hidden": "ADJ", "curious": "ADJ",
    "in": "ADP", "near": "ADP", "under": "ADP", "over": "ADP",
    "across": "ADP", "through": "ADP", "past": "ADP", "at": "ADP",
    "quickly": "ADV", "happily": "ADV", "gently": "ADV", "slowly": "ADV",
    "softly": "ADV", "loudly": "ADV", "early": "ADV",
}

# Grammar template: sequence of POS tags for sentence generation
# Uses universal tagset tags (from NLTK Brown)
GRAMMAR_TEMPLATES = [
    ["DET", "ADJ", "NOUN", "VERB", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "DET", "ADJ", "NOUN"],
    ["DET", "ADJ", "NOUN", "VERB", "ADP", "DET", "NOUN"],
    ["DET", "NOUN", "VERB", "ADV"],
    ["DET", "NOUN", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN"],
]


def load_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    return split_into_sentences(raw)


def split_into_sentences(raw_text):
    raw_text = raw_text.replace("\n", " ")
    parts = re.split(r'(?<=[.!?])\s+', raw_text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9'\s]", " ", sentence)
    return [w for w in sentence.split() if w]


# =============================================================================
# 1. THE TWO-LAYER HEBBIAN MODEL
# =============================================================================

class TwoLayerHebbianModel:
    def __init__(self, learning_rate=0.15, decay=0.001, window=4):
        self.lr = learning_rate
        self.decay = decay
        self.window = window

        self.base_weights = defaultdict(float)
        self.base_freq = defaultdict(int)
        self.base_frozen = False

        self.domain_weights = defaultdict(float)
        self.domain_freq = defaultdict(int)

        self.vocab = set()
        self.word_pos = {}        # word -> POS tag (auto-learned from corpus)
        self.pos_words = defaultdict(set)  # POS -> set of words

    def _register_pos(self, word, tag):
        self.word_pos[word] = tag
        self.pos_words[tag].add(word)

    def _hebbian_update(self, words, weights, freq):
        n = len(words)
        for i, w1 in enumerate(words):
            freq[w1] += 1
            self.vocab.add(w1)
            for j in range(i + 1, min(i + 1 + self.window, n)):
                w2 = words[j]
                dist = j - i
                strength = self.lr / dist
                weights[(w1, w2)] += strength
                weights[(w2, w1)] += strength * 0.5

    def _decay_weights(self, weights):
        for pair in list(weights.keys()):
            weights[pair] *= (1 - self.decay)
            if weights[pair] < 1e-4:
                del weights[pair]

    def train_base(self, sentences, word_pos_map=None):
        if self.base_frozen:
            raise RuntimeError("Base is frozen. Create a new model to retrain.")
        for sentence in sentences:
            words = tokenize(sentence)
            self._hebbian_update(words, self.base_weights, self.base_freq)
        self._decay_weights(self.base_weights)

        if word_pos_map:
            for word, tag in word_pos_map.items():
                self._register_pos(word, tag)
        else:
            # Use fallback POS for bundled corpus
            for word, tag in FALLBACK_POS.items():
                self._register_pos(word, tag)

        # Tag any remaining vocab words as NOUN by default
        for w in self.vocab:
            if w not in self.word_pos:
                self.word_pos[w] = "NOUN"
                self.pos_words["NOUN"].add(w)

    def freeze_base(self):
        self.base_frozen = True
        print(f"[base frozen] {len(self.base_weights):,} weighted pairs, "
              f"{len(self.base_freq):,} unique words, "
              f"{len(self.pos_words)} POS classes learned.")

    def learn_domain(self, sentence: str):
        words = tokenize(sentence)
        self._hebbian_update(words, self.domain_weights, self.domain_freq)
        self._decay_weights(self.domain_weights)
        # Auto-tag new domain words as NOUN if unseen
        for w in words:
            if w not in self.word_pos:
                self.word_pos[w] = "NOUN"
                self.pos_words["NOUN"].add(w)

    def learn_domain_corpus(self, sentences):
        total = len(sentences)
        for i, s in enumerate(sentences, 1):
            self.learn_domain(s)
            if i % 100 == 0 or i == total:
                print(f"  learned {i}/{total} sentences...")

    def combined_score(self, w1, w2, base_weight=1.0, domain_weight=1.5):
        b = self.base_weights.get((w1, w2), 0.0)
        d = self.domain_weights.get((w1, w2), 0.0)
        return b * base_weight + d * domain_weight

    def combined_freq(self, word):
        return self.base_freq.get(word, 0) + self.domain_freq.get(word, 0)

    def generate(self, seed_word=None, length=10,
                 base_weight=1.0, domain_weight=2.0):
        import math

        if not self.vocab:
            return ""

        # Pick a grammar template
        template = random.choice(GRAMMAR_TEMPLATES)
        sentence = []
        used = set()

        # Seed word: use its POS to find the right slot, or start from DET
        if seed_word and seed_word in self.vocab:
            current = seed_word
        else:
            det_words = list(self.pos_words.get("DET", ["the"]))
            current = random.choice(det_words)

        sentence.append(current)
        used.add(current)

        for pos_needed in template[1:]:
            candidates = list(self.pos_words.get(pos_needed, self.vocab))
            if not candidates:
                candidates = list(self.vocab)

            scored = []
            context = sentence[-self.window:]
            for w in candidates:
                if w in used:
                    continue
                score = 0.0
                for i, cw in enumerate(reversed(context)):
                    recency = 1.0 / (1 + i)
                    score += self.combined_score(cw, w, base_weight, domain_weight) * recency

                freq = self.combined_freq(w)
                # IDF normalization so high-freq words don't dominate
                score = score / math.log(2 + freq)
                score += random.random() * 0.01
                scored.append((w, score))

            if not scored:
                break

            scored.sort(key=lambda x: -x[1])
            # Top-3 sampling for variety
            top = scored[:3]
            chosen = random.choices([w for w, _ in top],
                                    weights=[s for _, s in top])[0]
            sentence.append(chosen)
            used.add(chosen)

        return " ".join(sentence).capitalize() + "."

    def top_domain_associations(self, word, k=5):
        pairs = [(w2, s) for (w1, w2), s in self.domain_weights.items() if w1 == word]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def save(self, path):
        data = {
            "base_weights": {f"{a}|{b}": s for (a, b), s in self.base_weights.items()},
            "base_freq": dict(self.base_freq),
            "base_frozen": self.base_frozen,
            "domain_weights": {f"{a}|{b}": s for (a, b), s in self.domain_weights.items()},
            "domain_freq": dict(self.domain_freq),
            "vocab": list(self.vocab),
            "word_pos": self.word_pos,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[saved] {path}")

    def load(self, path):
        with open(path) as f:
            data = json.load(f)

        def unpack(d):
            out = defaultdict(float)
            for k, v in d.items():
                a, b = k.split("|", 1)
                out[(a, b)] = v
            return out

        self.base_weights = unpack(data["base_weights"])
        self.base_freq = defaultdict(int, data["base_freq"])
        self.base_frozen = data["base_frozen"]
        self.domain_weights = unpack(data["domain_weights"])
        self.domain_freq = defaultdict(int, data["domain_freq"])
        self.vocab = set(data["vocab"])
        self.word_pos = data.get("word_pos", {})
        self.pos_words = defaultdict(set)
        for word, tag in self.word_pos.items():
            self.pos_words[tag].add(word)


# =============================================================================
# 2. MAIN: train English once, then you choose the domain
# =============================================================================

if __name__ == "__main__":
    save_path = "hebbian_model_state.json"
    model = TwoLayerHebbianModel(learning_rate=0.15, decay=0.001, window=4)

    # --- Load saved model if exists ---
    if os.path.exists(save_path):
        print(f"[INFO] Found saved model: {save_path}")
        choice = input("Load existing model? (y=skip English retraining / n=retrain): ").strip().lower()
        if choice == "y":
            model.load(save_path)
            print(f"[OK] Loaded. Base frozen={model.base_frozen}, "
                  f"vocab={len(model.vocab):,} words, "
                  f"POS classes={len(model.pos_words)}")

    # --- Stage 1: Train English base if not already frozen ---
    if not model.base_frozen:
        print("\n" + "=" * 70)
        print("STAGE 1: Learning English (one time only)")
        print("=" * 70)

        sentences, word_pos = try_load_brown_tagged(max_sentences=3000)
        if sentences:
            print(f"Using NLTK Brown Corpus: {len(sentences)} sentences, "
                  f"{len(word_pos):,} auto-tagged words.")
        else:
            print("NLTK not available -- using bundled sample. "
                  "Run: pip install nltk && python -c \"import nltk; nltk.download('brown')\"")
            sentences = BUNDLED_GENERAL_CORPUS
            word_pos = None

        model.train_base(sentences, word_pos)
        model.freeze_base()
        model.save(save_path)

    # --- Quick generation from base only ---
    print("\nGeneration from English base (no domain yet):")
    for _ in range(3):
        print(" ", model.generate(length=8))

    # --- Stage 2: You choose the domain ---
    print("\n" + "=" * 70)
    print("STAGE 2: Teach the model a domain (your choice)")
    print("=" * 70)
    print("  1. Load a plain text file")
    print("  2. Type sentences manually")
    print("  3. Skip -- generate from base only")
    mode = input("\nChoose (1/2/3): ").strip()

    if mode == "1":
        path = input("Path to your text file: ").strip().strip('"')
        if os.path.exists(path):
            sentences = load_text_file(path)
            print(f"Loaded {len(sentences)} sentences from '{path}'")
            model.learn_domain_corpus(sentences)
            model.save(save_path)
        else:
            print(f"File not found: {path}")

    elif mode == "2":
        print("Type sentences one at a time. Empty line to stop.")
        count = 0
        while True:
            s = input("  > ").strip()
            if not s:
                break
            model.learn_domain(s)
            count += 1
            print(f"    [learned #{count}]")
        if count:
            model.save(save_path)

    # --- Stage 3: Generate ---
    print("\n" + "=" * 70)
    print("STAGE 3: Generate  (type a seed word, or press Enter for random)")
    print("         Type 'quit' to exit.")
    print("=" * 70)
    while True:
        seed = input("\nSeed word: ").strip().lower() or None
        if seed == "quit":
            break
        for _ in range(5):
            print(" ", model.generate(seed_word=seed, length=10))
