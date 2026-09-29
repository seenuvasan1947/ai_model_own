"""
Phase 1 — Four-Layer Hebbian Language Model
============================================

Layer 1: English Base (Hebbian co-occurrence, auto POS-tagged, frozen after training)
Layer 2: Knowledge Graph (subject→predicate→object triples, continuously updated)
Layer 3: Domain Layer (Hebbian co-occurrence for domain text, continuously updated)
Layer 4: Generation (POS-guided, graph-seeded, no backprop anywhere)

Usage:
    python model.py
"""

import re
import os
import json
import random
import math
from collections import defaultdict


# =============================================================================
# 0. CORPUS UTILITIES
# =============================================================================

def try_load_brown_tagged(max_sentences=3000):
    try:
        import nltk
        from nltk.corpus import brown
        try:
            tagged = brown.tagged_sents(tagset='universal')
        except LookupError:
            nltk.download('brown', quiet=True)
            nltk.download('universal_tagset', quiet=True)
            tagged = brown.tagged_sents(tagset='universal')

        sentences, pos_counts = [], defaultdict(lambda: defaultdict(int))
        for sent in list(tagged)[:max_sentences]:
            words = []
            for word, tag in sent:
                w = word.lower()
                words.append(w)
                pos_counts[w][tag] += 1
            sentences.append(" ".join(words))

        word_pos = {w: max(c, key=c.get) for w, c in pos_counts.items()}
        return sentences, word_pos
    except Exception:
        return None, None


BUNDLED_CORPUS = [
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
]

FALLBACK_POS = {
    "the": "DET", "a": "DET", "an": "DET", "this": "DET", "that": "DET",
    "dog": "NOUN", "cat": "NOUN", "man": "NOUN", "woman": "NOUN",
    "sun": "NOUN", "river": "NOUN", "tree": "NOUN", "house": "NOUN",
    "runs": "VERB", "reads": "VERB", "flows": "VERB", "builds": "VERB",
    "quick": "ADJ", "bright": "ADJ", "old": "ADJ", "gentle": "ADJ",
    "in": "ADP", "near": "ADP", "under": "ADP", "over": "ADP",
    "quickly": "ADV", "happily": "ADV", "gently": "ADV", "softly": "ADV",
}

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
    return split_sentences(raw)


def split_sentences(text):
    text = text.replace("\n", " ")
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(sentence):
    s = sentence.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return [w for w in s.split() if w]


# =============================================================================
# 1. LAYER 1 + 3 — HEBBIAN CO-OCCURRENCE
# =============================================================================

class HebbianLayer:
    def __init__(self, lr=0.15, decay=0.001, window=4):
        self.lr = lr
        self.decay = decay
        self.window = window
        self.weights = defaultdict(float)
        self.freq = defaultdict(int)
        self.frozen = False

    def update(self, words):
        if self.frozen:
            return
        n = len(words)
        for i, w1 in enumerate(words):
            self.freq[w1] += 1
            for j in range(i + 1, min(i + 1 + self.window, n)):
                w2 = words[j]
                dist = j - i
                s = self.lr / dist
                self.weights[(w1, w2)] += s
                self.weights[(w2, w1)] += s * 0.5
        for pair in list(self.weights):
            self.weights[pair] *= (1 - self.decay)
            if self.weights[pair] < 1e-4:
                del self.weights[pair]

    def learn(self, sentence):
        self.update(tokenize(sentence))

    def learn_corpus(self, sentences, label=""):
        total = len(sentences)
        for i, s in enumerate(sentences, 1):
            self.learn(s)
            if i % 200 == 0 or i == total:
                print(f"  {label} {i}/{total}")

    def score(self, w1, w2):
        return self.weights.get((w1, w2), 0.0)

    def freeze(self):
        self.frozen = True

    def top(self, word, k=5):
        pairs = [(w2, s) for (w1, w2), s in self.weights.items() if w1 == word]
        return sorted(pairs, key=lambda x: -x[1])[:k]

    def to_dict(self):
        return {
            "weights": {f"{a}|{b}": s for (a, b), s in self.weights.items()},
            "freq": dict(self.freq),
            "frozen": self.frozen,
        }

    def from_dict(self, d):
        self.weights = defaultdict(float)
        for k, v in d["weights"].items():
            a, b = k.split("|", 1)
            self.weights[(a, b)] = v
        self.freq = defaultdict(int, d["freq"])
        self.frozen = d["frozen"]


# =============================================================================
# 2. LAYER 2 — KNOWLEDGE GRAPH
# =============================================================================

class KnowledgeGraph:
    """
    Stores facts as (subject, predicate, object) triples.
    Auto-extracts triples from plain sentences using simple NLP patterns.
    Answers questions by looking up relevant triples.
    No backprop, no ML — pure graph traversal.
    """

    # Simple extraction patterns: (regex, predicate label)
    PATTERNS = [
        (r"^(.+?)\s+is\s+(?:a|an|the)?\s*(.+)$",          "is-a"),
        (r"^(.+?)\s+are\s+(?:a|an|the)?\s*(.+)$",          "is-a"),
        (r"^(.+?)\s+means?\s+(.+)$",                        "means"),
        (r"^(.+?)\s+(?:blocks?|prevents?)\s+(.+)$",         "blocks"),
        (r"^(.+?)\s+(?:allows?|enables?)\s+(.+)$",          "allows"),
        (r"^(.+?)\s+(?:detects?|finds?)\s+(.+)$",           "detects"),
        (r"^(.+?)\s+(?:attacks?|exploits?|hacks?)\s+(.+)$", "attacks"),
        (r"^(.+?)\s+(?:protects?|secures?)\s+(.+)$",        "protects"),
        (r"^(.+?)\s+(?:stores?|saves?|holds?)\s+(.+)$",     "stores"),
        (r"^(.+?)\s+(?:runs?|executes?|operates?)\s+(.+)$", "runs"),
        (r"^(.+?)\s+(?:sends?|transmits?)\s+(.+)$",         "sends"),
        (r"^(.+?)\s+(?:names?|refers? to)\s+(.+)$",         "names"),
        (r"^(.+?)\s+(?:describes?|represents?)\s+(.+)$",    "describes"),
        (r"^(.+?)\s+(?:contains?|includes?|has)\s+(.+)$",   "contains"),
        (r"^(.+?)\s+(?:causes?|leads? to)\s+(.+)$",         "causes"),
        (r"^(.+?)\s+(?:uses?|utilizes?)\s+(.+)$",           "uses"),
    ]

    def __init__(self):
        # triples: subject -> list of (predicate, object)
        self.graph = defaultdict(list)
        self.all_triples = []

    def extract_triples(self, sentence):
        """Try each pattern against the sentence, return matched triples."""
        s = sentence.strip().rstrip(".")
        triples = []
        for pattern, pred in self.PATTERNS:
            m = re.match(pattern, s, re.IGNORECASE)
            if m:
                subj = m.group(1).strip().lower()
                obj  = m.group(2).strip().lower()
                # Clean up subj/obj
                subj = re.sub(r'^(a|an|the)\s+', '', subj)
                obj  = re.sub(r'^(a|an|the)\s+', '', obj)
                if subj and obj and subj != obj:
                    triples.append((subj, pred, obj))
                break  # first match wins
        return triples

    def add_sentence(self, sentence):
        triples = self.extract_triples(sentence)
        for subj, pred, obj in triples:
            entry = (pred, obj)
            if entry not in self.graph[subj]:
                self.graph[subj].append(entry)
                self.all_triples.append((subj, pred, obj))
        return triples

    def add_corpus(self, sentences):
        count = 0
        for s in sentences:
            count += len(self.add_sentence(s))
        return count

    def lookup(self, concept):
        """Return all (predicate, object) facts for a concept."""
        concept = concept.strip().lower()
        concept = re.sub(r'^(a|an|the)\s+', '', concept)
        return self.graph.get(concept, [])

    def answer(self, question):
        """
        Simple question answering:
        - Extract the key concept from the question
        - Look up facts in the graph
        - Return a list of answer strings
        """
        q = question.lower().strip().rstrip("?")

        # Extract concept from common question patterns
        concept = None
        patterns = [
            r"what (?:is|are|does|do) (?:a |an |the )?(.+?)(?:\s+do|\s+mean|\s+work)?$",
            r"what (?:is|are) (?:the )?(?:purpose|role|function) of (.+)$",
            r"how does (.+?) work",
            r"tell me about (.+)$",
            r"explain (.+)$",
            r"define (.+)$",
            r"what can (.+?) do",
            r"(?:is|are) (.+?) (?:a|an) (.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, q)
            if m:
                concept = m.group(1).strip()
                break

        if not concept:
            # Fall back: last meaningful word in question
            words = [w for w in q.split() if len(w) > 3
                     and w not in {"what","does","that","this","have","been","will","from","with","about","tell","explain","define"}]
            concept = words[-1] if words else None

        if not concept:
            return [], None

        facts = self.lookup(concept)
        if not facts:
            # Try partial match
            for key in self.graph:
                if concept in key or key in concept:
                    facts = self.graph[key]
                    concept = key
                    break

        return facts, concept

    def to_dict(self):
        return {
            "graph": {k: v for k, v in self.graph.items()},
            "all_triples": self.all_triples,
        }

    def from_dict(self, d):
        self.graph = defaultdict(list, {k: v for k, v in d["graph"].items()})
        self.all_triples = d["all_triples"]


# =============================================================================
# 3. LAYER 4 — GENERATION
# =============================================================================

class Generator:
    def __init__(self, base_layer, domain_layer, pos_words, vocab):
        self.base = base_layer
        self.domain = domain_layer
        self.pos_words = pos_words  # POS -> set of words
        self.vocab = vocab

    def _score(self, context, word):
        score = 0.0
        for i, cw in enumerate(reversed(context[-4:])):
            recency = 1.0 / (1 + i)
            score += (self.base.score(cw, word) * 1.0 +
                      self.domain.score(cw, word) * 2.0) * recency
        freq = self.base.freq.get(word, 0) + self.domain.freq.get(word, 0)
        return score / math.log(2 + freq)

    def generate(self, seed_words=None, length=10):
        """
        seed_words: list of words to bias generation toward (from graph lookup)
        """
        template = random.choice(GRAMMAR_TEMPLATES)
        sentence = []
        used = set()

        # Start with a determiner or first seed word
        det_pool = list(self.pos_words.get("DET", {"the", "a"}))
        first = random.choice(det_pool)
        sentence.append(first)
        used.add(first)

        for pos_needed in template[1:]:
            candidates = list(self.pos_words.get(pos_needed, self.vocab))
            if not candidates:
                candidates = list(self.vocab)

            # Boost seed words that match this POS slot
            boosted = set()
            if seed_words:
                boosted = {w for w in seed_words
                           if w in candidates and w not in used}

            scored = []
            for w in candidates:
                if w in used:
                    continue
                s = self._score(sentence, w)
                if w in boosted:
                    s *= 3.0  # strong boost for graph-retrieved words
                s += random.random() * 0.01
                scored.append((w, s))

            if not scored:
                break

            scored.sort(key=lambda x: -x[1])
            top = scored[:5]
            chosen = random.choices([w for w, _ in top],
                                    weights=[max(s, 0.001) for _, s in top])[0]
            sentence.append(chosen)
            used.add(chosen)

        return " ".join(sentence).capitalize() + "."


# =============================================================================
# 4. FULL MODEL — WIRES ALL FOUR LAYERS
# =============================================================================

class Phase1Model:
    def __init__(self):
        self.base   = HebbianLayer(lr=0.15, decay=0.001, window=4)
        self.domain = HebbianLayer(lr=0.15, decay=0.001, window=4)
        self.graph  = KnowledgeGraph()
        self.vocab  = set()
        self.word_pos = {}
        self.pos_words = defaultdict(set)

    def _register_pos(self, word, tag):
        self.word_pos[word] = tag
        self.pos_words[tag].add(word)
        self.vocab.add(word)

    # --- Stage 1: train English base ---
    def train_english(self, sentences, word_pos_map=None):
        print(f"Training English base on {len(sentences)} sentences...")
        self.base.learn_corpus(sentences, label="English")
        pos_map = word_pos_map or FALLBACK_POS
        for word, tag in pos_map.items():
            self._register_pos(word, tag)
        for w in self.base.freq:
            if w not in self.word_pos:
                self._register_pos(w, "NOUN")
        self.base.freeze()
        print(f"[base frozen] {len(self.base.weights):,} pairs, "
              f"{len(self.vocab):,} words, {len(self.pos_words)} POS classes.")

    # --- Stage 2: learn domain continuously ---
    def learn(self, sentence: str):
        """Feed one sentence — updates domain layer AND knowledge graph."""
        self.domain.learn(sentence)
        # Register new words
        for w in tokenize(sentence):
            if w not in self.word_pos:
                self._register_pos(w, "NOUN")
        # Extract triples into graph
        triples = self.graph.add_sentence(sentence)
        return triples

    def learn_file(self, path):
        sentences = load_text_file(path)
        print(f"Learning {len(sentences)} sentences from '{path}'...")
        triple_count = 0
        for i, s in enumerate(sentences, 1):
            triples = self.learn(s)
            triple_count += len(triples)
            if i % 100 == 0 or i == len(sentences):
                print(f"  {i}/{len(sentences)} sentences, "
                      f"{triple_count} facts extracted so far")
        print(f"[done] {triple_count} total facts stored in knowledge graph.")

    # --- Stage 3: answer a question ---
    def ask(self, question):
        """
        1. Look up concept in knowledge graph
        2. Compose answer directly from retrieved triples
        3. Fall back to generation only if no facts found
        """
        facts, concept = self.graph.answer(question)

        if not facts:
            print(f"  [no facts found for '{concept}' -- generating from base only]")
            gen = Generator(self.base, self.domain, self.pos_words, self.vocab)
            return gen.generate(seed_words=tokenize(question), length=10)

        print(f"  [found {len(facts)} fact(s) about '{concept}']")
        for pred, obj in facts[:3]:
            print(f"    ({concept}) --[{pred}]--> ({obj})")

        # Compose answer directly from triples — no generation randomness
        answers = []
        for pred, obj in facts[:3]:
            if pred == "is-a":
                answers.append(f"{concept.capitalize()} is {obj}.")
            elif pred == "means":
                answers.append(f"{concept.capitalize()} means {obj}.")
            elif pred == "blocks":
                answers.append(f"{concept.capitalize()} blocks {obj}.")
            elif pred == "allows":
                answers.append(f"{concept.capitalize()} allows {obj}.")
            elif pred == "detects":
                answers.append(f"{concept.capitalize()} detects {obj}.")
            elif pred == "attacks":
                answers.append(f"{concept.capitalize()} attacks {obj}.")
            elif pred == "protects":
                answers.append(f"{concept.capitalize()} protects {obj}.")
            elif pred == "stores":
                answers.append(f"{concept.capitalize()} stores {obj}.")
            elif pred == "runs":
                answers.append(f"{concept.capitalize()} runs {obj}.")
            elif pred == "uses":
                answers.append(f"{concept.capitalize()} uses {obj}.")
            elif pred == "causes":
                answers.append(f"{concept.capitalize()} causes {obj}.")
            elif pred == "contains":
                answers.append(f"{concept.capitalize()} contains {obj}.")
            elif pred == "names":
                answers.append(f"{concept.capitalize()} names {obj}.")
            elif pred == "describes":
                answers.append(f"{concept.capitalize()} describes {obj}.")
            else:
                answers.append(f"{concept.capitalize()} {pred} {obj}.")

        return " ".join(answers)

    # --- Generate freely ---
    def generate(self, seed_word=None):
        seed = [seed_word] if seed_word else []
        gen = Generator(self.base, self.domain, self.pos_words, self.vocab)
        return gen.generate(seed_words=seed, length=10)

    # --- Save / Load ---
    def save(self, path="phase1_model.json"):
        data = {
            "base":      self.base.to_dict(),
            "domain":    self.domain.to_dict(),
            "graph":     self.graph.to_dict(),
            "word_pos":  self.word_pos,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[saved] {path}")

    def load(self, path="phase1_model.json"):
        with open(path) as f:
            data = json.load(f)
        self.base.from_dict(data["base"])
        self.domain.from_dict(data["domain"])
        self.graph.from_dict(data["graph"])
        self.word_pos = data["word_pos"]
        self.pos_words = defaultdict(set)
        for word, tag in self.word_pos.items():
            self.pos_words[tag].add(word)
            self.vocab.add(word)
        print(f"[loaded] base frozen={self.base.frozen}, "
              f"vocab={len(self.vocab):,}, "
              f"graph facts={len(self.graph.all_triples)}")


# =============================================================================
# 5. MAIN LOOP
# =============================================================================

if __name__ == "__main__":
    save_path = "phase1_model.json"
    model = Phase1Model()

    # --- Load or train ---
    if os.path.exists(save_path):
        print(f"[INFO] Found saved model: {save_path}")
        c = input("Load it? (y=load / n=retrain English): ").strip().lower()
        if c == "y":
            model.load(save_path)

    if not model.base.frozen:
        print("\n" + "=" * 60)
        print("STAGE 1: Learning English (one time only)")
        print("=" * 60)
        sentences, word_pos = try_load_brown_tagged(max_sentences=3000)
        if sentences:
            print(f"NLTK Brown Corpus: {len(sentences)} sentences, "
                  f"{len(word_pos):,} tagged words.")
        else:
            print("NLTK not found -- using bundled sample.")
            print("Tip: pip install nltk && python -c "
                  "\"import nltk; nltk.download('brown')\"")
            sentences, word_pos = BUNDLED_CORPUS, None
        model.train_english(sentences, word_pos)
        model.save(save_path)

    # --- Main interactive loop ---
    print("\n" + "=" * 60)
    print("Phase 1 Model — Commands:")
    print("  learn <sentence>   — teach the model a fact")
    print("  file <path>        — load a plain text file to learn from")
    print("  ask <question>     — ask the model something")
    print("  gen <word>         — generate sentences from a seed word")
    print("  facts <concept>    — show stored facts about a concept")
    print("  save               — save current model state")
    print("  quit               — exit")
    print("=" * 60)

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        parts = raw.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "quit":
            break

        elif cmd == "learn":
            if arg:
                triples = model.learn(arg)
                print(f"  [learned]", end="")
                if triples:
                    for subj, pred, obj in triples:
                        print(f" | fact: ({subj}) --[{pred}]--> ({obj})", end="")
                print()
            else:
                print("  Usage: learn <sentence>")

        elif cmd == "file":
            if arg and os.path.exists(arg):
                model.learn_file(arg)
                model.save(save_path)
            else:
                print(f"  File not found: {arg}")

        elif cmd == "ask":
            if arg:
                answer = model.ask(arg)
                print(f"  Answer: {answer}")
            else:
                print("  Usage: ask <question>")

        elif cmd == "gen":
            seed = arg if arg else None
            for _ in range(5):
                print(" ", model.generate(seed_word=seed))

        elif cmd == "facts":
            if arg:
                facts = model.graph.lookup(arg)
                if facts:
                    print(f"  Facts about '{arg}':")
                    for pred, obj in facts:
                        print(f"    --[{pred}]--> {obj}")
                else:
                    print(f"  No facts stored about '{arg}'")
            else:
                print("  Usage: facts <concept>")

        elif cmd == "save":
            model.save(save_path)

        else:
            print(f"  Unknown command: {cmd}")
            print("  Commands: learn, file, ask, gen, facts, save, quit")
