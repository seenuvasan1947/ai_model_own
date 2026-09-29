"""Generate synthetic LANGUAGE-ONLY training data for Layer 1.

REVISED from the first version: world-facts have been removed because
Layer 1 is the FROZEN language substrate -- baking specific facts (capitals,
planets, "the sky is blue", "cats have four legs") into Layer 1 defeats
the whole design, since Layer 2/3 exists precisely to store facts that can
be added, updated, and forgotten over time.

This file now ONLY contains language-mechanics data:
  - Number words and arithmetic patterns (teaches numeric vocabulary as
    English syntax: "three plus four is seven")
  - Day/month name sequences (teaches those names as vocabulary and their
    canonical ordering; days/months are a closed vocabulary set, not
    per-user facts)
  - Opposites (a closed English vocabulary of word-pairs, like tense
    forms -- part of the language, not per-user facts)
  - Grammar patterns: tense, negation, questions, plurals, sentence shapes

Explicitly dropped from the previous version:
  - Capitals of countries -> Layer 2/3 (fed via continual_read.py)
  - Planets, solar-system facts -> Layer 2/3
  - "The sky is blue" / "grass is green" style world-facts -> Layer 2/3
  - Animal-attribute facts ("a cat has four legs") -> Layer 2/3

Output: data/synthetic.txt (~50-150K tokens, language patterns only).
Run:    python data_prep/gen_synthetic.py
"""

import os
import random


OUT_PATH = "data/synthetic.txt"

random.seed(0)  # deterministic


NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine", "ten", "eleven", "twelve"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple",
          "pink", "brown", "gray"]
ANIMALS = ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "rabbit",
           "elephant", "lion", "tiger", "bear", "wolf", "fox", "mouse", "duck"]
OBJECTS = ["book", "table", "chair", "car", "tree", "house", "door", "window",
           "cup", "plate", "clock", "phone", "computer", "pen", "paper"]
PLACES = ["kitchen", "garden", "park", "school", "library", "office", "store"]
# Opposites: closed English vocabulary of word-pairs, kept as language content
OPPOSITES = [("hot", "cold"), ("big", "small"), ("tall", "short"),
             ("fast", "slow"), ("happy", "sad"), ("up", "down"),
             ("in", "out"), ("open", "closed"), ("light", "dark"),
             ("hard", "soft"), ("new", "old"), ("full", "empty"),
             ("wet", "dry"), ("strong", "weak"), ("clean", "dirty"),
             ("near", "far"), ("above", "below"), ("early", "late")]


def counting_lines():
    out = []
    for n in range(12):
        out.append(f"The number after {NUMBERS[n]} is {NUMBERS[n+1]}.")
        out.append(f"After {NUMBERS[n]} comes {NUMBERS[n+1]}.")
    for n in range(1, 13):
        out.append(f"The number before {NUMBERS[n]} is {NUMBERS[n-1]}.")
        out.append(f"Before {NUMBERS[n]} comes {NUMBERS[n-1]}.")
    # ordinal usage patterns
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth",
                "seventh", "eighth", "ninth", "tenth"]
    for i, ord_ in enumerate(ordinals):
        n = NUMBERS[i+1]
        out.append(f"The {ord_} number is {n}.")
    return out


def arithmetic_lines():
    """Number-language patterns. We repeat many combinations so the model
    learns the SYNTAX of arithmetic sentences, not any specific fact."""
    out = []
    for _ in range(600):
        a, b = random.randint(0, 6), random.randint(0, 6)
        s = a + b
        if s > 12:
            continue
        out.append(f"{NUMBERS[a].capitalize()} plus {NUMBERS[b]} is {NUMBERS[s]}.")
        out.append(f"{NUMBERS[a].capitalize()} plus {NUMBERS[b]} equals {NUMBERS[s]}.")
    for _ in range(600):
        a, b = random.randint(0, 12), random.randint(0, 12)
        if a < b:
            continue
        d = a - b
        if d > 12:
            continue
        out.append(f"{NUMBERS[a].capitalize()} minus {NUMBERS[b]} is {NUMBERS[d]}.")
    for _ in range(300):
        a, b = random.randint(1, 4), random.randint(1, 4)
        p = a * b
        if p > 12:
            continue
        out.append(f"{NUMBERS[a].capitalize()} times {NUMBERS[b]} is {NUMBERS[p]}.")
    return out


def calendar_vocabulary_lines():
    """Day/month names as vocabulary + their canonical sequence.
    This teaches the NAMES as words and how they're ordered as a closed
    linguistic set (like tense forms), NOT calendar facts."""
    out = []
    for i, d in enumerate(DAYS):
        out.append(f"The day after {d} is {DAYS[(i+1) % 7]}.")
        out.append(f"The day before {d} is {DAYS[(i-1) % 7]}.")
    for i, m in enumerate(MONTHS):
        out.append(f"The month after {m} is {MONTHS[(i+1) % 12]}.")
        out.append(f"The month before {m} is {MONTHS[(i-1) % 12]}.")
    # canonical enumerations (teaching the closed sets as sequences)
    out.append("The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday.")
    out.append("The months of the year are January, February, March, April, May, June, July, August, September, October, November, and December.")
    return out


def opposite_lines():
    """Opposites: closed vocabulary of English word-pairs. This is
    language (like knowing tense forms), not world-facts."""
    out = []
    for a, b in OPPOSITES:
        out.append(f"The opposite of {a} is {b}.")
        out.append(f"The opposite of {b} is {a}.")
        out.append(f"{a.capitalize()} and {b} are opposites.")
        out.append(f"{a.capitalize()} is not {b}.")
        out.append(f"{b.capitalize()} is not {a}.")
    return out


def sentence_pattern_lines():
    """Grammar patterns exercising tense, negation, questions, plurals.
    No specific facts about the world -- these are template sentences
    illustrating English sentence structure."""
    out = []
    verbs = [("run", "runs", "ran"), ("walk", "walks", "walked"),
             ("eat", "eats", "ate"), ("see", "sees", "saw"),
             ("read", "reads", "read"), ("write", "writes", "wrote"),
             ("play", "plays", "played"), ("build", "builds", "built"),
             ("open", "opens", "opened"), ("close", "closes", "closed"),
             ("find", "finds", "found"), ("carry", "carries", "carried")]
    subjects = ["The boy", "The girl", "The teacher", "A child", "My friend",
                "The man", "The woman", "The doctor", "Her brother", "His sister"]
    objects_ = ["a book", "the ball", "a picture", "a letter", "the garden",
                "a story", "the door", "a window", "the paper", "an apple"]

    for _ in range(600):
        s = random.choice(subjects)
        base, third, past = random.choice(verbs)
        o = random.choice(objects_)
        # tense triplet
        out.append(f"{s} {third} {o}.")
        out.append(f"{s} {past} {o} yesterday.")
        out.append(f"{s} will {base} {o} tomorrow.")
        # negation
        out.append(f"{s} does not {base} {o}.")
        out.append(f"{s} did not {base} {o}.")
        # question
        out.append(f"Does {s.lower()} {base} {o}?")
        out.append(f"Did {s.lower()} {base} {o}?")

    # plurals
    plurals = [("boy", "boys"), ("girl", "girls"), ("book", "books"),
               ("child", "children"), ("man", "men"), ("woman", "women"),
               ("mouse", "mice"), ("foot", "feet"), ("tree", "trees"),
               ("house", "houses"), ("box", "boxes"), ("leaf", "leaves"),
               ("wolf", "wolves"), ("baby", "babies"), ("city", "cities")]
    for sing, plur in plurals:
        out.append(f"One {sing}, two {plur}.")
        out.append(f"The {plur} are here.")
        out.append(f"A {sing} is small; many {plur} are together.")
        out.append(f"There is one {sing}.")
        out.append(f"There are several {plur}.")

    # simple attributive constructions (adjective + noun) -- uses colors
    # and animals ONLY as linguistic slots to exercise grammar, no factual
    # claims about them.
    for _ in range(400):
        c = random.choice(COLORS)
        o = random.choice(OBJECTS)
        out.append(f"A {c} {o} sat in the {random.choice(PLACES)}.")
        out.append(f"She saw a {c} {o}.")
    for _ in range(300):
        c = random.choice(COLORS)
        a = random.choice(ANIMALS)
        out.append(f"The {c} {a} ran across the {random.choice(PLACES)}.")

    return out


def main():
    lines = []
    lines += counting_lines()
    lines += arithmetic_lines()
    lines += calendar_vocabulary_lines()
    lines += opposite_lines()
    lines += sentence_pattern_lines()

    # Shuffle so a training batch draws from mixed categories.
    random.shuffle(lines)

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    words = sum(len(l.split()) for l in lines)
    print(f"Wrote {len(lines):,} sentences ({words:,} words) to {OUT_PATH}")
    print("Contents: number-language, day/month vocabulary, opposites, and")
    print("grammar-pattern sentences. NO world-facts (capitals, planets, etc.")
    print("-- those belong in Layer 2/3, fed via continual_read.py).")


if __name__ == "__main__":
    main()
