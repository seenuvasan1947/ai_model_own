"""Generate synthetic grammar/numbers/facts training data for Layer 1.

Produces canonical English sentences covering:
  - Number literacy (counting, arithmetic, ordinals)
  - Days of the week, months, seasons
  - Colors, common animals, common objects
  - Comparatives / superlatives / opposites
  - Simple factual assertions (capitals, planets, body parts, etc.)
  - Sentence-pattern grammar (declarative, question, negation, tense)

None of this comes from a specific author -- it's raw, clean, generic English,
the kind of content a grammar textbook uses. This is deliberately what Layer 1
should learn from; domain content lives in Layer 2/3, not baked into weights.

Output: data/synthetic.txt (~100-500K tokens).
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
SEASONS = ["spring", "summer", "autumn", "winter"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple",
          "pink", "brown", "gray"]
ANIMALS = ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "rabbit",
           "elephant", "lion", "tiger", "bear", "wolf", "fox", "mouse", "duck"]
OBJECTS = ["book", "table", "chair", "car", "tree", "house", "door", "window",
           "cup", "plate", "clock", "phone", "computer", "pen", "paper"]
PLACES = ["kitchen", "garden", "park", "school", "library", "office", "store"]
OPPOSITES = [("hot", "cold"), ("big", "small"), ("tall", "short"),
             ("fast", "slow"), ("happy", "sad"), ("up", "down"),
             ("in", "out"), ("open", "closed"), ("light", "dark"),
             ("hard", "soft"), ("new", "old"), ("full", "empty")]
CAPITALS = [("France", "Paris"), ("England", "London"), ("Japan", "Tokyo"),
            ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
            ("India", "New Delhi"), ("China", "Beijing"), ("Russia", "Moscow"),
            ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Brazil", "Brasilia")]
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]


def counting_lines():
    out = []
    for _ in range(200):
        n = random.randint(0, 12)
        out.append(f"The number after {NUMBERS[n]} is {NUMBERS[n+1]}." if n < 12 else "The number after twelve is thirteen.")
    for _ in range(200):
        n = random.randint(1, 12)
        out.append(f"The number before {NUMBERS[n]} is {NUMBERS[n-1]}.")
    return out


def arithmetic_lines():
    out = []
    for _ in range(500):
        a, b = random.randint(0, 9), random.randint(0, 9)
        out.append(f"{NUMBERS[a]} plus {NUMBERS[b]} is {NUMBERS[min(a+b, 12)]}." if a+b <= 12 else f"{a} plus {b} is {a+b}.")
    for _ in range(500):
        a, b = random.randint(0, 12), random.randint(0, 12)
        if a >= b:
            out.append(f"{NUMBERS[a]} minus {NUMBERS[b]} is {NUMBERS[a-b]}." if a-b >= 0 and a-b <= 12 else f"{a} minus {b} is {a-b}.")
    for _ in range(300):
        a, b = random.randint(1, 6), random.randint(1, 6)
        out.append(f"{NUMBERS[a]} times {NUMBERS[b]} is {a*b}.")
    return out


def calendar_lines():
    out = []
    for i, d in enumerate(DAYS):
        nxt = DAYS[(i+1) % 7]
        prv = DAYS[(i-1) % 7]
        out.append(f"The day after {d} is {nxt}.")
        out.append(f"The day before {d} is {prv}.")
    for i, m in enumerate(MONTHS):
        nxt = MONTHS[(i+1) % 12]
        prv = MONTHS[(i-1) % 12]
        out.append(f"The month after {m} is {nxt}.")
        out.append(f"The month before {m} is {prv}.")
    out.extend([
        "There are seven days in a week.",
        "There are twelve months in a year.",
        "There are four seasons in a year.",
        "The four seasons are spring, summer, autumn, and winter.",
        "January is the first month of the year.",
        "December is the last month of the year.",
        "Monday is the first day of the work week.",
        "Sunday comes after Saturday.",
    ])
    return out


def color_object_lines():
    out = []
    for _ in range(300):
        c = random.choice(COLORS)
        o = random.choice(OBJECTS)
        out.append(f"The {c} {o} is on the table." if o != "table" else f"The {c} table is in the {random.choice(PLACES)}.")
    for _ in range(200):
        c = random.choice(COLORS)
        a = random.choice(ANIMALS)
        out.append(f"A {c} {a} sat in the {random.choice(PLACES)}.")
    out.extend([
        "The sky is blue.",
        "Grass is green.",
        "Snow is white.",
        "Coal is black.",
        "The sun is yellow.",
        "Blood is red.",
        "Fire is orange and red.",
        "Milk is white.",
        "A cat has four legs.",
        "A bird has two legs.",
        "A dog has four legs and a tail.",
        "A fish lives in water.",
        "A bird can fly.",
        "A rabbit likes to eat carrots.",
    ])
    return out


def opposite_lines():
    out = []
    for a, b in OPPOSITES:
        out.append(f"The opposite of {a} is {b}.")
        out.append(f"The opposite of {b} is {a}.")
        out.append(f"{a.capitalize()} is not {b}.")
        out.append(f"{b.capitalize()} is not {a}.")
    return out


def capital_planet_lines():
    out = []
    for country, cap in CAPITALS:
        out.append(f"The capital of {country} is {cap}.")
        out.append(f"{cap} is the capital of {country}.")
    for p in PLANETS:
        out.append(f"{p} is a planet in our solar system.")
    out.extend([
        "There are eight planets in our solar system.",
        "Earth is the third planet from the sun.",
        "The sun is a star.",
        "The moon orbits the Earth.",
        "The Pacific is the largest ocean on Earth.",
        "Water is made of hydrogen and oxygen.",
        "Ice is frozen water.",
        "Steam is water in gas form.",
    ])
    return out


def grammar_pattern_lines():
    """Basic sentence patterns exercising tense, negation, questions, plurals."""
    out = []
    verbs = [("run", "runs", "ran"), ("walk", "walks", "walked"),
             ("eat", "eats", "ate"), ("see", "sees", "saw"),
             ("read", "reads", "read"), ("write", "writes", "wrote"),
             ("play", "plays", "played"), ("build", "builds", "built")]
    subjects = ["The boy", "The girl", "The teacher", "A child", "My friend", "The man", "The woman"]
    objects_ = ["a book", "the ball", "a picture", "a letter", "the garden", "a story"]

    for _ in range(400):
        s = random.choice(subjects)
        base, third, past = random.choice(verbs)
        o = random.choice(objects_)
        out.append(f"{s} {third} {o}.")
        out.append(f"{s} {past} {o} yesterday.")
        out.append(f"{s} will {base} {o} tomorrow.")
        out.append(f"{s} does not {base} {o}.")
        out.append(f"Does {s.lower()} {base} {o}?")

    plurals = [("boy", "boys"), ("girl", "girls"), ("book", "books"),
               ("child", "children"), ("man", "men"), ("woman", "women"),
               ("mouse", "mice"), ("foot", "feet"), ("tree", "trees"),
               ("house", "houses")]
    for sing, plur in plurals:
        out.append(f"One {sing}, two {plur}.")
        out.append(f"The {plur} are here.")
        out.append(f"A {sing} is small; many {plur} are together.")

    return out


def main():
    lines = []
    lines += counting_lines()
    lines += arithmetic_lines()
    lines += calendar_lines()
    lines += color_object_lines()
    lines += opposite_lines()
    lines += capital_planet_lines()
    lines += grammar_pattern_lines()

    # Shuffle so a training batch draws from mixed categories, not one at a time.
    random.shuffle(lines)

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    words = sum(len(l.split()) for l in lines)
    print(f"Wrote {len(lines):,} sentences ({words:,} words) to {OUT_PATH}")


if __name__ == "__main__":
    main()
