"""Expand the Q&A-behavior training set programmatically.

REVISED to only generate LANGUAGE-COMPETENCE Q&A pairs (arithmetic patterns,
opposites-as-vocabulary, calendar sequences, plurals). No world-facts
(capitals, planets, "the sky is blue") -- those belong in Layer 2/3 and
are tested via eval_qa_continual.py after being fed through continual_read.py.

The Q&A-behavior set's purpose is to teach the *skill of answering questions*
(the "A:" pattern), NOT to teach specific factual knowledge. Both are needed
in Layer 1, but only the behavior skill -- so this script deliberately keeps
the pattern-teaching content and drops factual content.

Output: data/qa_behavior.txt (overwrites, keeps all original hand-written pairs
at the top so nothing is lost, then appends the generated pairs).
Run:    python data_prep/expand_qa.py
"""

import json
import os
import random


ORIG_PATH = "data/qa_behavior.txt"
EVAL_PATH = "data/eval_qa.jsonl"

random.seed(1)


def load_eval_questions():
    if not os.path.exists(EVAL_PATH):
        return set()
    return {json.loads(l)["question"].lower() for l in open(EVAL_PATH) if l.strip()}


def load_original():
    if not os.path.exists(ORIG_PATH):
        return ""
    return open(ORIG_PATH, "r", encoding="utf-8").read().rstrip() + "\n\n"


NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine", "ten", "eleven", "twelve"]


def arith_qa():
    out = []
    for _ in range(300):
        a, b = random.randint(0, 6), random.randint(0, 6)
        s = a + b
        if s > 12:
            continue
        out.append((f"What is {NUMBERS[a]} plus {NUMBERS[b]}?",
                    f"{NUMBERS[a].capitalize()} plus {NUMBERS[b]} is {NUMBERS[s]}."))
    for _ in range(300):
        a, b = random.randint(0, 12), random.randint(0, 12)
        if a < b:
            continue
        d = a - b
        if d > 12:
            continue
        out.append((f"What is {NUMBERS[a]} minus {NUMBERS[b]}?",
                    f"{NUMBERS[a].capitalize()} minus {NUMBERS[b]} is {NUMBERS[d]}."))
    return out


def opposite_qa():
    pairs = [("hot", "cold"), ("big", "small"), ("tall", "short"),
             ("fast", "slow"), ("happy", "sad"), ("up", "down"),
             ("in", "out"), ("open", "closed"), ("light", "dark"),
             ("hard", "soft"), ("new", "old"), ("full", "empty"),
             ("wet", "dry"), ("strong", "weak"), ("clean", "dirty")]
    out = []
    for a, b in pairs:
        out.append((f"What is the opposite of {a}?",
                    f"The opposite of {a} is {b}."))
        out.append((f"What is the opposite of {b}?",
                    f"The opposite of {b} is {a}."))
    return out


def calendar_qa():
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    out = []
    for i, d in enumerate(days):
        nxt = days[(i+1) % 7]
        out.append((f"What day comes after {d}?", f"The day after {d} is {nxt}."))
    for i, m in enumerate(months):
        nxt = months[(i+1) % 12]
        out.append((f"What month comes after {m}?", f"The month after {m} is {nxt}."))
    return out


def plural_qa():
    """Q&A patterns exercising English plurals (grammar, not facts)."""
    pairs = [("boy", "boys"), ("girl", "girls"), ("book", "books"),
             ("child", "children"), ("man", "men"), ("woman", "women"),
             ("mouse", "mice"), ("foot", "feet"), ("tree", "trees"),
             ("box", "boxes"), ("leaf", "leaves"), ("wolf", "wolves"),
             ("baby", "babies"), ("city", "cities")]
    out = []
    for sing, plur in pairs:
        out.append((f"What is the plural of {sing}?", f"The plural of {sing} is {plur}."))
        out.append((f"What is the singular of {plur}?", f"The singular of {plur} is {sing}."))
    return out


def main():
    forbidden = load_eval_questions()
    print(f"Excluding {len(forbidden)} exact questions from held-out eval set.")

    generated = []
    generated += arith_qa()
    generated += opposite_qa()
    generated += calendar_qa()
    generated += plural_qa()

    # Filter out any generated Q that verbatim-matches the eval set (extreme
    # caution -- keyword overlap in eval scoring means near-matches don't cause
    # leakage, but exact matches would obviously invalidate the eval).
    kept = [(q, a) for q, a in generated if q.lower() not in forbidden]
    dropped = len(generated) - len(kept)
    if dropped:
        print(f"Dropped {dropped} generated Qs that overlapped exactly with eval set.")

    random.shuffle(kept)

    original = load_original()
    with open(ORIG_PATH, "w", encoding="utf-8") as f:
        f.write(original)
        for q, a in kept:
            f.write(f"Q: {q}\nA: {a}\n\n")

    print(f"Wrote {len(kept):,} generated Q&A pairs (on top of original) to {ORIG_PATH}")


if __name__ == "__main__":
    main()
