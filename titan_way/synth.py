"""Synthetic (question, memory, answer) data for training the decoder.

The decoder learns to WRITE a sentence from a question plus what the memory
recalled ("M:"). Entity names are random letter strings, so the model cannot
memorise names; it has to copy them from the prompt.
"""
import random

CONS, VOW = "bcdfghklmnprstvz", "aeiou"


ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _word(rng):
    if rng.random() < 0.5:   # pronounceable
        w = "".join(rng.choice(CONS) + rng.choice(VOW) for _ in range(rng.randint(2, 4)))
        return w + (rng.choice(CONS) if rng.random() < 0.3 else "")
    return "".join(rng.choice(ALPHA) for _ in range(rng.randint(3, 11)))  # any letters


def name(rng, cap=True):
    w = _word(rng)
    if rng.random() < 0.15:  # two-word names ("New Delhi", "Guido van Rossum" style)
        w += " " + _word(rng)
    return w.title() if cap else w


def prompt(q, m):
    return f"Q:{q}\nM:{m}\nA:"


def sample(rng, kind=None):
    """Returns (question, memory_string, answer_sentence)."""
    kind = kind or rng.choice(["capital", "capital", "creator", "boiling", "isa",
                               "copy", "two", "none"])
    X, Y, Z, W = (name(rng) for _ in range(4))
    if kind == "capital":
        q = rng.choice([f"what is the capital of {X}?", f"capital of {X}?",
                        f"which city is the capital of {X}?", f"tell me the capital of {X}?"])
        return q, Y, f"The capital of {X} is {Y}."
    if kind == "creator":
        q = rng.choice([f"who created {X}?", f"who is the creator of {X}?", f"creator of {X}?"])
        return q, f"{Y} {Z}", f"{Y} {Z} created {X}."
    if kind == "boiling":
        n = str(rng.randint(10, 999))
        q = rng.choice([f"what is the boiling point of {X}?", f"boiling point of {X}?"])
        return q, f"{n} degrees", f"The boiling point of {X} is {n} degrees."
    if kind == "isa":
        q = rng.choice([f"what is {X.lower()}?", f"what is a {X.lower()}?", f"define {X.lower()}?"])
        y = name(rng, cap=False)
        return q, y, f"{X.lower().capitalize()} is {y}."
    if kind == "copy":
        s = f"{X} produces {Y} for the {Z.lower()}."
        q = rng.choice([f"tell me about {X}?", f"what does {X} do?", f"{X}?"])
        return q, s, s
    if kind == "two":
        return (f"capital of {X} and {Z}?", f"{Y} ; {W}",
                f"The capital of {X} is {Y}, and the capital of {Z} is {W}.")
    return f"what is the capital of {X}?", "NONE", "I don't know."


def dataset(n, seed):
    rng = random.Random(seed)
    return [sample(rng) for _ in range(n)]


# real, human-readable checks the model never saw in training
REAL = [
    ("what is the capital of India?", "New Delhi", "The capital of India is New Delhi."),
    ("who created Python?", "Guido van Rossum", "Guido van Rossum created Python."),
    ("capital of Peru?", "Lima", "The capital of Peru is Lima."),
    ("what is the boiling point of water?", "100 degrees", "The boiling point of water is 100 degrees."),
    ("capital of Spain and Chile?", "Madrid ; Santiago",
     "The capital of Spain is Madrid, and the capital of Chile is Santiago."),
    ("tell me about Mitochondria?", "Mitochondria produce energy for the cell.",
     "Mitochondria produce energy for the cell."),
    ("what is the capital of Atlantis?", "NONE", "I don't know."),
]
