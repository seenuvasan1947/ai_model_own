"""School-style classes. Each class makes lessons (text to learn) and quiz items (prompt -> answer)."""
import random
import string

LOW = string.ascii_lowercase
UP = string.ascii_uppercase
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
             "eighteen", "nineteen", "twenty"]
WORDS = ["cat", "dog", "sun", "cup", "bag", "hat", "pen", "box", "bus", "red", "big", "hot",
         "run", "sit", "map", "egg", "ant", "bed", "fan", "jam", "bat", "car", "fish", "milk",
         "ball", "tree", "book", "door", "hand", "girl", "boy", "cake", "frog", "star", "moon"]
OPPOSITES = [("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"), ("day", "night"),
             ("old", "new"), ("happy", "sad"), ("open", "close"), ("in", "out"), ("tall", "short"),
             ("wet", "dry"), ("good", "bad"), ("light", "dark"), ("hard", "soft"), ("full", "empty")]
SYNONYMS = [("big", "large"), ("small", "little"), ("happy", "glad"), ("fast", "quick"),
            ("begin", "start"), ("shut", "close"), ("end", "finish"), ("sad", "unhappy"),
            ("smart", "clever"), ("look", "see"), ("talk", "speak"), ("hard", "difficult")]
NOUNS = ["the cat", "the dog", "the boy", "the girl", "the bird", "my friend", "the teacher"]
VERBS = ["runs", "eats", "sleeps", "jumps", "reads", "sings", "plays"]
ADJS = ["big", "small", "happy", "fast", "tall", "kind", "red"]


# ---------------- class 1: letters and digits ----------------
def class1_lesson():
    r = random.random()
    if r < 0.2:
        return random.choice([" ".join(LOW), " ".join(UP), " ".join("0123456789")])
    i = random.randrange(25)
    if r < 0.45:
        return f"after {LOW[i]} comes {LOW[i + 1]}"
    if r < 0.6:
        return f"before {LOW[i + 1]} comes {LOW[i]}"
    if r < 0.8:
        c = random.randrange(26)
        return f"big {UP[c]} small {LOW[c]}"
    d = random.randrange(9)
    return f"after {d} comes {d + 1}"


def class1_quiz():
    r, i = random.random(), random.randrange(25)
    if r < 0.35:
        return f"after {LOW[i]} comes ", LOW[i + 1]
    if r < 0.55:
        return f"before {LOW[i + 1]} comes ", LOW[i]
    if r < 0.8:
        c = random.randrange(26)
        return f"big {UP[c]} small ", LOW[c]
    d = random.randrange(9)
    return f"after {d} comes ", str(d + 1)


# ---------------- class 2: spelling, words, counting ----------------
def class2_lesson():
    r = random.random()
    w = random.choice(WORDS)
    n = random.randrange(21)
    if r < 0.3:
        return f"{' '.join(w)} spells {w}"
    if r < 0.45:
        return f"{w} starts with {w[0]}"
    if r < 0.65:
        return f"{n} is {NUM_WORDS[n]}"
    if r < 0.85 and n < 20:
        return f"after {NUM_WORDS[n]} comes {NUM_WORDS[n + 1]}"
    return " ".join(NUM_WORDS[1:11])


def class2_quiz():
    r = random.random()
    w = random.choice(WORDS)
    n = random.randrange(20)
    if r < 0.35:
        return f"{' '.join(w)} spells ", w
    if r < 0.5:
        return f"{w} starts with ", w[0]
    if r < 0.75:
        return f"{n} is ", NUM_WORDS[n]
    return f"after {NUM_WORDS[n]} comes ", NUM_WORDS[n + 1]


# ---------------- class 3: math, sentences, opposites, synonyms ----------------
def _sum():
    a, b = random.randrange(11), random.randrange(11)
    if random.random() < 0.6:
        return f"{a} + {b} = ", str(a + b)
    a, b = max(a, b), min(a, b)
    return f"{a} - {b} = ", str(a - b)


def class3_lesson():
    r = random.random()
    if r < 0.35:
        p, a = _sum()
        return p + a
    if r < 0.55:
        x, y = random.choice(OPPOSITES)
        if random.random() < 0.5:
            x, y = y, x
        return f"the opposite of {x} is {y}"
    if r < 0.75:
        x, y = random.choice(SYNONYMS)
        return f"{x} means {y}"
    if r < 0.88:
        return f"{random.choice(NOUNS)} {random.choice(VERBS)}."
    return f"{random.choice(NOUNS)} is {random.choice(ADJS)}."


def class3_quiz():
    r = random.random()
    if r < 0.5:
        return _sum()
    if r < 0.75:
        x, y = random.choice(OPPOSITES)
        if random.random() < 0.5:
            x, y = y, x
        return f"the opposite of {x} is ", y
    x, y = random.choice(SYNONYMS)
    return f"{x} means ", y


# ---------------- class 4: punctuation and sentence shape ----------------
PUNCT_RULES = [
    "a full stop . ends a telling sentence.",
    "a question mark ? ends a question.",
    "an exclamation mark ! shows strong feeling.",
    "a comma , separates things in a list.",
    "a sentence starts with a capital letter.",
    "an apostrophe ' shows that something belongs to someone.",
    "quotation marks \" \" show the words someone says.",
    "a colon : comes before a list.",
]
QUESTIONS = ["where is the cat", "what is your name", "how old are you", "who is the teacher",
             "can you run fast", "is the dog big", "why is the sky blue", "when do you sleep"]
EXCLAIMS = ["wow", "help", "stop", "look out", "what a big dog", "hurray"]


def _statement():
    return f"{random.choice(NOUNS)} {random.choice(VERBS)}"


def class4_lesson():
    r = random.random()
    if r < 0.2:
        return random.choice(PUNCT_RULES)
    if r < 0.4:
        s = _statement()
        return s[0].upper() + s[1:] + "."
    if r < 0.6:
        q = random.choice(QUESTIONS)
        return q[0].upper() + q[1:] + "?"
    if r < 0.7:
        e = random.choice(EXCLAIMS)
        return e[0].upper() + e[1:] + "!"
    if r < 0.85:
        a, b, c = random.sample(WORDS, 3)
        return f"I have a {a}, a {b} and a {c}."
    n = random.choice(NOUNS)
    return f'{n[0].upper() + n[1:]} said, "I am {random.choice(ADJS)}."'


def class4_quiz():
    r = random.random()
    if r < 0.35:
        q = random.choice(QUESTIONS)
        return q[0].upper() + q[1:], "?"
    if r < 0.7:
        s = _statement()
        return s[0].upper() + s[1:], "."
    if r < 0.85:
        e = random.choice(EXCLAIMS)
        return e[0].upper() + e[1:], "!"
    return "a question mark ", "? ends a question."


# ---------------- class 5: grammar ----------------
SINGULAR = ["cat", "dog", "bird", "boy", "girl", "book", "tree", "apple", "egg", "ant", "box", "bus"]
PLURAL = {"box": "boxes", "bus": "buses"}
VERB_TENSES = [("walk", "walks", "walked"), ("play", "plays", "played"), ("jump", "jumps", "jumped"),
               ("eat", "eats", "ate"), ("run", "runs", "ran"), ("sing", "sings", "sang"),
               ("go", "goes", "went"), ("see", "sees", "saw"), ("read", "reads", "read")]
PRONOUNS = [("I", "am", "was"), ("you", "are", "were"), ("he", "is", "was"), ("she", "is", "was"),
            ("we", "are", "were"), ("they", "are", "were"), ("it", "is", "was")]


def _plural(n):
    return PLURAL.get(n, n + "s")


def _article(n):
    return "an" if n[0] in "aeiou" else "a"


def class5_lesson():
    r = random.random()
    n = random.choice(SINGULAR)
    if r < 0.2:
        return f"this is {_article(n)} {n}. these are {_plural(n)}."
    if r < 0.4:
        return f"one {n} is here. two {_plural(n)} are here."
    if r < 0.6:
        base, s, past = random.choice(VERB_TENSES)
        return f"today I {base}. today she {s}. yesterday she {past}."
    if r < 0.8:
        p, now, then = random.choice(PRONOUNS)
        adj = random.choice(ADJS)
        return f"{p} {now} {adj} now. {p} {then} {adj} yesterday."
    return f"{_article(n)} {n} is {random.choice(ADJS)}. the {_plural(n)} are {random.choice(ADJS)}."


def class5_quiz():
    r = random.random()
    n = random.choice(SINGULAR)
    if r < 0.3:
        return f"one {n} is here. two ", f"{_plural(n)} are here."
    if r < 0.6:
        base, s, past = random.choice(VERB_TENSES)
        return f"today I {base}. today she {s}. yesterday she ", f"{past}."
    p, now, then = random.choice(PRONOUNS)
    adj = random.choice(ADJS)
    return f"{p} {now} {adj} now. {p} ", f"{then} {adj} yesterday."


def grammar_lesson():
    """Base-language practice mixed into every learning step (the 'grammar anchor')."""
    return random.choice([class3_lesson, class4_lesson, class5_lesson, class5_lesson])()


CLASSES = [
    {"name": "class 1: letters and digits", "lesson": class1_lesson, "quiz": class1_quiz},
    {"name": "class 2: spelling, words, counting", "lesson": class2_lesson, "quiz": class2_quiz},
    {"name": "class 3: math, sentences, opposites, synonyms", "lesson": class3_lesson, "quiz": class3_quiz},
    {"name": "class 4: punctuation and sentence shape", "lesson": class4_lesson, "quiz": class4_quiz},
    {"name": "class 5: grammar (is/are, a/an, plurals, tense)", "lesson": class5_lesson, "quiz": class5_quiz},
]


# ---------------- classes from files: lessons/english/class06_*.txt, class07_*.txt, ... ----------------
# Quiz lines:  ?? prompt => answer            exact recall (early classes)
#              ?? prompt => answer | other    any listed answer is right (later classes: own words)
def _file_class(path):
    import os
    lessons, quiz = [], []
    para = []
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        if s.startswith("#"):
            continue
        if s.startswith("??") and "=>" in s:
            p, a = s[2:].split("=>", 1)
            quiz.append((p.strip() + " ", a.strip()))
            continue
        if s:
            para.append(s)
        elif para:
            lessons.append(" ".join(para))
            para = []
    if para:
        lessons.append(" ".join(para))
    # quiz items are also taught, like homework answers (first answer form)
    lessons += [p + a.split("|")[0].strip() for p, a in quiz]
    if not lessons:
        return None

    def lesson():
        return random.choice(lessons)

    def quiz_fn():
        if quiz:
            return random.choice(quiz)
        # no quiz items: continue a lesson by one word
        words = random.choice(lessons).split()
        if len(words) < 2:
            return "", words[0]
        k = random.randrange(1, len(words))
        return " ".join(words[:k]) + " ", words[k]

    name = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
    return {"name": name, "lesson": lesson, "quiz": quiz_fn}


def all_classes(english_dir):
    import glob
    import os
    out = list(CLASSES)
    for path in sorted(glob.glob(os.path.join(english_dir, "*.txt"))):
        c = _file_class(path)
        if c:
            out.append(c)
    return out
