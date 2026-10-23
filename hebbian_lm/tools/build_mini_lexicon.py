#!/usr/bin/env python3
"""Generate hebbian_lm/resources/mini_lexicon.json: a small hand-curated dictionary (about 400 entries) that
covers the demo world and common everyday words, so the vocabulary engine works offline.

    python tools/build_mini_lexicon.py
"""

import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hebbian_lm", "resources",
                   "mini_lexicon.json")
E = []


def add(lemma, pos, kind="", gloss="", hyp=(), ant=(), syn=(), proper=False, ex=()):
    d = dict(lemma=lemma, pos=pos, kind=kind, gloss=gloss, hypernyms=list(hyp), antonyms=list(ant),
             synonyms=list(syn), examples=list(ex))
    if proper:
        d["proper"] = True
    E.append(d)


# ---- top of the ontology
for lemma, hyp, gloss in [
    ("thing", [], "anything that exists"), ("organism", ["thing"], "a living thing"),
    ("animal", ["organism"], "a living creature that moves and eats"), ("plant", ["organism"], "a living thing that grows in the ground"),
    ("person", ["organism"], "a human being"), ("mammal", ["animal"], "an animal that feeds its young milk"),
    ("bird", ["animal"], "an animal with feathers and wings"), ("insect", ["animal"], "a small animal with six legs"),
    ("reptile", ["animal"], "a cold-blooded animal with scales"), ("amphibian", ["animal"], "an animal that lives in water and on land"),
    ("fish", ["animal"], "an animal that lives and breathes in water"), ("arachnid", ["animal"], "an animal with eight legs"),
    ("canine", ["mammal"], "a doglike mammal"), ("feline", ["mammal"], "a catlike mammal"), ("equine", ["mammal"], "a horselike mammal"),
    ("bovine", ["mammal"], "a cowlike mammal"), ("rodent", ["mammal"], "a small gnawing mammal"), ("primate", ["mammal"], "a monkey, ape or human"),
    ("food", ["thing"], "something people or animals eat"), ("fruit", ["food"], "sweet food that grows on a plant"),
    ("vegetable", ["food"], "a plant eaten as food"), ("drink", ["thing"], "a liquid you drink"),
    ("object", ["thing"], "a thing you can touch"), ("furniture", ["object"], "things like tables and chairs in a room"),
    ("container", ["object"], "something that holds things"), ("toy", ["object"], "a thing to play with"),
    ("place", ["thing"], "a location"), ("building", ["place"], "a place with walls and a roof"),
    ("color", ["property"], "what you see when light is red, blue, green and so on"), ("property", ["thing"], "a quality of a thing"),
    ("sound", ["thing"], "what you hear"), ("body part", ["thing"], "a part of a body"), ("time", ["thing"], "a period or moment"),
    ("substance", ["thing"], "a kind of matter"), ("weather", ["thing"], "conditions of the air outside"),
]:
    add(lemma, "n", "abstract" if lemma in ("thing", "property", "color", "sound", "time") else lemma if lemma in ("animal", "plant", "person", "food", "drink", "place", "object") else "", gloss, hyp)

ANIMALS = {"dog": ["canine"], "cat": ["feline"], "bird": ["animal"], "cow": ["bovine"], "horse": ["equine"], "duck": ["bird"],
           "pig": ["mammal"], "frog": ["amphibian"], "rabbit": ["mammal"], "lion": ["feline"], "bear": ["mammal"],
           "monkey": ["primate"], "mouse": ["rodent"], "elephant": ["mammal"], "owl": ["bird"], "zebra": ["equine"],
           "sparrow": ["bird"], "kitten": ["feline"], "puppy": ["canine"], "spider": ["arachnid"], "ant": ["insect"],
           "bee": ["insect"], "snake": ["reptile"], "wolf": ["canine"], "tiger": ["feline"], "giraffe": ["mammal"],
           "sheep": ["mammal"], "goat": ["mammal"], "fox": ["canine"], "squirrel": ["rodent"], "turtle": ["reptile"],
           "chicken": ["bird"], "eagle": ["bird"], "dolphin": ["mammal"], "whale": ["mammal"], "shark": ["fish"], "butterfly": ["insect"]}
GLOSS = {"dog": "a domestic animal that barks", "cat": "a small furry animal that purrs", "cow": "a farm animal that gives milk",
         "horse": "a large animal people ride", "duck": "a bird that swims", "zebra": "a striped horselike animal of Africa",
         "spider": "a small animal with eight legs that spins webs", "ant": "a tiny insect that lives in colonies",
         "bird": "an animal with feathers that can usually fly", "frog": "a small animal that hops and lives near water"}
for a, h in ANIMALS.items():
    add(a, "n", "animal", GLOSS.get(a, f"a kind of {h[0]}"), h + ([] if h == ["animal"] else ["animal"]))
PEOPLE = {"boy": "a young male person", "girl": "a young female person", "man": "an adult male person",
          "woman": "an adult female person", "child": "a young person", "teacher": "a person who teaches",
          "baby": "a very young child", "friend": "a person you like", "doctor": "a person who treats sick people",
          "farmer": "a person who grows crops or keeps animals", "king": "a male ruler", "queen": "a female ruler",
          "mother": "a female parent", "father": "a male parent", "brother": "a male sibling", "sister": "a female sibling"}
for p, g in PEOPLE.items():
    add(p, "n", "person", g, ["person"])
for name in ["tom", "anna", "ben", "lily", "sam", "mia", "max", "emma", "jack", "zoe"]:
    add(name, "n", "person", "a first name", ["person"], proper=True)
FRUITS = ["apple", "banana", "pear", "peach", "lemon", "strawberry", "cherry", "orange", "grape", "plum"]
for f in FRUITS:
    add(f, "n", "food", "a kind of fruit", ["fruit", "food"])
for v in ["carrot", "tomato", "corn", "potato", "onion", "pea"]:
    add(v, "n", "food", "a vegetable", ["vegetable", "food"])
for f in ["cake", "cookie", "egg", "sandwich", "bread", "rice", "soup", "chocolate", "cheese", "meat", "pizza"]:
    add(f, "n", "food", "something to eat", ["food"])
for d in ["milk", "water", "juice", "tea", "coffee"]:
    add(d, "n", "drink", "something to drink", ["drink"])
for o in ["ball", "book", "car", "door", "hat", "pen", "bag", "box", "cup", "kite", "bike", "boat", "clock", "lamp", "bottle", "key"]:
    add(o, "n", "object", "a thing you can hold or use", ["container" if o in ("box", "bag", "cup", "bottle") else "object", "object"])
add("toy", "n", "object", "a thing to play with", ["object"])
for o in ["table", "chair", "bed", "desk", "sofa"]:
    add(o, "n", "object", "a piece of furniture", ["furniture", "object"])
for pl in ["park", "garden", "farm", "school", "house", "room", "shop", "pond", "nest", "forest", "jungle", "river", "kitchen", "zoo", "street", "city", "village", "beach", "library", "field"]:
    add(pl, "n", "place", "a place", ["place"])
add("tree", "n", "plant", "a tall plant with a trunk", ["plant"])
for pl in ["grass", "rose", "flower", "leaf", "seed"]:
    add(pl, "n", "plant", "a part of a plant or a plant", ["plant"])
for n, k, g, h in [("sky", "natural", "the space above the earth", ["place"]), ("sun", "natural", "the star that gives us light", ["thing"]),
                   ("snow", "natural", "frozen rain that falls in flakes", ["weather"]), ("coal", "substance", "a black rock that burns", ["substance"]),
                   ("sea", "natural", "a large body of salt water", ["place"]), ("ocean", "natural", "a very large sea", ["sea", "place"]),
                   ("cloud", "natural", "white or grey mass in the sky", ["weather"]), ("sand", "substance", "tiny grains of rock on a beach", ["substance"]),
                   ("blood", "body", "the red liquid in the body", ["substance"]), ("night", "time", "the dark part of the day", ["time"]),
                   ("day", "time", "the time when it is light", ["time"]), ("rain", "natural", "water falling from clouds", ["weather"]),
                   ("moon", "natural", "the body that goes around the earth", ["thing"]), ("star", "natural", "a point of light in the night sky", ["thing"])]:
    add(n, "n", k, g, h)
for part in ["leg", "arm", "wing", "tail", "eye", "ear", "nose", "mouth", "hand", "foot", "head", "paw"]:
    add(part, "n", "body", "a part of the body", ["body part"])
COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown", "pink", "gray", "grey", "orange", "purple"]
for c in COLORS:
    add(c, "a", "color", f"the colour {c}", ["color"])
    add(c, "n", "color", f"the colour {c}", ["color"])
SOUNDS = ["woof", "meow", "moo", "quack", "oink", "roar", "neigh", "ribbit", "hoot", "tweet", "squeak", "growl", "trumpet", "bark", "purr", "buzz", "hiss", "baa"]
for s in SOUNDS:
    add(s, "n", "sound", "an animal sound", ["sound"])
# ---- adjectives (with antonyms)
ANT = [("big", "small"), ("big", "little"), ("hot", "cold"), ("fast", "slow"), ("happy", "sad"), ("tall", "short"), ("good", "bad"),
       ("open", "closed"), ("clean", "dirty"), ("wet", "dry"), ("hard", "soft"), ("full", "empty"), ("old", "new"), ("old", "young"),
       ("long", "short"), ("heavy", "light"), ("loud", "quiet"), ("strong", "weak"), ("rich", "poor"), ("early", "late"),
       ("hungry", "full"), ("sweet", "sour"), ("up", "down"), ("in", "out")]
adj = {}
for a, b in ANT:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)
for w in ["tired", "kind", "little", "brave", "angry", "funny", "quick", "smart", "shy", "lazy", "busy", "cute", "gentle", "wild"]:
    adj.setdefault(w, set())
for w, ants in sorted(adj.items()):
    add(w, "a", "", f"having the quality {w}", [], sorted(ants))
add("day", "n", "time", "the time when it is light", ["time"], ["night"])
# ---- verbs
VERBS = {"run": "to move fast on foot", "sleep": "to rest with the eyes closed", "jump": "to push off the ground into the air",
         "walk": "to move on foot", "play": "to do something for fun", "sit": "to rest on a seat", "fly": "to move through the air",
         "swim": "to move through water", "hop": "to jump on one leg or with small jumps", "talk": "to speak", "laugh": "to make a happy sound",
         "smile": "to show happiness with the mouth", "work": "to do a job", "eat": "to take food into the body", "drink": "to take liquid into the body",
         "read": "to look at words and understand them", "see": "to notice with the eyes", "like": "to enjoy", "love": "to like very much",
         "open": "to move a door or lid so it is not shut", "carry": "to hold and move something", "throw": "to send through the air",
         "kick": "to hit with the foot", "help": "to make easier for someone", "feed": "to give food to", "live": "to have a home somewhere",
         "say": "to make a sound or words", "have": "to own or hold", "chase": "to run after", "catch": "to take hold of something moving",
         "bite": "to cut with the teeth", "find": "to discover", "hold": "to keep in the hands", "give": "to hand over", "take": "to get hold of",
         "make": "to build or create", "build": "to put parts together", "paint": "to put colour on", "write": "to put words on paper",
         "sing": "to make music with the voice", "dance": "to move to music", "cook": "to prepare food with heat", "buy": "to pay for",
         "watch": "to look at for a time", "ride": "to sit on and travel", "push": "to move away with force", "pull": "to move towards you",
         "climb": "to go up", "drive": "to control a car", "go": "to move from a place", "come": "to move towards a place", "want": "to wish for",
         "know": "to have in the mind", "think": "to use the mind", "wash": "to clean with water", "bark": "to make a dog sound"}
for v, g in VERBS.items():
    add(v, "v", "verb", g)
for adv in ["quickly", "slowly", "happily", "loudly", "quietly", "always", "never", "often", "very", "too", "again", "here", "there", "now", "today"]:
    add(adv, "r", "", "an adverb")
for w, g in [("opposite", "a word with the contrary meaning"), ("antonym", "a word that means the opposite"),
             ("synonym", "a word that means the same"), ("meaning", "what a word stands for"), ("word", "a unit of language"),
             ("kind", "a type or sort of thing"), ("number", "a count or amount"), ("name", "what something is called"),
             ("size", "how big something is"), ("shape", "the outline of a thing"), ("answer", "a reply to a question"),
             ("question", "a sentence that asks something")]:
    add(w, "n", "abstract", g, ["thing"])
add("mean", "v", "verb", "to stand for")
# ---- words for reading short factual passages (events, history, travel)
for w, k, g, h in [
    ("soldier", "person", "a person who serves in an army", ["person"]), ("army", "group", "a large organised group of soldiers", ["thing"]),
    ("officer", "person", "a person with authority in an army or a company", ["person"]), ("leader", "person", "a person who leads others", ["person"]),
    ("rebel", "person", "a person who fights against the rulers", ["person"]), ("rebellion", "event", "a fight against the rulers", ["thing"]),
    ("mutiny", "event", "a rebellion by soldiers", ["thing"]), ("king", "person", "a man who rules a country", ["person"]),
    ("queen", "person", "a woman who rules a country", ["person"]), ("crown", "object", "a ring worn by a king or queen; the ruling power", ["object"]),
    ("company", "group", "a business; a group of people working together", ["thing"]), ("captain", "person", "the person in charge of a ship", ["person"]),
    ("baker", "person", "a person who makes bread", ["person"]), ("bakery", "building", "a place where bread is made", ["building"]),
    ("inventor", "person", "a person who makes new things", ["person"]), ("astronaut", "person", "a person who travels in space", ["person"]),
    ("brother", "person", "a boy or man with the same parents", ["person"]), ("survivor", "person", "a person who lives through a disaster", ["person"]),
    ("fire", "event", "burning that gives off heat and light", ["thing"]), ("flame", "thing", "the glowing part of a fire", ["thing"]),
    ("wind", "weather", "air that moves", ["weather"]), ("ocean", "place", "a very large sea", ["place"]), ("sea", "place", "a large body of salt water", ["place"]),
    ("island", "place", "land with water all around", ["place"]), ("city", "place", "a large town", ["place"]), ("country", "place", "a land with its own people", ["place"]),
    ("museum", "building", "a place where old and important things are shown", ["building"]), ("church", "building", "a building for prayer", ["building"]),
    ("cathedral", "building", "a large important church", ["building"]), ("diary", "object", "a book in which you write each day", ["object"]),
    ("house", "building", "a building to live in", ["building"]), ("ship", "object", "a large boat", ["object"]), ("iceberg", "object", "a huge piece of floating ice", ["object"]),
    ("disaster", "event", "a very bad event", ["thing"]), ("war", "event", "a fight between countries", ["thing"]), ("battle", "event", "a fight between armies", ["thing"]),
    ("voyage", "event", "a long journey by sea", ["thing"]), ("flight", "event", "a journey through the air", ["thing"]), ("airplane", "object", "a machine that flies", ["object"]),
    ("machine", "object", "a thing with parts that does work", ["object"]), ("engine", "object", "a machine that makes things move", ["object"]),
    ("rocket", "object", "a vehicle that flies into space", ["object"]), ("spacecraft", "object", "a vehicle that travels in space", ["object"]),
    ("moon", "place", "the large round object that circles the earth", ["place"]), ("earth", "place", "the planet we live on", ["place"]),
    ("rock", "object", "a piece of stone", ["object"]), ("photograph", "object", "a picture made with a camera", ["object"]), ("television", "object", "a screen that shows programmes", ["object"]),
    ("surface", "thing", "the outside or top of something", ["thing"]), ("glider", "object", "a plane without an engine", ["object"]),
    ("lifeboat", "object", "a small boat for saving people from a ship", ["object"]), ("cartridge", "object", "a case holding a bullet and powder", ["object"]),
    ("rifle", "object", "a long gun", ["object"]), ("fat", "substance", "an oily substance from animals", ["substance"]), ("sand", "substance", "tiny grains of rock", ["substance"]),
    ("dune", "place", "a hill of sand", ["place"]), ("rumour", "thing", "a story that may not be true", ["thing"]), ("control", "abstract", "power over something", ["thing"]),
    ("diary", "object", "a book in which you write each day", ["object"]), ("people", "person", "men, women and children", ["person"]),
    ("hour", "time", "sixty minutes", ["time"]), ("minute", "time", "sixty seconds", ["time"]), ("second", "time", "a very short time", ["time"]),
    ("day", "time", "twenty-four hours", ["time"]), ("week", "time", "seven days", ["time"]), ("month", "time", "about thirty days", ["time"]),
    ("year", "time", "twelve months", ["time"]), ("night", "time", "the dark part of the day", ["time"]), ("morning", "time", "the early part of the day", ["time"]),
]:
    add(w, "n", k, g, h)
for w, g in [("start", "to begin"), ("begin", "to start"), ("end", "to finish"), ("finish", "to come to an end"), ("burn", "to be on fire"),
             ("attack", "to hurt or fight someone"), ("march", "to walk with steady steps"), ("lead", "to go in front and guide"), ("take", "to get hold of"),
             ("write", "to put words on paper"), ("design", "to plan how something will look"), ("order", "to tell someone to do something"),
             ("pull", "to draw towards you"), ("stop", "to end or prevent"), ("spread", "to reach over a wider area"), ("destroy", "to ruin completely"),
             ("rescue", "to save from danger"), ("command", "to be in charge of"), ("hit", "to strike"), ("sink", "to go down under water"),
             ("die", "to stop living"), ("leave", "to go away from"), ("carry", "to hold and move"), ("land", "to come down to the ground"),
             ("launch", "to send off"), ("fly", "to move through the air"), ("stay", "to remain"), ("collect", "to gather"), ("watch", "to look at"),
             ("return", "to come or go back"), ("refuse", "to say no to"), ("serve", "to work for"), ("rule", "to be in charge of a country"),
             ("forget", "to fail to remember"), ("call", "to give a name to"), ("make", "to build or create"), ("build", "to make by putting parts together"),
             ("test", "to try something to see how it works"), ("work", "to do a job"), ("become", "to begin to be"), ("cover", "to go over a distance"),
             ("last", "to continue for a time"), ("display", "to put on show"), ("use", "to do something with a thing for a purpose"), ("find", "to come across"),
             ("grease", "to put oil or fat on"), ("happen", "to take place"), ("cause", "to make something happen"), ("say", "to speak words"), ("step", "to put down a foot"), ("fall", "to drop down"), ("die", "to stop living")]:
    add(w, "v", "verb", g)
for w, g in [("angry", "feeling cross"), ("strong", "having great force"), ("wooden", "made of wood"), ("large", "big"), ("first", "coming before all others"),
             ("new", "not old"), ("old", "not new"), ("safe", "not in danger"), ("successful", "having done well"), ("same", "not different"),
             ("lunar", "of the moon"), ("small", "little"), ("great", "very big or important"), ("british", "of Britain"), ("indian", "of India")]:
    add(w, "a", "adjective", g)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"entries": E}, f, indent=0, ensure_ascii=False)
print(f"wrote {len(E)} entries to {OUT}")
