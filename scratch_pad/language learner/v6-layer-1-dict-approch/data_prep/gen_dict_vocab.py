"""Generate a CLOSED-VOCABULARY dictionary corpus for Layer 1.

This is the missing piece the dict-approach needs on top of gen_synthetic.py
(which already covers numbers/arithmetic/calendar/opposites/grammar
patterns with no world-facts). This file adds:

  1. A fixed core-vocabulary word list (WORDS below) with a hand-written,
     one-line DEFINITION per word -- dictionary-style, not encyclopedic
     fact. "A dog is an animal that barks and lives with people." teaches
     word MEANING + grammar together, the same way a dictionary entry does.
  2. Each word used in 2-3 template SENTENCES so Layer 1 sees it in
     context, not just in isolation (isolated word+definition pairs teach
     weak contextual usage).
  3. Special-character / punctuation drill sentences (comma, period,
     question mark, exclamation mark, apostrophe, quotes, colon,
     semicolon, hyphen) since the user's design explicitly calls these
     out as something Layer 1 must master.

Deliberately NOT included (belongs in Layer 2/3 via continual_read.py):
  - named entities, historical facts, specific people/places/dates
  - domain/technical vocabulary (medical, legal, code) -- by design, so
    the "what happens with an unseen word" boundary is easy to test:
    anything not in WORDS below is, by construction, out-of-vocabulary
    for Layer 1's training.

Output: data/dict_vocab.txt
Run:    python data_prep/gen_dict_vocab.py
"""

import os

OUT_PATH = "data/dict_vocab.txt"

# word -> (part_of_speech, one-line definition, [example sentences])
# Kept intentionally small-but-real (~140 words) rather than a padded
# 1000-word list with low-quality auto-generated definitions. Expand this
# dict directly to grow the controlled vocabulary -- that's the knob the
# "1000 words with contextual grammar and meaning" idea in the chat maps to.
WORDS = {
    "dog":    ("noun", "A dog is an animal that barks and lives with people.",
               ["The dog ran to the door.", "A small dog slept on the mat."]),
    "cat":    ("noun", "A cat is an animal that says meow and likes to sleep.",
               ["The cat sat on the chair.", "A black cat walked past the window."]),
    "book":   ("noun", "A book is a set of pages with words that you read.",
               ["She opened the book slowly.", "He put the book on the table."]),
    "water":  ("noun", "Water is a clear liquid that people and animals drink.",
               ["He drank a glass of water.", "The water in the river was cold."]),
    "fire":   ("noun", "Fire is heat and light that burns and gives off smoke.",
               ["The fire kept them warm.", "She lit a fire in the stove."]),
    "house":  ("noun", "A house is a building where people live.",
               ["They built a new house.", "The house had a red door."]),
    "run":    ("verb", "To run means to move very fast using your legs.",
               ["The boy ran to school.", "She runs every morning."]),
    "walk":   ("verb", "To walk means to move on foot at a normal speed.",
               ["They walked to the park.", "He walks his dog at night."]),
    "eat":    ("verb", "To eat means to put food in your mouth and swallow it.",
               ["She ate an apple.", "They eat dinner at six."]),
    "sleep":  ("verb", "To sleep means to rest with your eyes closed at night.",
               ["The baby slept for two hours.", "He sleeps early on weekdays."]),
    "speak":  ("verb", "To speak means to say words out loud.",
               ["She spoke softly.", "He speaks three languages."]),
    "write":  ("verb", "To write means to make words using a pen or a keyboard.",
               ["He wrote a letter.", "She writes in her notebook every day."]),
    "happy":  ("adjective", "Happy means feeling pleased or glad.",
               ["She felt happy about the news.", "The happy child laughed loudly."]),
    "sad":    ("adjective", "Sad means feeling unhappy or sorry.",
               ["He was sad after the game.", "A sad song played on the radio."]),
    "big":    ("adjective", "Big means large in size.",
               ["The big dog barked loudly.", "They live in a big house."]),
    "small":  ("adjective", "Small means little in size.",
               ["A small bird sat on the branch.", "She has a small bag."]),
    "fast":   ("adjective", "Fast means moving with great speed.",
               ["The fast car passed us.", "He is a fast runner."]),
    "slow":   ("adjective", "Slow means moving without much speed.",
               ["The slow turtle crossed the road.", "She walked at a slow pace."]),
    "number": ("noun", "A number is a word or symbol used for counting.",
               ["Pick a number between one and ten.", "Her favorite number is seven."]),
    "letter": ("noun", "A letter is one of the symbols used to write words, or a written message.",
               ["The word starts with the letter A.", "She sent him a letter."]),
    "word":   ("noun", "A word is a unit of language that has a meaning.",
               ["Can you spell that word?", "Each word in the sentence matters."]),
    "sentence": ("noun", "A sentence is a group of words that expresses a complete thought.",
               ["He wrote a short sentence.", "A sentence usually ends with a period."]),
    "question": ("noun", "A question is something you ask to get information.",
               ["She asked a good question.", "I have a question for you."]),
    "answer": ("noun", "An answer is what you say or write to reply to a question.",
               ["He gave the correct answer.", "What is your answer to this?"]),
    "friend": ("noun", "A friend is a person you like and trust.",
               ["She called her best friend.", "He is a good friend to me."]),
    "teacher": ("noun", "A teacher is a person who helps students learn.",
               ["The teacher explained the lesson.", "Our teacher is very kind."]),
    "school": ("noun", "A school is a place where children go to learn.",
               ["They walk to school together.", "The school has a big garden."]),
    "computer": ("noun", "A computer is an electronic device that stores and processes information.",
               ["She typed on the computer.", "The computer turned on slowly."]),
    "language": ("noun", "A language is a system of words and grammar that people use to communicate.",
               ["English is a widely spoken language.", "He is learning a new language."]),
    "morning": ("noun", "Morning is the early part of the day before noon.",
               ["She wakes up every morning at six.", "We will meet in the morning."]),
    "night":  ("noun", "Night is the part of the day when it is dark.",
               ["The stars shine at night.", "He works at night."]),
    "open":   ("verb", "To open means to move something so it is no longer closed.",
               ["She opened the window.", "Please open the box."]),
    "close":  ("verb", "To close means to shut something so it is no longer open.",
               ["He closed the door.", "Close your eyes and rest."]),
    "give":   ("verb", "To give means to hand something to someone.",
               ["She gave him a gift.", "Please give me the book."]),
    "take":   ("verb", "To take means to get or hold something in your hands.",
               ["He took the keys.", "Take a seat, please."]),
    "think":  ("verb", "To think means to use your mind to consider something.",
               ["She thought about the problem.", "I think it will rain."]),
    "know":   ("verb", "To know means to have information or understanding about something.",
               ["He knows the answer.", "Do you know her name?"]),
    "bird":   ("noun", "A bird is an animal with feathers and wings that can usually fly.",
               ["The bird flew over the tree.", "A small bird sang in the morning."]),
    "fish":   ("noun", "A fish is an animal that lives in water and breathes through gills.",
               ["The fish swam in the pond.", "He caught a fish in the river."]),
    "tree":   ("noun", "A tree is a tall plant with a trunk, branches, and leaves.",
               ["The tree grew near the house.", "Birds built a nest in the tree."]),
    "flower": ("noun", "A flower is the colorful part of a plant that often smells sweet.",
               ["She picked a red flower.", "The flower grew in the garden."]),
    "sun":    ("noun", "The sun is the bright star that gives light and heat during the day.",
               ["The sun rose in the morning.", "It was hot under the sun."]),
    "moon":   ("noun", "The moon is the bright object seen in the sky at night.",
               ["The moon shone over the lake.", "They watched the moon rise."]),
    "rain":   ("noun", "Rain is water that falls from clouds in the sky.",
               ["The rain fell all afternoon.", "She stayed inside because of the rain."]),
    "wind":   ("noun", "Wind is air that moves, sometimes strongly, outside.",
               ["The wind blew the leaves away.", "A cold wind came through the window."]),
    "road":   ("noun", "A road is a long path made for cars and people to travel on.",
               ["The car drove down the road.", "They crossed the busy road."]),
    "city":   ("noun", "A city is a large place where many people live and work.",
               ["He moved to a big city.", "The city had tall buildings."]),
    "food":   ("noun", "Food is what people and animals eat to live and grow.",
               ["They shared the food together.", "She cooked food for the family."]),
    "money":  ("noun", "Money is what people use to buy and sell things.",
               ["He saved his money carefully.", "She paid with money at the store."]),
    "door":   ("noun", "A door is a flat object that opens and closes to let people in or out.",
               ["He knocked on the door.", "The door was locked at night."]),
    "window": ("noun", "A window is an opening in a wall, usually with glass, that lets in light.",
               ["She looked out the window.", "The window was open all day."]),
    "table":  ("noun", "A table is a piece of furniture with a flat top used for eating or working.",
               ["They sat around the table.", "He put his book on the table."]),
    "chair":  ("noun", "A chair is a piece of furniture made for one person to sit on.",
               ["She sat in a comfortable chair.", "The chair was next to the table."]),
    "hand":   ("noun", "A hand is the part of the body at the end of the arm, used for holding things.",
               ["He held the cup with his hand.", "She waved her hand at them."]),
    "eye":    ("noun", "An eye is a part of the body used for seeing.",
               ["She closed her eyes to sleep.", "His eye was sharp and bright."]),
    "head":   ("noun", "The head is the top part of the body that contains the brain, eyes, and mouth.",
               ["He turned his head to look.", "She wore a hat on her head."]),
    "heart":  ("noun", "The heart is an organ in the body that pumps blood.",
               ["Her heart beat fast after running.", "He felt it deep in his heart."]),
    "time":   ("noun", "Time is how we measure moments, hours, and days passing.",
               ["They spent time together.", "It took a long time to finish."]),
    "day":    ("noun", "A day is a period of twenty-four hours, including daylight and night.",
               ["They met every day.", "It was a warm sunny day."]),
    "work":   ("verb", "To work means to do a job or activity that takes effort.",
               ["She works at a school.", "He worked hard all afternoon."]),
    "play":   ("verb", "To play means to do an activity for fun or enjoyment.",
               ["The children played outside.", "They play games together."]),
    "help":   ("verb", "To help means to make something easier for someone else.",
               ["She helped him carry the bag.", "Can you help me with this?"]),
    "make":   ("verb", "To make means to create or build something.",
               ["He made a wooden chair.", "She makes bread every morning."]),
    "see":    ("verb", "To see means to notice something with your eyes.",
               ["She saw a bird in the tree.", "Can you see the mountain from here?"]),
    "hear":   ("verb", "To hear means to notice a sound with your ears.",
               ["He heard a loud noise.", "Did you hear the music?"]),
    "feel":   ("verb", "To feel means to sense something physically or emotionally.",
               ["She felt cold in the wind.", "He feels happy today."]),
    "want":   ("verb", "To want means to desire or wish for something.",
               ["He wants a new book.", "They wanted to go outside."]),
    "need":   ("verb", "To need means to require something necessary.",
               ["She needs more time.", "They need water to survive."]),
    "like":   ("verb", "To like means to enjoy or have a positive feeling about something.",
               ["He likes reading books.", "Do you like this song?"]),
    "love":   ("verb", "To love means to feel deep affection for someone or something.",
               ["She loves her family.", "They love spending time outdoors."]),
    "start":  ("verb", "To start means to begin doing something.",
               ["The class starts at nine.", "He started his new job."]),
    "stop":   ("verb", "To stop means to end an action or movement.",
               ["The car stopped at the light.", "She stopped talking suddenly."]),
    "come":   ("verb", "To come means to move toward a place or person.",
               ["He came to the party.", "Please come inside."]),
    "go":     ("verb", "To go means to move away from a place.",
               ["She went to the market.", "They go to school every day."]),
    "good":   ("adjective", "Good means of high quality or pleasing.",
               ["She had a good day.", "This is a good book."]),
    "bad":    ("adjective", "Bad means of low quality or not pleasing.",
               ["It was a bad idea.", "The weather was bad yesterday."]),
    "hot":    ("adjective", "Hot means having a high temperature.",
               ["The soup was very hot.", "It is hot in summer."]),
    "cold":   ("adjective", "Cold means having a low temperature.",
               ["The water felt cold.", "Winter days are often cold."]),
    "new":    ("adjective", "New means recently made or not used before.",
               ["She bought a new phone.", "They moved into a new house."]),
    "old":    ("adjective", "Old means having existed for a long time.",
               ["He read an old letter.", "The old tree was very tall."]),
    "young":  ("adjective", "Young means not old; in an early part of life.",
               ["The young boy ran fast.", "She was young when she moved here."]),
    "strong": ("adjective", "Strong means having great physical power or force.",
               ["He is a strong swimmer.", "The strong wind broke the branch."]),
    "weak":   ("adjective", "Weak means lacking strength or power.",
               ["He felt weak after being sick.", "The weak signal made the call drop."]),
    "bright": ("adjective", "Bright means giving out or full of light.",
               ["The bright sun warmed the field.", "She wore a bright yellow shirt."]),
    "dark":   ("adjective", "Dark means having little or no light.",
               ["The room was dark at night.", "They walked through the dark forest."]),
    "quiet":  ("adjective", "Quiet means making little or no noise.",
               ["The library was quiet.", "She spoke in a quiet voice."]),
    "loud":   ("adjective", "Loud means making a lot of noise.",
               ["The music was very loud.", "He gave a loud shout."]),
    "soon":   ("adverb", "Soon means in a short time from now.",
               ["They will arrive soon.", "She will finish the work soon."]),
    "always": ("adverb", "Always means at all times, every time.",
               ["He always wakes up early.", "She always says thank you."]),
    "never":  ("adverb", "Never means at no time, not ever.",
               ["She never lies.", "He has never seen snow."]),
    "often":  ("adverb", "Often means happening many times or frequently.",
               ["They often walk to the park.", "She often reads before bed."]),
    "here":   ("adverb", "Here means in or at this place.",
               ["Please sit here.", "The book is here on the table."]),
    "there":  ("adverb", "There means in or at that place.",
               ["He put the box there.", "She lives over there."]),
    "today":  ("noun", "Today means this current day.",
               ["We have a lot to do today.", "Today is a sunny day."]),
    "tomorrow": ("noun", "Tomorrow means the day after today.",
               ["They will travel tomorrow.", "See you again tomorrow."]),
    "yesterday": ("noun", "Yesterday means the day before today.",
               ["It rained yesterday.", "She called me yesterday."]),
}

# Reuse the SAME closed categories gen_synthetic.py already exercises in
# grammar patterns, so these words get BOTH a dictionary-style definition
# here AND narrative usage there -- reinforcing the same word instead of
# splitting training signal across disjoint word sets.
CATEGORY_WORDS = {
    "cat": ("noun", "animal"), "horse": ("noun", "animal"), "cow": ("noun", "animal"),
    "sheep": ("noun", "animal"), "rabbit": ("noun", "animal"), "elephant": ("noun", "animal"),
    "lion": ("noun", "animal"), "tiger": ("noun", "animal"), "bear": ("noun", "animal"),
    "wolf": ("noun", "animal"), "fox": ("noun", "animal"), "mouse": ("noun", "animal"),
    "duck": ("noun", "animal"),
    "pen": ("noun", "object"), "paper": ("noun", "object"), "cup": ("noun", "object"),
    "plate": ("noun", "object"), "clock": ("noun", "object"), "phone": ("noun", "object"),
    "car": ("noun", "object"),
    "red": ("adjective", "color"), "blue": ("adjective", "color"), "green": ("adjective", "color"),
    "yellow": ("adjective", "color"), "black": ("adjective", "color"), "white": ("adjective", "color"),
    "orange": ("adjective", "color"), "purple": ("adjective", "color"), "pink": ("adjective", "color"),
    "brown": ("adjective", "color"), "gray": ("adjective", "color"),
    "kitchen": ("noun", "place"), "garden": ("noun", "place"), "park": ("noun", "place"),
    "library": ("noun", "place"), "office": ("noun", "place"), "store": ("noun", "place"),
}

CATEGORY_TEMPLATES = {
    "animal": "{word} is a kind of animal.",
    "object": "{word} is a common object people use.",
    "color":  "{word} is a color.",
    "place":  "{word} is a place people go to.",
}


def category_lines():
    out = []
    for word, (pos, category) in CATEGORY_WORDS.items():
        out.append(CATEGORY_TEMPLATES[category].format(word=word.capitalize()))
        out.append(f'"{word}" is a {pos}.')
    return out


PUNCT_LINES = [
    "A period ends a statement, like in this sentence.",
    "A question mark ends a question, like this one?",
    "An exclamation mark shows strong feeling, like this!",
    "A comma separates items in a list, such as apples, oranges, and pears.",
    "A comma can also separate two clauses, for example: she was tired, but she kept walking.",
    "An apostrophe shows a contraction, such as don't, can't, and isn't.",
    "An apostrophe also shows possession, such as the dog's bone or Maria's book.",
    "Quotation marks show someone's exact words, such as she said, \"I am ready.\"",
    "A colon introduces a list or explanation: apples, oranges, and pears.",
    "A semicolon joins two related sentences; it is stronger than a comma.",
    "A hyphen joins two words into one idea, such as well-known or ten-year-old.",
    "Parentheses add extra information (like this) inside a sentence.",
]


def word_lines():
    out = []
    for word, (pos, definition, examples) in WORDS.items():
        out.append(definition)
        out.append(f'"{word}" is a {pos}.')
        out.extend(examples)
    return out


def main():
    lines = []
    lines += word_lines()
    lines += category_lines()
    lines += PUNCT_LINES

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    words_total = sum(len(l.split()) for l in lines)
    print(f"Wrote {len(lines):,} lines ({words_total:,} words) to {OUT_PATH}")
    print(f"Core vocabulary: {len(WORDS)} words with definitions + usage examples.")
    print("Anything NOT in WORDS is out-of-vocabulary for Layer 1 by design --")
    print("that's the boundary to test; new words go through continual_read.py instead.")


if __name__ == "__main__":
    main()
