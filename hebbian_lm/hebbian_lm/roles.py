"""Roles of the phrases around a verb: when, where, with what, about what, how long, from where.

"The mutiny started in Meerut on 10 May 1857" is one event with two adjuncts. The reader keeps them on the fact as roles
(``where = in meerut``, ``when = on 10 may 1857``) so that "where did the mutiny start ?" and "when ?" are answered from the same
fact, and "what happened in 1858 ?" can be answered by looking for events whose time matches.

Word lists here are the small closed classes of time and place words and verb synonyms (vocabulary, like the number words); whether
a noun is a place otherwise comes from the dictionary.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .numbers import WORD_VALUE, parse_number

MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
SEASONS = {"spring", "summer", "autumn", "fall", "winter"}
TIME_NOUNS = {"morning", "evening", "night", "afternoon", "noon", "midnight", "day", "week", "month", "year", "century", "today",
              "yesterday", "tomorrow", "dawn", "dusk", "moment", "while", "time", "age", "era", "decade"}
TIME_UNITS = {"second", "minute", "hour", "day", "week", "month", "year", "decade", "century", "seconds", "minutes", "hours", "days",
              "weeks", "months", "years", "decades", "centuries"}
TIME_PREPS = {"in", "on", "at", "during", "after", "before", "until", "till", "since", "by", "around"}
SPATIAL_PREPS = {"in", "at", "on", "near", "under", "inside", "outside", "behind", "beside", "above", "below", "across", "through",
                 "over", "toward", "towards", "into", "onto", "to", "along", "around", "between", "among", "beyond", "within"}
APPROX = {"about", "around", "nearly", "almost", "roughly", "approximately", "over", "some", "only", "just"}
NOT_PLACES = {"abstract", "act", "event", "time", "group", "state", "feeling", "cognition", "attribute", "relation",
              "process", "phenomenon", "possession", "motive", "quantity", "shape"}
PLACE_KINDS = {"place", "building", "location", "region", "area", "body_of_water", "land", "structure"}

# groups of verbs that answer the same question ("when did it begin" is answered by "started")
VERB_GROUPS = [
    {"begin", "start", "commence", "open", "launch", "originate"},
    {"end", "finish", "stop", "conclude", "cease", "close", "terminate"},
    {"make", "create", "build", "construct", "form"},
    {"destroy", "ruin", "wreck", "demolish", "burn", "damage"},
    {"lead", "command", "head", "rule", "govern"},
    {"take", "seize", "capture", "grab", "conquer"},
    {"write", "compose", "author", "pen"},
    {"design", "plan", "draw", "invent"},
    {"die", "perish"},
    {"attack", "assault", "charge", "invade", "strike"},
    {"go", "leave", "depart", "sail", "travel", "journey", "march", "walk", "move", "head"},
    {"say", "tell", "announce", "declare", "state"},
    {"order", "command", "tell", "demand", "instruct"},
]


def same_verb(a: str, b: str) -> bool:
    a, b = a.split("_")[0], b.split("_")[0]
    return a == b or any(a in g and b in g for g in VERB_GROUPS)


def time_phrase(rest, j: int, prep: str) -> Optional[Tuple[str, str, int]]:
    """Read a time expression starting at rest[j] (after the preposition ``prep``): returns (role, text, end) or None.
    role is 'when' (a date, a part of the day, a season) or 'duration' ('for four days')."""
    n = len(rest)
    if j >= n:
        return None
    k = j
    while k + 1 < n and rest[k].text in APPROX:
        k += 1                                              # "in about three hours"
    if rest[k].text in ("the", "a", "an", "this", "that", "next", "last", "every", "each") and k + 1 < n:
        k += 1
    w = rest[k].text
    if prep == "for":
        e = k
        while e < n and (rest[e].text.isdigit() or rest[e].text in WORD_VALUE or rest[e].text in ("hundred", "thousand", "a", "an", "few", "many", "several", "long", "some")):
            e += 1
        if e < n and rest[e].text in TIME_UNITS and e > k:
            return "duration", " ".join(x.text for x in rest[j:e + 1]), e + 1
        if k < n and rest[k].text in TIME_UNITS:
            return "duration", " ".join(x.text for x in rest[j:k + 1]), k + 1
        return None
    if prep not in TIME_PREPS:
        return None
    if prep in ("in", "within") and (w.isdigit() or w in WORD_VALUE or w in ("a", "an", "few", "several")):
        e = k + 1
        while e < n and (rest[e].text.isdigit() or rest[e].text in WORD_VALUE or rest[e].text in ("hundred", "thousand")):
            e += 1
        if e < n and rest[e].text in TIME_UNITS:
            return "duration", " ".join(x.text for x in rest[j:e + 1]), e + 1     # "in about three hours" = it took three hours
    if w.isdigit() or w in MONTHS:
        e, parts = k, []
        while e < n and (rest[e].text.isdigit() or rest[e].text in MONTHS) and len(parts) < 3:
            parts.append(rest[e].text)
            e += 1
        if any(p in MONTHS for p in parts) or (len(parts) == 1 and len(parts[0]) == 4):
            return "when", " ".join(parts), e
        return None
    if w in DAYS or w in SEASONS or w in TIME_NOUNS:
        if k + 2 < n and rest[k + 1].text == "of" and (rest[k + 2].text.isdigit() or rest[k + 2].text in MONTHS):
            sub = time_phrase(rest, k + 2, prep)                                  # "the night of 14 April"
            if sub is not None:
                return "when", " ".join(x.text for x in rest[j:k + 2]) + " " + sub[1], sub[2]
        return "when", " ".join(x.text for x in rest[j:k + 1]), k + 1
    return None


def classify(prep: str, np, lex) -> str:
    """The role of 'prep NP' as an adjunct of the verb."""
    if prep in ("after", "before", "during", "until", "till", "since"):
        return "when"
    if prep == "with":
        return "with"
    if prep in ("about", "over", "because"):
        return "about"
    if prep == "from":
        return "from"
    if prep == "for":
        return "for"
    if prep == "by":
        return "by"
    if prep in ("to", "into", "onto", "toward", "towards"):
        return "to"
    if prep in SPATIAL_PREPS:
        k = np.kind or lex.kind(np.head)
        if np.proper or k not in NOT_PLACES:
            return "where"                                  # "in a bakery", "on Pudding Lane" but not "on its first voyage"
    return ""


def is_place(np, lex) -> bool:
    if np.proper:
        return True
    k = np.kind or lex.kind(np.head)
    return k in PLACE_KINDS or lex.isa_any_pos(np.head, "place") or lex.isa_any_pos(np.head, "location")
