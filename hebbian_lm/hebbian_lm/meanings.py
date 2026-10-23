"""Core word meanings: what common things are for, made of, made up of, and where they are found; and questions about them.

``resources/core_meanings.tsv`` is a hand-written table of about 110 everyday nouns (spoon, bridge, ship, farm ...). It is a
starting vocabulary of meaning that goes beyond "is a kind of": if the user tells the memory something ("a spoon is used for
eating soup"), what was told wins; the table is only the fallback. It is small by design and says "i do not know" for the rest.

Question forms understood:  what is a X used for | what is X made of | what does a X have | what are the parts of a X |
where is a X found | where do you find X.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple


class CoreMeanings:
    def __init__(self, path: str = "") -> None:
        path = path or os.path.join(os.path.dirname(__file__), "resources", "core_meanings.tsv")
        self.table: Dict[str, Dict[str, object]] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip() or line.startswith("#"):
                        continue
                    f = (line.rstrip("\n").split("\t") + [""] * 5)[:5]
                    self.table[f[0]] = dict(used_for=f[1], made_of=f[2], parts=[x for x in f[3].split(",") if x], found_in=f[4])

    def get(self, noun: str, rel: str):
        e = self.table.get(noun)
        return e.get(rel) if e else None

    def __contains__(self, noun: str) -> bool:
        return noun in self.table


_DET = {"a", "an", "the", "some", "every"}


def parse_knowledge_question(w: List[str], lemma) -> Optional[Tuple[str, str]]:
    """Words of a question -> (relation, noun) with relation in used_for | made_of | parts | found_in; None if it is not one."""
    w = [x for x in w if x not in _DET]
    n = len(w)
    if n >= 4 and w[0] == "what" and w[1] in ("is", "are") and w[-1] == "for" and w[-2] == "used":
        return "used_for", lemma(w[2])
    if n >= 4 and w[0] == "what" and w[1] in ("do", "can") and w[2] == "you" and w[3] == "use" and w[-1] == "for":
        return "used_for", lemma(w[4]) if n > 5 else ""
    if n == 4 and w[0] == "what" and w[1] in ("is", "are") and w[3] == "for":
        return "used_for", lemma(w[2])
    if n >= 4 and w[0] == "what" and w[1] in ("is", "are") and w[-1] in ("of", "from") and w[-2] == "made":
        return "made_of", lemma(w[2])
    if n >= 4 and w[0] == "what" and w[1] in ("is", "are") and "made" in w and w[-3:-1] == ["made", "out"]:
        return "made_of", lemma(w[2])
    if n == 4 and w[0] == "what" and w[1] in ("does", "do") and w[3] in ("have", "has"):
        return "parts", lemma(w[2])
    if n >= 5 and w[0] == "what" and w[1] in ("are", "is") and w[2] == "parts" and w[3] == "of":
        return "parts", lemma(w[4])
    if n == 5 and w[0] == "what" and w[1] == "parts" and w[2] in ("does", "do") and w[4] in ("have", "has"):
        return "parts", lemma(w[3])
    if n >= 4 and w[0] == "where" and w[1] in ("is", "are") and w[-1] == "found":
        return "found_in", lemma(w[2])
    if n >= 4 and w[0] == "where" and "find" in w:
        return "found_in", lemma(w[-1])
    return None
