"""Tiny word-level tokenizer: lowercase words, numbers and a few punctuation marks."""

from __future__ import annotations

import re
from typing import Iterable, Iterator, List

BOS = "<s>"
END_TOKENS = frozenset({".", "?", "!"})
# Question/answer protocol markers are "transparent": they set the reading mode but never enter
# word contexts, so a fact told as a statement and the same fact given as an answer share cells.
PROTOCOL = frozenset({"question", "answer", ":"})
# Tokens that are not "content": they never enter the topic trace (punctuation and the two
# discourse markers of the question/answer protocol).
NON_CONTENT = frozenset({BOS, ".", "?", "!", ",", ":", ";", "question", "answer"})

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,?!:;]")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def split_sentences(tokens: Iterable[str]) -> Iterator[List[str]]:
    """Split a token stream into sentences ending at . ? !"""
    cur: List[str] = []
    for t in tokens:
        cur.append(t)
        if t in END_TOKENS:
            yield cur
            cur = []
    if cur:
        yield cur


def split_documents(text: str) -> List[str]:
    """Documents are separated by blank lines. The topic trace resets per document."""
    return [d for d in re.split(r"\n\s*\n", text.strip()) if d.strip()]
