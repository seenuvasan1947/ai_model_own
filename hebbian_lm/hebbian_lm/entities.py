"""Entities: one instance per introduced thing, so "tom has a red ball . anna has a blue ball ." is two balls.

Without this, the semantic memory keys everything by lemma and "the ball" is always the same ball. With it, an
indefinite mention ("a ball", "tom's ball") of a particular thing creates an *instance* with an id such as
``ball#2``; facts are then stored on the id, and a definite mention ("the ball") is resolved against the instances
that exist (the only one, the most recently mentioned one when reading, or a clarifying question when asked).

Kind-level knowledge (``ball`` itself: "a ball is a toy", "the zebra is brown" with no zebra introduced) is
unchanged, and an instance inherits from its kind with no discount (see ``SemanticMemory``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def lemma_of(x: str) -> str:
    """'ball#2' -> 'ball'; a plain lemma is returned unchanged."""
    i = x.find("#")
    return x if i < 0 else x[:i]


def is_instance(x: str) -> bool:
    return "#" in x


@dataclass
class Entity:
    id: str
    lemma: str
    owner: str = ""                  # a name ("tom") or another entity id; "" if none
    number: str = "sg"
    adjs: Tuple[str, ...] = ()       # qualifiers it was introduced with ("red")
    born: int = 0


@dataclass
class EntityStore:
    items: Dict[str, Entity] = field(default_factory=dict)
    clock: int = 0

    def new(self, lemma: str, owner: str = "", number: str = "sg", adjs=()) -> Entity:
        k = 1 + sum(1 for e in self.items.values() if e.lemma == lemma)
        while f"{lemma}#{k}" in self.items:
            k += 1
        self.clock += 1
        e = Entity(f"{lemma}#{k}", lemma, owner, number, tuple(adjs), self.clock)
        self.items[e.id] = e
        return e

    def of(self, lemma: str) -> List[Entity]:
        """Instances of a noun, newest first."""
        return sorted((e for e in self.items.values() if e.lemma == lemma), key=lambda e: -e.born)

    def find(self, lemma: str, owner: str, adjs: Optional[Tuple[str, ...]] = None) -> Optional[Entity]:
        for e in self.of(lemma):
            if e.owner == owner and (adjs is None or e.adjs == tuple(adjs)):
                return e
        return None

    def __len__(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict:
        return dict(clock=self.clock, items=[dict(id=e.id, lemma=e.lemma, owner=e.owner, number=e.number,
                                                  adjs=list(e.adjs), born=e.born) for e in self.items.values()])

    def from_dict(self, d: dict) -> None:
        self.clock = d.get("clock", 0)
        self.items = {x["id"]: Entity(x["id"], x["lemma"], x["owner"], x["number"], tuple(x["adjs"]), x["born"])
                      for x in d.get("items", [])}
