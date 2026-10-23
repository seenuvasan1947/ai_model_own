"""Number sense: counting and arithmetic that are *learned*, not looked up.

A child learns numbers in this order, and so does this module:

1. **Counting.** Saying "zero one two three ... nine ten eleven ..." again and again. Every time one number follows another, the
   synapse from the last digit of the first to the last digit of the second is strengthened (a Hebbian, asymmetric
   "this-then-that" link). When the last digit wraps (nine -> zero), the next place changed as well, and that is strengthened too:
   a *carry* link. The same digit-to-digit successor links serve every place (units, tens, hundreds ...), which is why a child who
   has counted to a hundred can count to a thousand. Nothing is stored about big numbers; the learned digit links produce them.
   A mirror set of links is learned for counting backwards.
2. **Adding is counting on** (3 + 4: start at 4, take 3 steps), **subtracting is counting back**, **multiplying is repeated adding**,
   **dividing is repeated subtracting**. None of these uses arithmetic: only the learned successor and predecessor links.
3. **Practice makes memory.** A sum that has been worked out by counting leaves a trace; each time it is worked out again (or
   taught: "two plus three is five") the trace is strengthened, and once strong enough the answer is *recalled* instead of
   counted. Traces fade if unused (so the skill falls back to counting, it is not lost).

What is *not* learned and is given as vocabulary: the words for numbers ("twenty seven" <-> 27) and the fact that numerals are
written in places (positional notation). What is learned: the order of the digits, the carry, and every operation built on them.

Exact Python arithmetic is used only by the tests and benchmarks, as the answer key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORD_VALUE = {w: i for i, w in enumerate(UNITS)}
WORD_VALUE.update(TENS)
MAX_STEPS = 20000


class CannotCount(Exception):
    """The learned links do not (yet) cover this step."""


# ------------------------------------------------------------------------------------ number words (vocabulary)
def to_words(n: int) -> str:
    if n < 0:
        return "minus " + to_words(-n)
    if n < 20:
        return UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return list(TENS)[t - 2] + (" " + UNITS[u] if u else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{UNITS[h]} hundred" + (" and " + to_words(r) if r else "")
    if n < 1_000_000:
        k, r = divmod(n, 1000)
        return f"{to_words(k)} thousand" + (" and " + to_words(r) if r and r < 100 else (" " + to_words(r) if r else ""))
    return str(n)


def parse_number(words: List[str]) -> Optional[int]:
    """'27' / 'twenty seven' / 'one hundred and five' / 'two thousand' -> int; None if the words are not one number."""
    ws = [w for w in words if w != "and"]
    if not ws:
        return None
    if len(ws) == 1 and ws[0].isdigit():
        return int(ws[0])
    total, current = 0, 0
    seen = False
    for w in ws:
        if w in WORD_VALUE:
            current += WORD_VALUE[w]
            seen = True
        elif w == "hundred":
            if not seen and current == 0:
                return None
            current = max(current, 1) * 100
        elif w == "thousand":
            total += max(current, 1) * 1000
            current = 0
            seen = True
        else:
            return None
    return total + current if seen else None


def digits(n: int) -> List[int]:
    """Little-endian digits: 407 -> [7, 0, 4]."""
    out = []
    while True:
        n, d = divmod(n, 10)
        out.append(d)
        if n == 0:
            return out


def from_digits(ds: List[int]) -> int:
    v = 0
    for d in reversed(ds):
        v = v * 10 + d
    return v


# ------------------------------------------------------------------------------------ the learned chain
class Chain:
    """Digit-to-digit successor synapses for counting in one direction (+1 forward, -1 backward)."""

    def __init__(self, direction: int) -> None:
        self.dir = direction
        self.w = np.zeros((10, 10))        # w[a, b]: the digit after a is b
        self.carry = np.zeros(10)          # when this digit wrapped, the next place changed too
        self.lr = 1.0

    def _wrapped(self, a: int, b: int) -> bool:
        return b < a if self.dir > 0 else b > a

    def learn(self, a: int, b: int) -> bool:
        """One observation: b comes right after a (forward) / right before a (backward). False if it is not a single step.
        The last place changed; if it wrapped (9 -> 0) the next place changed too (a carry), and so on."""
        da, db = digits(a), digits(b)
        n = max(len(da), len(db))
        da += [0] * (n - len(da))
        db += [0] * (n - len(db))
        changed = [p for p in range(n) if da[p] != db[p]]
        if not changed or changed != list(range(len(changed))):
            return False
        for k, p in enumerate(changed):
            if self._wrapped(da[p], db[p]) == (k == len(changed) - 1):      # every place but the last wraps, the last does not
                return False
        for k, p in enumerate(changed):
            self.w[da[p], db[p]] += self.lr
            if k < len(changed) - 1:
                self.carry[da[p]] += self.lr
        return True

    def next_digit(self, a: int) -> Optional[int]:
        row = self.w[a]
        if row.max() <= 0:
            return None
        return int(row.argmax())

    def step(self, n: int) -> int:
        d = digits(n)
        place = 0
        while True:
            cur = d[place] if place < len(d) else 0
            nxt = self.next_digit(cur)
            if nxt is None:
                raise CannotCount(f"i have not learned what comes {'after' if self.dir > 0 else 'before'} {cur} yet .")
            while place >= len(d):
                d.append(0)
            d[place] = nxt
            if not self._wrapped(cur, nxt):
                break
            if self.carry[cur] <= 0:
                raise CannotCount(f"i have not learned what happens after {cur} {'wraps' if self.dir > 0 else 'wraps back'} yet .")
            place += 1
        return from_digits(d)

    def trained(self) -> bool:
        return bool(self.w.sum() > 0)


@dataclass
class Trace:
    value: int
    strength: float = 0.0
    last: float = 0.0
    rem: int = 0                 # remainder, for division


@dataclass
class Result:
    value: int
    method: str                  # counted | recalled
    steps: int = 0
    remainder: int = 0


class NumberSense:
    def __init__(self) -> None:
        self.fwd = Chain(+1)
        self.bwd = Chain(-1)
        self.traces: Dict[Tuple[str, int, int], Trace] = {}
        self.recall_at = 2.0                # strength at which a worked-out answer is recalled instead of counted again
        self.half_life = 400.0
        self.clock = 0.0
        self.counted_to = 0
        self.max_taught = -1

    # ------------------------------------------------------------------ learning to count
    def hear(self, seq: List[int]) -> int:
        """A counting run ("three four five six"): every neighbouring pair strengthens the links. Returns pairs used."""
        used = 0
        for a, b in zip(seq, seq[1:]):
            if b == a + 1:
                ok_f = self.fwd.learn(a, b)
                self.bwd.learn(b, a)
                used += 1
                self.max_taught = max(self.max_taught, b)
            elif a == b + 1:                          # counting down: "five four three"
                self.fwd.learn(b, a)
                self.bwd.learn(a, b)
                used += 1
                self.max_taught = max(self.max_taught, a)
        return used

    def teach_counting(self, upto: int = 100, repeats: int = 1) -> None:
        for _ in range(repeats):
            self.hear(list(range(0, upto + 1)))

    def can_count(self) -> bool:
        return self.fwd.trained()

    # ------------------------------------------------------------------ primitives built on the links
    def succ(self, n: int) -> int:
        return self.fwd.step(n)

    def pred(self, n: int) -> int:
        if n == 0:
            raise CannotCount("there is nothing before zero .")
        return self.bwd.step(n)

    def rank(self) -> Dict[int, int]:
        """Order of the digits as the learned links give it: follow 'next' from zero."""
        order, d = {}, 0
        while d not in order and d is not None:
            order[d] = len(order)
            d = self.fwd.next_digit(d)
        return order

    def less(self, a: int, b: int) -> bool:
        """a < b, by length then by digit order (the learned order, from the left)."""
        if a == b:
            return False
        da, db = digits(a), digits(b)
        if len(da) != len(db):
            return len(da) < len(db)
        r = self.rank()
        for x, y in zip(reversed(da), reversed(db)):
            if x != y:
                if x not in r or y not in r:
                    raise CannotCount("i have not learned to count yet .")
                return r[x] < r[y]
        return False

    # ------------------------------------------------------------------ operations = counting
    def _count_on(self, start: int, steps: int, forward: bool = True) -> int:
        if steps > MAX_STEPS:
            raise CannotCount(f"that would take {steps} steps of counting and i have not learned a shortcut yet .")
        x = start
        for _ in range(steps):
            x = self.succ(x) if forward else self.pred(x)
        return x

    def _work(self, op: str, a: int, b: int) -> Tuple[int, int, int]:
        """Work an answer out by counting. Returns (value, steps, remainder)."""
        if op == "+":
            lo, hi = (a, b) if not self.less(b, a) else (b, a)             # count on from the larger number
            return self._count_on(hi, self._as_steps(lo), True), self._as_steps(lo), 0
        if op == "-":
            if self.less(a, b):
                raise CannotCount("i have not learned numbers below zero .")
            return self._count_on(a, self._as_steps(b), False), self._as_steps(b), 0
        if op == "*":
            total, steps = 0, 0
            for _ in range(self._as_steps(b)):
                r = self.compute("+", total, a)
                total, steps = r.value, steps + r.steps
            return total, steps, 0
        if op == "/":
            if b == 0:
                raise CannotCount("i cannot divide by zero .")
            q, rest, steps = 0, a, 0
            while not self.less(rest, b):
                r = self.compute("-", rest, b)
                rest, steps = r.value, steps + r.steps
                q = self.succ(q)
                steps += 1
            return q, steps, rest
        raise CannotCount(f"i have not learned '{op}' .")

    def _as_steps(self, n: int) -> int:
        return n          # the number of steps to take is the quantity itself (taking n steps = counting n things)

    def compute(self, op: str, a: int, b: int) -> Result:
        key = (op, a, b)
        tr = self.traces.get(key)
        if tr is not None and tr.strength >= self.recall_at:
            tr.strength += 0.3
            tr.last = self.clock
            return Result(tr.value, "recalled", 0, tr.rem)
        value, steps, rem = self._work(op, a, b)
        if tr is None or tr.value != value:
            tr = Trace(value, 0.0, 0.0, rem)
            self.traces[key] = tr
        tr.strength += 1.0
        tr.last = self.clock
        return Result(value, "counted", steps, rem)

    def check_taught(self, op: str, a: int, b: int, claimed: int) -> Tuple[bool, Result]:
        """The user says 'a op b is claimed'. Work it out; if it agrees, practise it (strengthen the trace); if not, the
        counting wins and the claim is reported as not matching."""
        r = self.compute(op, a, b)
        return r.value == claimed, r

    # ------------------------------------------------------------------ time
    def advance(self, units: float) -> int:
        """Practice fades: traces lose strength; when too weak they are dropped (the answer is counted again next time)."""
        self.clock += units
        gone = 0
        for k in list(self.traces):
            t = self.traces[k]
            t.strength *= 0.5 ** (units / self.half_life)
            if t.strength < 0.2:
                del self.traces[k]
                gone += 1
        return gone

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return dict(fwd=dict(w=self.fwd.w.tolist(), carry=self.fwd.carry.tolist()),
                    bwd=dict(w=self.bwd.w.tolist(), carry=self.bwd.carry.tolist()),
                    traces=[[op, a, b, t.value, t.strength, t.last, t.rem] for (op, a, b), t in self.traces.items()],
                    clock=self.clock, max_taught=self.max_taught)

    def from_dict(self, d: dict) -> None:
        for ch, key in ((self.fwd, "fwd"), (self.bwd, "bwd")):
            ch.w, ch.carry = np.array(d[key]["w"]), np.array(d[key]["carry"])
        self.traces = {(op, a, b): Trace(v, s, l, r) for op, a, b, v, s, l, r in d.get("traces", [])}
        self.clock, self.max_taught = d.get("clock", 0.0), d.get("max_taught", -1)
