"""Evaluation against baselines (plan_future item 6). Several seeds.

Facts: "capital of <Name>" -> <Name>. Keys differ only by one word, so this is
a hard interference test. Test with a PARAPHRASED question.

Systems:
  chance        : 1/N
  RAG (sentence store): nearest stored question by cosine -> its answer (brainlm-style)
  Titan memory  : NeuralMemory learned at test time with rehearsal (agent.py)
Also: sequential learning (4 batches, re-test everything after each), and size.
"""
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import TitanAgent
from synth import name

SEEDS = (0, 1, 2)


def facts(n, rng):
    seen, out = set(), []
    while len(out) < n:
        x, y = name(rng), name(rng)
        if x not in seen and y not in seen:
            seen |= {x, y}; out.append((f"capital of {x}", f"what is the capital of {x}?", y))
    return out


def titan_acc(agent, fs):
    ok = 0
    for _, para, ans in fs:
        ranked, _ = agent.recall(para)
        ok += bool(ranked) and ranked[0][0] == ans
    return ok / len(fs)


def rag_acc(agent, fs):
    keys = np.stack([agent.E(q)[0].numpy() for q, _, _ in fs])
    ok = 0
    for i, (_, para, ans) in enumerate(fs):
        ok += fs[int((keys @ agent.E(para)[0].numpy()).argmax())][2] == ans
    return ok / len(fs)


def main():
    print("== 1. accuracy vs number of facts (paraphrased test question), mean of 3 seeds")
    print(f"   {'N':>4} {'chance':>7} {'RAG':>6} {'Titan':>6} {'teach time':>11}")
    for n in (10, 25, 50, 100):
        t, r, times = [], [], []
        for s in SEEDS:
            rng = random.Random(s); fs = facts(n, rng); tmp = tempfile.mkdtemp()
            ag = TitanAgent(tmp, seed=s); t0 = time.time()
            for q, _, a in fs:
                ag.teach(q, a)
            times.append(time.time() - t0)
            t.append(titan_acc(ag, fs)); r.append(rag_acc(ag, fs)); shutil.rmtree(tmp)
        print(f"   {n:>4} {1/n:>7.2f} {np.mean(r):>6.2f} {np.mean(t):>6.2f} {np.mean(times):>9.1f}s   (Titan std {np.std(t):.2f})")

    print("\n== 2. sequential learning: 4 batches of 15 facts; accuracy on ALL facts seen so far")
    print("   (rehearsal ON = replays old facts; OFF = each fact learned alone)")
    for label, replay in (("rehearsal ON ", True), ("rehearsal OFF", False)):
        import agent as A
        A.REPLAY = 16 if replay else 0
        rows = []
        for s in SEEDS:
            rng = random.Random(s); fs = facts(60, rng); tmp = tempfile.mkdtemp()
            ag = TitanAgent(tmp, seed=s); acc = []
            for b in range(4):
                for q, _, a in fs[b * 15:(b + 1) * 15]:
                    ag.teach(q, a)
                acc.append(titan_acc(ag, fs[:(b + 1) * 15]))
            rows.append(acc); shutil.rmtree(tmp)
        print(f"   {label}: after each batch -> " + "  ".join(f"{x:.2f}" for x in np.mean(rows, 0)))
    A.REPLAY = 16

    print("\n== 3. size")
    ag = TitanAgent(tempfile.mkdtemp())
    print(f"   Titan memory params: {sum(w.numel() for w in ag.mem.M)} (fixed, does not grow with facts)")
    print(f"   RAG store: grows by one {ag.enc.dim}-float vector per fact")


if __name__ == "__main__":
    main()
