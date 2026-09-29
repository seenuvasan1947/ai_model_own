"""Small-scale check: can the neural memory learn at test time?

Every test presents input vectors ONCE (or a few times) to update(), never
touching an optimizer, then reads them back with retrieve().
recall accuracy = the retrieved vector is closest to the RIGHT stored value
among all stored values (chance = 1/N).
"""
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from neural_memory import NeuralMemory

torch.manual_seed(0)
IN = 64


def recall_acc(mem, X):
    out = F.normalize(mem.retrieve(X), dim=-1)
    tgt = F.normalize(mem.target(X), dim=-1)
    return (out @ tgt.T).argmax(1).eq(torch.arange(len(X))).float().mean().item()


def learn(mem, X, passes=1, **kw):
    for _ in range(passes):
        for i in range(len(X)):
            mem.update(X[i:i + 1], **kw)


def T(title):
    print(f"\n== {title}")


def main():


    T("1. untrained memory vs after seeing each item ONCE (N=20)")
    X = torch.randn(20, IN)
    m = NeuralMemory(IN)
    print(f"   before learning: {recall_acc(m, X):.2f}   (chance 0.05)")
    learn(m, X)
    print(f"   after 1 pass   : {recall_acc(m, X):.2f}")
    learn(m, X, passes=4)
    print(f"   after 5 passes : {recall_acc(m, X):.2f}")

    T("2. capacity: recall vs number of stored items (5 passes)")
    for n in (10, 20, 50, 100, 200):
        X = torch.randn(n, IN); m = NeuralMemory(IN)
        learn(m, X, passes=5)
        print(f"   N={n:<4} recall {recall_acc(m, X):.2f}")

    T("3. ablation: what each Titans ingredient adds (N=50, 5 passes)")
    X = torch.randn(50, IN)
    for name, kw_mem, kw_up in [
        ("full (deep=2, momentum, decay)", {}, {}),
        ("linear memory (depth=1)", {"depth": 1}, {}),
        ("deeper (depth=3)", {"depth": 3}, {}),
        ("no momentum", {}, {"momentum": False}),
        ("no decay", {}, {"decay": False}),
    ]:
        m = NeuralMemory(IN, **kw_mem); learn(m, X, passes=5, **kw_up)
        print(f"   {name:<32} recall {recall_acc(m, X):.2f}")

    T("4. surprise: new vs already-seen items")
    X = torch.randn(10, IN); m = NeuralMemory(IN); learn(m, X, passes=5)
    new = torch.randn(10, IN)
    print(f"   seen  items surprise {m.surprise(X).mean():.3f}")
    print(f"   new   items surprise {m.surprise(new).mean():.3f}   (should be higher)")

    T("5. forgetting: learn A, then stream B (N=20 each)")
    A, B = torch.randn(20, IN), torch.randn(20, IN)
    for label, alpha in [("no decay (alpha=0)", 0.0), ("decay alpha=0.01", 0.01), ("decay alpha=0.05", 0.05)]:
        m = NeuralMemory(IN, alpha=alpha)
        learn(m, A, passes=5)
        a0 = recall_acc(m, A)
        learn(m, B, passes=5)
        print(f"   {label:<20} A after A {a0:.2f} | A after B {recall_acc(m, A):.2f} | B {recall_acc(m, B):.2f}")

    T("6. persistence: save weights, load into a fresh memory")
    X = torch.randn(20, IN); m = NeuralMemory(IN); learn(m, X, passes=5)
    p = os.path.join(tempfile.mkdtemp(), "memory.pt"); m.save(p)
    m2 = NeuralMemory(IN, seed=99); m2.load(p)
    print(f"   original {recall_acc(m, X):.2f} | reloaded {recall_acc(m2, X):.2f} | file {os.path.getsize(p)/1024:.0f} KB")


if __name__ == "__main__":
    main()
