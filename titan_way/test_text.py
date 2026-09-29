"""Real-text check: learn question -> answer facts at test time, recall them
from PARAPHRASED questions, then keep learning more without a training phase.
Uses the hash encoder from brainlm (no training)."""
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from brainlm.encoders import HashEncoder
from neural_memory import NeuralMemory

torch.manual_seed(0)
enc = HashEncoder(512)
E = lambda t: torch.tensor(enc.encode(t))[None]

FACTS = [
    ("capital of France", "Paris"), ("capital of Italy", "Rome"),
    ("capital of Japan", "Tokyo"), ("capital of Egypt", "Cairo"),
    ("capital of Canada", "Ottawa"), ("capital of Kenya", "Nairobi"),
    ("creator of Python", "Guido van Rossum"), ("creator of Linux", "Linus Torvalds"),
    ("boiling point of water", "100 degrees Celsius"), ("freezing point of water", "0 degrees Celsius"),
    ("powerhouse of the cell", "mitochondria"), ("largest planet", "Jupiter"),
    ("fastest land animal", "cheetah"), ("chemical symbol of gold", "Au"),
    ("author of Hamlet", "Shakespeare"), ("painter of the Mona Lisa", "Leonardo da Vinci"),
    ("inventor of the telephone", "Bell"), ("tallest mountain", "Everest"),
    ("longest river in Africa", "Nile"), ("smallest prime number", "two"),
]
PARAPHRASE = lambda q: "what is the " + q + "?"

def answer(mem, q, answers):
    out = F.normalize(mem.retrieve(E(q)), dim=-1)
    tgt = F.normalize(torch.cat([mem.target(E(a)) for a in answers]), dim=-1)
    return (out @ tgt.T).argmax().item()

def accuracy(mem, facts, para):
    ans = [a for _, a in facts]
    hit = sum(answer(mem, PARAPHRASE(q) if para else q, ans) == i for i, (q, _) in enumerate(facts))
    return hit / len(facts)

mem = NeuralMemory(512, dim=64, depth=1)
first, second = FACTS[:10], FACTS[10:]
print("before learning     : %.2f  (chance 0.10)" % accuracy(mem, first, False))
for _ in range(5):
    for q, a in first:
        mem.update(E(q), E(a))
print("exact questions     : %.2f" % accuracy(mem, first, False))
print("paraphrased ('what is the ...?'): %.2f" % accuracy(mem, first, True))
for _ in range(5):
    for q, a in second:
        mem.update(E(q), E(a))
print("-- then learned 10 MORE facts, no training phase --")
print("old facts (paraphrased) : %.2f" % accuracy(mem, first, True))
print("new facts (paraphrased) : %.2f" % accuracy(mem, second, True))
print("all 20 facts            : %.2f" % accuracy(mem, FACTS, True))
