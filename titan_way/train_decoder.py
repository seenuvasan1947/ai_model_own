"""Train the decoder from scratch and test on UNSEEN names.

  python train_decoder.py --backbone gru --steps 1500
"""
import argparse
import random
import time

import torch
import torch.nn.functional as F

from decoder import ByteSeq, batchify
from synth import REAL, dataset, sample

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="gru", choices=["gru", "transformer"])
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--threads", type=int, default=0)
p.add_argument("--out", default=None)
a = p.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

torch.manual_seed(0)
model = ByteSeq(a.backbone)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
rng = random.Random(1)
print(f"{a.backbone}: {sum(x.numel() for x in model.parameters())/1e6:.2f}M params", flush=True)
t0 = time.time()
model.train()
for step in range(1, a.steps + 1):
    for g in opt.param_groups:
        g["lr"] = a.lr * min(1, step / 100) * (0.1 + 0.9 * (1 - step / a.steps))
    x, y, m = batchify([sample(rng) for _ in range(a.batch)])
    loss = (F.cross_entropy(model(x).transpose(1, 2), y, reduction="none") * m).sum() / m.sum()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 250 == 0:
        print(f"  step {step}: loss {loss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)
train_time = time.time() - t0

test = dataset(100, seed=999)  # names never seen in training
em = sum(model.generate(q, m) == ans for q, m, ans in test) / len(test)
print(f"exact match on 100 held-out samples (unseen random names): {em:.2f}")
print("real examples:")
for q, m, ans in REAL:
    g = model.generate(q, m)
    print(f"  {'OK ' if g == ans else 'BAD'} Q:{q} M:{m}\n      -> {g}")
print(f"train time {train_time:.0f}s")
out = a.out or f"decoder_{a.backbone}.pt"
torch.save({"backbone": a.backbone, "state": model.state_dict(), "em": em,
            "train_time": train_time}, out)
print("saved", out)
