"""Associative recall across segments: can the model use info it saw many
segments ago? Sequence = N (key, value) pairs in the FIRST segment, filler,
then QUERY + a key; predict that key's value (last position only).
Only the memory (or full attention) can see the pairs when the query arrives.

  python train_full.py --mode mac --steps 1500
"""
import argparse
import time

import torch
import torch.nn.functional as F

from titans_full import Titans

K = 16                       # keys 0..15, values 16..31, filler 32..47, QUERY 48
VALUE0, FILL0, QUERY, VOCAB = K, 2 * K, 3 * K, 3 * K + 1


def batch(B, N, T, gen):
    tok = torch.randint(FILL0, FILL0 + K, (B, T), generator=gen)
    keys = torch.stack([torch.randperm(K, generator=gen)[:N] for _ in range(B)])
    vals = torch.randint(0, K, (B, N), generator=gen) + VALUE0
    tok[:, 0:2 * N:2], tok[:, 1:2 * N:2] = keys, vals
    pick = torch.randint(0, N, (B,), generator=gen)
    tok[:, T - 2] = QUERY
    tok[:, T - 1] = keys[torch.arange(B), pick]
    return tok, vals[torch.arange(B), pick], vals


@torch.no_grad()
def accuracy(model, N, T, n=256, update=True, seed=123):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    tok, ans, _ = batch(n, N, T, gen)
    pred = model(tok, update=update)[:, -1].argmax(-1)
    return (pred == ans).float().mean().item()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="mac", choices=["mac", "local", "full"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--T", type=int, default=96)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--theta0", type=float, default=0.1, help="initial memory step size")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    model = Titans(VOCAB, a.mode, theta0=a.theta0)
    print(f"{a.mode}: {sum(x.numel() for x in model.parameters())/1e3:.0f}K params | train len {a.T}, {a.N} pairs, chance {1/a.N:.2f} (if it knew the pairs) / {1/K:.2f} (guessing a value)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(1)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1, step / 50) * (0.1 + 0.9 * (1 - step / a.steps))
        model.train()
        tok, ans, _ = batch(a.batch, a.N, a.T, gen)
        loss = F.cross_entropy(model(tok)[:, -1], ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"  step {step}: loss {loss.item():.3f} acc@{a.T} {accuracy(model, a.N, a.T):.2f}  ({time.time()-t0:.0f}s)", flush=True)
    res = {L: accuracy(model, a.N, L) for L in (a.T, 2 * a.T, 4 * a.T)}
    print("accuracy by sequence length:", {k: round(v, 2) for k, v in res.items()})
    if a.mode == "mac":
        res0 = {L: accuracy(model, a.N, L, update=False) for L in (a.T, 2 * a.T, 4 * a.T)}
        print("same model, memory update OFF at test time:", {k: round(v, 2) for k, v in res0.items()})
    torch.save({"mode": a.mode, "state": model.state_dict(), "res": res, "train_time": time.time() - t0},
               a.out or f"full_{a.mode}.pt")
