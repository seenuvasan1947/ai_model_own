"""Phase 1 training loop: plain backprop, single-language corpus.

Usage:
    python tokenizer.py data/corpus.txt data/tokenizer.json 4000
    python train.py
"""

import os

import numpy as np
import torch

from config import GPTConfig, TrainConfig
from model import BrainInspiredGPT
from tokenizer import load_tokenizer


def get_batch(data, batch_size, context_size, device):
    ix = torch.randint(len(data) - context_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + context_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + context_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg: TrainConfig, gcfg: GPTConfig):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, gcfg.context_size, cfg.device)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    tcfg = TrainConfig()
    device = tcfg.device if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(tcfg.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    gcfg = GPTConfig(vocab_size=vocab_size)

    with open(tcfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text).ids
    data = np.array(ids, dtype=np.uint16)

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = BrainInspiredGPT(gcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)

    os.makedirs(tcfg.out_dir, exist_ok=True)

    for it in range(tcfg.max_iters):
        if it % tcfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, tcfg, gcfg)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            torch.save(model.state_dict(), os.path.join(tcfg.out_dir, "checkpoint.pt"))

        x, y = get_batch(train_data, tcfg.batch_size, gcfg.context_size, device)
        _, loss, _ = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

    print("Training complete.")
    torch.save(model.state_dict(), os.path.join(tcfg.out_dir, "checkpoint_final.pt"))


if __name__ == "__main__":
    main()
