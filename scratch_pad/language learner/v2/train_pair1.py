"""Train Pair 1 with backprop on: (a) the raw book (knowledge), and
(b) the small generic Q&A-behavior set (the answering-pattern skill).
See piece_collect.md, "Knowledge vs. the skill of answering."

Usage:
    python tokenizer.py
    python train_pair1.py
"""

import os

import numpy as np
import torch

torch.set_num_threads(4)  # avoid thread-oversubscription overhead on tiny tensors

from config import Pair1Config, TrainConfig
from pair1_foundation import Pair1Foundation
from tokenizer import load_tokenizer

# How often a batch is drawn from the Q&A set vs. the book. The Q&A set is
# tiny (a few hundred words) compared to the book (~26k words), so without
# oversampling it, the answering-pattern would barely be seen during training.
QA_SAMPLE_RATIO = 0.35


def get_batch(book_data, qa_data, batch_size, context_size, device):
    n_qa = max(1, int(batch_size * QA_SAMPLE_RATIO))
    n_book = batch_size - n_qa

    def sample(data, n):
        ix = torch.randint(len(data) - context_size - 1, (n,))
        x = torch.stack([torch.from_numpy(data[i:i + context_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + context_size].astype(np.int64)) for i in ix])
        return x, y

    xb, yb = sample(book_data, n_book)
    xq, yq = sample(qa_data, n_qa) if len(qa_data) > context_size + 1 else (xb[:0], yb[:0])

    x = torch.cat([xb, xq], dim=0).to(device)
    y = torch.cat([yb, yq], dim=0).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, book_data, qa_data, cfg: TrainConfig, gcfg: Pair1Config, device):
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for k in range(cfg.eval_iters):
        x, y = get_batch(book_data, qa_data, cfg.batch_size, gcfg.context_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    tcfg = TrainConfig()
    device = tcfg.device if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(tcfg.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    gcfg = Pair1Config(vocab_size=vocab_size)

    with open(tcfg.book_path, "r", encoding="utf-8") as f:
        book_ids = tokenizer.encode(f.read()).ids
    with open(tcfg.qa_path, "r", encoding="utf-8") as f:
        qa_ids = tokenizer.encode(f.read()).ids

    book_data = np.array(book_ids, dtype=np.uint16)
    qa_data = np.array(qa_ids, dtype=np.uint16)

    print(f"book tokens: {len(book_data):,}  qa tokens: {len(qa_data):,}  vocab: {vocab_size:,}")

    model = Pair1Foundation(gcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Pair1Foundation parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)

    os.makedirs(tcfg.out_dir, exist_ok=True)

    for it in range(tcfg.max_iters):
        if it % tcfg.eval_interval == 0:
            loss_est = estimate_loss(model, book_data, qa_data, tcfg, gcfg, device)
            print(f"step {it}: loss {loss_est:.4f}")
            torch.save(model.state_dict(), os.path.join(tcfg.out_dir, "pair1_checkpoint.pt"))

        x, y = get_batch(book_data, qa_data, tcfg.batch_size, gcfg.context_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

    print("Phase 1 training complete.")
    torch.save(model.state_dict(), os.path.join(tcfg.out_dir, "pair1_final.pt"))


if __name__ == "__main__":
    main()
