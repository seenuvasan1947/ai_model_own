"""Train Pair 1 with backprop on: (a) the raw book (knowledge), and
(b) the small generic Q&A-behavior set (the answering-pattern skill).
See piece_collect.md, "Knowledge vs. the skill of answering."

v4 fix #3: v1-v3 had NO held-out evaluation -- estimate_loss() sampled
batches from the exact same data used for training, so a dropping loss
number could not distinguish real generalization from memorization. This
version splits both the book and QA token streams into train/val once at
load time (val_split, see config.py) and reports both losses at every
checkpoint, so overfitting (train loss keeps dropping, val loss plateaus
or rises) becomes visible instead of invisible.

Usage:
    python tokenizer.py
    python train_pair1.py
"""

import os

import numpy as np
import torch

torch.set_num_threads(4)  # avoid thread-oversubscription overhead on tiny tensors

from config import Layer1Config, TrainConfig
from layer1_language import Pair1Foundation
from tokenizer import load_tokenizer

# How often a batch is drawn from the Q&A set vs. the book. The Q&A set is
# tiny (a few hundred words) compared to the book, so without oversampling
# it, the answering-pattern would barely be seen during training.
QA_SAMPLE_RATIO = 0.35


def split_train_val(data: np.ndarray, val_split: float):
    """v4 fix #3: a simple held-out tail split. Not shuffled -- for text,
    a held-out contiguous tail (e.g. the end of the book) avoids leaking
    near-duplicate overlapping windows into both train and val, which a
    random per-token split would not fully prevent."""
    n_val = max(1, int(len(data) * val_split))
    return data[:-n_val], data[-n_val:]


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
def estimate_loss(model, book_data, qa_data, cfg: TrainConfig, gcfg: Layer1Config, device):
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
    gcfg = Layer1Config(vocab_size=vocab_size)

    with open(tcfg.book_path, "r", encoding="utf-8") as f:
        book_ids = tokenizer.encode(f.read()).ids
    with open(tcfg.qa_path, "r", encoding="utf-8") as f:
        qa_ids = tokenizer.encode(f.read()).ids

    book_all = np.array(book_ids, dtype=np.uint16)
    qa_all = np.array(qa_ids, dtype=np.uint16)

    book_train, book_val = split_train_val(book_all, tcfg.val_split)
    qa_train, qa_val = split_train_val(qa_all, tcfg.val_split)

    print(
        f"book tokens: {len(book_all):,} (train {len(book_train):,} / val {len(book_val):,})  "
        f"qa tokens: {len(qa_all):,} (train {len(qa_train):,} / val {len(qa_val):,})  "
        f"vocab: {vocab_size:,}"
    )

    model = Pair1Foundation(gcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Pair1Foundation parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)

    os.makedirs(tcfg.out_dir, exist_ok=True)

    for it in range(tcfg.max_iters):
        if it % tcfg.eval_interval == 0:
            train_loss = estimate_loss(model, book_train, qa_train, tcfg, gcfg, device)
            val_loss = estimate_loss(model, book_val, qa_val, tcfg, gcfg, device)
            gap = val_loss - train_loss
            flag = "  <- val loss notably higher than train: overfitting signal" if gap > 0.3 else ""
            print(f"step {it}: train loss {train_loss:.4f}  val loss {val_loss:.4f}{flag}")
            torch.save(model.state_dict(), os.path.join(tcfg.out_dir, "layer1_checkpoint.pt"))

        x, y = get_batch(book_train, qa_train, tcfg.batch_size, gcfg.context_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

    print("Phase 1 training complete.")
    torch.save(model.state_dict(), os.path.join(tcfg.out_dir, "layer1_final.pt"))


if __name__ == "__main__":
    main()
