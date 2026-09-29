"""Feed new plain text to the already-trained Pair 1 foundation and let
Pair 2 (short-term) and Pair 3 (long-term) learn from it -- no backprop,
no changes to Pair 1's weights at all (see architecture_design.md,
Section 3: structural separation, forward-pass only).

Usage:
    python train_pair1.py                 # one-time backprop foundation
    python continual_read.py new_text.txt # ongoing, forward-pass-only learning
"""

import os
import sys

import torch

from config import Pair1Config, Pair2Config, Pair3Config, TrainConfig
from pair1_foundation import Pair1Foundation
from pair2_shortterm import ShortTermMemory
from pair3_longterm import LongTermStore
from tokenizer import load_tokenizer

PAIR2_STATE_PATH = "out/pair2_state.pt"
PAIR3_STATE_PATH = "out/pair3_state.pt"


def main():
    tcfg = TrainConfig()
    device = "cpu"  # Pair 2/3 are lightweight; no need for GPU here

    tokenizer = load_tokenizer(tcfg.tokenizer_path)
    gcfg = Pair1Config(vocab_size=tokenizer.get_vocab_size())

    pair1 = Pair1Foundation(gcfg).to(device)
    pair1.load_state_dict(torch.load(f"{tcfg.out_dir}/pair1_final.pt", map_location=device))
    pair1.eval()  # frozen -- structurally, no optimizer is ever created for it here

    pair2 = ShortTermMemory(Pair2Config(dim=gcfg.n_embd))
    pair3 = LongTermStore(Pair3Config(dim=gcfg.n_embd))
    if os.path.exists(PAIR2_STATE_PATH):
        pair2.load(PAIR2_STATE_PATH)
        print("[loaded existing Pair 2 short-term state]")
    if os.path.exists(PAIR3_STATE_PATH):
        pair3.load(PAIR3_STATE_PATH)
        print(f"[loaded existing Pair 3 long-term state: {len(pair3)} entries]")

    text_path = sys.argv[1] if len(sys.argv) > 1 else "data/new_text.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    promotions = 0
    with torch.no_grad():
        for start in range(0, idx.size(1) - 1, gcfg.context_size):
            chunk = idx[:, start:start + gcfg.context_size]
            targets = idx[:, start + 1:start + 1 + gcfg.context_size]
            if targets.size(1) == 0:
                break
            chunk = chunk[:, :targets.size(1)]  # trim to match targets on the final partial chunk

            logits, loss, hidden = pair1(chunk, targets=targets, return_hidden=True)

            # Pair 2: local Hebbian update, one token position at a time
            for t in range(hidden.size(1)):
                pair2.step(hidden[0, t])

            # Pair 3: promotion check on the chunk's mean representation,
            # using Pair 1's own (frozen, unmodified-by-this) loss as the
            # surprise signal -- Pair 1's weights are never updated here.
            mean_vec = hidden[0].mean(dim=0)
            snippet = tokenizer.decode(chunk[0].tolist())[:80]
            idx_added, outcome = pair3.maybe_promote(mean_vec, snippet, loss.item())
            if outcome != "not_promoted":
                promotions += 1
                print(f"  [{outcome}] \"{snippet}...\"")

    pair2.save(PAIR2_STATE_PATH)
    pair3.save(PAIR3_STATE_PATH)

    print(f"\nDone. Long-term store size: {len(pair3)}  |  promotions this run: {promotions}")
    print(f"Short-term trace norm: {pair2.current().norm().item():.4f}")
    print(f"Saved Pair 2/3 state to {PAIR2_STATE_PATH} / {PAIR3_STATE_PATH}")


if __name__ == "__main__":
    main()
