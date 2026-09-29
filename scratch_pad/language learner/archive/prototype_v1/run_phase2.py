"""Loads a Phase-1-trained checkpoint and runs Phase 2 continual learning on new text.

Usage:
    python train.py                 # Phase 1: backprop foundation training
    python run_phase2.py new.txt    # Phase 2: forward-only local updates on new text
"""

import sys

import numpy as np
import torch

from config import GPTConfig, TrainConfig
from model import BrainInspiredGPT
from tokenizer import load_tokenizer
from continual import ContinualLearner


def main():
    tcfg = TrainConfig()
    device = tcfg.device if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(tcfg.tokenizer_path)
    gcfg = GPTConfig(vocab_size=tokenizer.get_vocab_size())

    model = BrainInspiredGPT(gcfg).to(device)
    model.load_state_dict(torch.load(f"{tcfg.out_dir}/checkpoint_final.pt", map_location=device))

    learner = ContinualLearner(model, lr=1e-3, decay=1e-4)

    new_text_path = sys.argv[1] if len(sys.argv) > 1 else "data/new_text.txt"
    with open(new_text_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text).ids
    data = np.array(ids, dtype=np.uint16)

    for i in range(0, len(data) - gcfg.context_size, gcfg.context_size):
        chunk = data[i:i + gcfg.context_size]
        idx = torch.from_numpy(chunk.astype(np.int64)).unsqueeze(0).to(device)
        learner.step(idx)

        if (i // gcfg.context_size) % 20 == 0:
            learner.decay_unused(min_usage=1.0)
            print(f"processed {i} tokens, ran decay pass")

    torch.save(model.state_dict(), f"{tcfg.out_dir}/checkpoint_after_phase2.pt")
    print("Phase 2 pass complete. Weights updated locally, no backward pass used.")


if __name__ == "__main__":
    main()
