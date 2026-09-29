"""Feed new plain text through the whole v5 stack. Layer 1 stays frozen;
Layer 2 grows units and forms connections based on Layer 1's hidden states;
Layer 3 forms slower, longer-lasting connections from L2's reinforced
patterns. Nothing uses backprop.

Usage:
    python train_layer1.py                  # one-time backprop foundation
    python continual_read.py new_text.txt  # ongoing, forward-pass-only learning
"""

import sys

import torch

from config import Layer1Config, Layer2Config, Layer3Config, StackConfig, TrainConfig
from layer1_language import Pair1Foundation as Layer1Foundation
from stack import MemoryStack
from tokenizer import load_tokenizer


def main():
    tcfg = TrainConfig()
    device = "cpu"  # memory layers are CPU; keep Layer 1 on the same device

    tokenizer = load_tokenizer(tcfg.tokenizer_path)
    l1_cfg = Layer1Config(vocab_size=tokenizer.get_vocab_size())

    layer1 = Layer1Foundation(l1_cfg).to(device)
    layer1.load_state_dict(torch.load(f"{tcfg.out_dir}/layer1_final.pt", map_location=device))
    layer1.eval()

    stack = MemoryStack(layer1,
                          l2_cfg=Layer2Config(dim=l1_cfg.n_embd),
                          l3_cfg=Layer3Config(dim=l1_cfg.n_embd),
                          stack_cfg=StackConfig())
    stack.load(tcfg.out_dir)
    before = stack.summary()
    print(f"[before] {before}")

    text_path = sys.argv[1] if len(sys.argv) > 1 else "data/new_text.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = tokenizer.encode(text).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, idx.size(1), l1_cfg.context_size):
            chunk = idx[:, start:start + l1_cfg.context_size]
            if chunk.size(1) == 0:
                break
            hidden = layer1.forward_returning_hidden(chunk)  # (1, T, C)
            for t in range(hidden.size(1)):
                stack.observe(hidden[0, t])

    stack.save(tcfg.out_dir)
    after = stack.summary()
    print(f"[after]  {after}")

    l2_grown = after["layer2"]["units"] - before["layer2"]["units"]
    l3_grown = after["layer3"]["units"] - before["layer3"]["units"]
    l2_new_conns = after["layer2"]["connections"] - before["layer2"]["connections"]
    l3_new_conns = after["layer3"]["connections"] - before["layer3"]["connections"]
    print(f"\nLayer 2: +{l2_grown} units, {l2_new_conns:+d} connections")
    print(f"Layer 3: +{l3_grown} units, {l3_new_conns:+d} connections")
    print(f"State saved to {tcfg.out_dir}/layer2_state.pt and layer3_state.pt")


if __name__ == "__main__":
    main()
