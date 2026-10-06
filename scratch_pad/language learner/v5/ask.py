"""Ask the v5 stack a question. Layer 1 generates; Layer 2/3 activate on
each hidden state and add their combined recall vector to Layer 1's
residual stream before the LM head projects to vocab.

Use --no-memory to disable memory feedback for ablation comparison
(this measures whether L2/L3 actually helps generation).

Usage:
    python ask.py "Who is Zeta?"
    python ask.py --no-memory "Who is Zeta?"
"""

import os
import sys
from datetime import datetime

import torch

from config import Layer1Config, Layer2Config, Layer3Config, StackConfig, TrainConfig
from layer1_language import Pair1Foundation as Layer1Foundation
from stack import MemoryStack
from tokenizer import load_tokenizer


def log_session(out_dir: str, lines: list[str]):
    """Append a timestamped record to out/session_log.md -- see
    continual_read.py's log_session for why this is a tracked text file."""
    with open(f"{out_dir}/session_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat(timespec='seconds')} -- ask.py\n")
        for line in lines:
            f.write(line + "\n")


def ask(layer1, stack, tokenizer, device, question: str, use_memory: bool = True,
        max_new_tokens: int = 60):
    prompt = f"Q: {question}\nA:"
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    if use_memory:
        memory_fn = stack.recall_vector
    else:
        # Bypass mode: memory contributes zero to the residual stream.
        # This is the ablation baseline.
        zero = torch.zeros(layer1.config.n_embd)
        def memory_fn(hidden):
            return zero

    out = layer1.generate_with_memory_callback(
        idx, max_new_tokens=max_new_tokens,
        memory_fn=memory_fn,
        temperature=0.8, top_k=40,
    )
    text = tokenizer.decode(out[0].tolist())
    answer = text[len(prompt):]
    if "Q:" in answer:
        answer = answer.split("Q:")[0]
    return answer.strip()


def main():
    args = sys.argv[1:]
    use_memory = True
    if args and args[0] == "--no-memory":
        use_memory = False
        args = args[1:]

    tcfg = TrainConfig()
    device = "cpu"

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
    print(f"[stack] {stack.summary()}")
    print(f"[memory feedback: {'ON' if use_memory else 'OFF (ablation)'}]")

    def run(question):
        print(f"\nQ: {question}")
        answer = ask(layer1, stack, tokenizer, device, question, use_memory=use_memory)
        print("A:", answer)
        log_session(tcfg.out_dir, [
            f"memory: {'ON' if use_memory else 'OFF (ablation)'}",
            f"Q: {question}",
            f"A: {answer}",
        ])

    if args:
        run(" ".join(args))
        return

    print("\nInteractive mode. Type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q.lower() == "quit" or not q:
            break
        run(q)


if __name__ == "__main__":
    main()
