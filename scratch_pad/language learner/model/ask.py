"""Ask the full system a question -- Pair 1 (fixed) generates, informed by
Pair 2 (short-term trace) and Pair 3 (long-term store) if they hold any
state from continual_read.py runs.

This is the generative test: Pair 1 was trained on (a) the raw book text
and (b) a small, unrelated, generic Q&A-behavior set -- never shown a
question-answer pair about the book itself. If Pair 2/3 have learned
anything since training (via continual_read.py), that's injected here as
extra context text before generation -- not hardcoded, not retrieved
verbatim as the final answer, only as additional context Pair 1's own
generative process can use or ignore. See piece_collect.md, "Knowledge
vs. the skill of answering," and architecture_design.md Section 3.

Usage:
    python ask.py "Who did Alice meet at the tea party?"
"""

import os
import sys

import torch
import torch.nn.functional as F

from config import Pair1Config, Pair2Config, Pair3Config, TrainConfig
from pair1_foundation import Pair1Foundation
from pair2_shortterm import ShortTermMemory
from pair3_longterm import LongTermStore
from tokenizer import load_tokenizer

PAIR2_STATE_PATH = "out/pair2_state.pt"
PAIR3_STATE_PATH = "out/pair3_state.pt"


@torch.no_grad()
def encode_query(model, tokenizer, device, text: str):
    """Mean-pooled hidden vector for `text`, in the same space Pair 2/3 read
    from -- used both to query Pair 3 and as the fallback if Pair 2 needs one."""
    ids = tokenizer.encode(text).ids[-model.config.context_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, hidden = model(idx, return_hidden=True)
    return hidden[0].mean(dim=0)


@torch.no_grad()
def decode_trace_to_words(model, tokenizer, trace: torch.Tensor, k: int = 5):
    """Pair 2's trace is a raw vector, not text. Decode it back through
    Pair 1's own output head (tied embedding) to see which vocabulary
    tokens it's currently closest to -- a lightweight, non-hardcoded way
    to turn "what's recently been active" into a short list of words."""
    if trace.norm().item() < 1e-6:
        return []
    logits = model.head(trace)
    top = torch.topk(logits, k).indices.tolist()
    words = [tokenizer.decode([t]).strip() for t in top]
    return [w for w in words if w]


def ask(model, tokenizer, device, question: str, pair2=None, pair3=None, max_new_tokens=60):
    context_lines = []

    if pair3 is not None and len(pair3) > 0:
        query_vec = encode_query(model, tokenizer, device, question)
        matches = pair3.retrieve(query_vec, k=2)
        relevant = [label for label, sim, _ in matches if sim > 0.3]
        if relevant:
            context_lines.append("Context: " + " ".join(relevant))

    if pair2 is not None:
        recent_words = decode_trace_to_words(model, tokenizer, pair2.current())
        if recent_words:
            context_lines.append("Recent: " + " ".join(recent_words))

    prefix = "\n".join(context_lines) + "\n" if context_lines else ""
    prompt = f"{prefix}Q: {question}\nA:"

    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    text = tokenizer.decode(out[0].tolist())

    answer_part = text[len(prompt):]
    if "Q:" in answer_part:
        answer_part = answer_part.split("Q:")[0]
    return answer_part.strip(), context_lines


def main():
    tcfg = TrainConfig()
    device = "cpu"  # Pair 2/3 state is CPU-only (see continual_read.py); keep Pair 1 on the same device

    tokenizer = load_tokenizer(tcfg.tokenizer_path)
    gcfg = Pair1Config(vocab_size=tokenizer.get_vocab_size())

    model = Pair1Foundation(gcfg).to(device)
    model.load_state_dict(torch.load(f"{tcfg.out_dir}/pair1_final.pt", map_location=device))
    model.eval()

    pair2 = ShortTermMemory(Pair2Config(dim=gcfg.n_embd))
    pair3 = LongTermStore(Pair3Config(dim=gcfg.n_embd))
    if os.path.exists(PAIR2_STATE_PATH):
        pair2.load(PAIR2_STATE_PATH)
    if os.path.exists(PAIR3_STATE_PATH):
        pair3.load(PAIR3_STATE_PATH)
        print(f"[Pair 3 long-term store loaded: {len(pair3)} entries]")

    def run(question):
        answer, ctx = ask(model, tokenizer, device, question, pair2=pair2, pair3=pair3)
        if ctx:
            print(f"  (injected: {' | '.join(ctx)})")
        print(answer)

    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
        return

    print("Interactive mode. Type a question, or 'quit' to exit.")
    while True:
        q = input("\n> ").strip()
        if q.lower() == "quit" or not q:
            break
        run(q)


if __name__ == "__main__":
    main()
