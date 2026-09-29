"""Ask the full system a question -- Pair 1 (fixed) generates, informed by
Pair 2 (short-term trace) and Pair 3 (long-term store) if they hold any
state from continual_read.py runs.

v2 change vs. v1 (see GAP_VS_TITANS.md, item 3 -- "Memory as Context"):
Pair 2/3's vectors are spliced directly into Pair 1's hidden-state sequence
via forward_with_memory/generate(memory_vecs=...), instead of being decoded
to text and pasted into the prompt. This avoids the lossy vector->text->
re-tokenize round trip from v1.

Usage:
    python ask.py "Who did Alice meet at the tea party?"
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


@torch.no_grad()
def encode_query(model, tokenizer, device, text: str):
    """Mean-pooled hidden vector for `text`, in the same space Pair 2/3 read
    from -- used to query Pair 3."""
    ids = tokenizer.encode(text).ids[-model.config.context_size:]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, hidden = model(idx, return_hidden=True)
    return hidden[0].mean(dim=0)


def build_memory_vecs(model, tokenizer, device, question: str, pair2=None, pair3=None, max_memory=4):
    """Collect Pair 3's retrieved vectors + Pair 2's current trace into one
    tensor to splice into Pair 1 via memory_vecs (item 3)."""
    vecs = []
    info = []

    if pair3 is not None and len(pair3) > 0:
        query_vec = encode_query(model, tokenizer, device, question)
        retrieved, ret_info = pair3.retrieve_vecs(query_vec, k=max_memory - 1)
        if retrieved.size(0) > 0:
            vecs.append(retrieved)
            info.extend([f"long-term:'{label[:40]}' sim={sim:.2f}" for label, sim in ret_info])

    if pair2 is not None and pair2.current().norm().item() > 1e-6:
        vecs.append(pair2.current().unsqueeze(0))
        info.append(f"short-term trace (norm={pair2.current().norm().item():.3f})")

    if not vecs:
        return None, []

    memory_vecs = torch.cat(vecs, dim=0)[:max_memory]
    return memory_vecs, info[:max_memory]


def ask(model, tokenizer, device, question: str, pair2=None, pair3=None, max_new_tokens=60):
    memory_vecs, info = build_memory_vecs(model, tokenizer, device, question, pair2, pair3)

    prompt = f"Q: {question}\nA:"
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40, memory_vecs=memory_vecs)
    text = tokenizer.decode(out[0].tolist())

    answer_part = text[len(prompt):]
    if "Q:" in answer_part:
        answer_part = answer_part.split("Q:")[0]
    return answer_part.strip(), info


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
        answer, info = ask(model, tokenizer, device, question, pair2=pair2, pair3=pair3)
        if info:
            print(f"  (memory injected as vectors, not text: {' | '.join(info)})")
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
