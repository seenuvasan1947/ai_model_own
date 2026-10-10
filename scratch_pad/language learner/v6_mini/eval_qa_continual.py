"""Continual-learning Q&A eval. For each entry in data/eval_qa_continual.jsonl:
  1. Reset the memory stack (L2/L3 back to empty).
  2. Feed the fact into memory via stack.observe() many times so it forms
     strong L2 connections and consolidates into L3.
  3. Ask the question TWICE: once with memory ON, once with memory OFF.
  4. Score both by keyword overlap with the expected answer.

The delta (memory ON - memory OFF) measures how much the memory pathway
actually contributes to answering questions about things the model was
told, but never trained on. This is the test that answers "does continual
learning actually work?" -- separate from eval_qa.py, which tests language
competence baked into Layer 1.

Usage:
    python eval_qa_continual.py
"""

import json
import os
import sys

import torch

from config import Layer1Config, Layer2Config, Layer3Config, StackConfig, TrainConfig
from layer1_language import Pair1Foundation as Layer1Foundation
from stack import MemoryStack
from tokenizer import load_tokenizer

from ask import ask
from eval_qa import score


QUESTIONS_PATH = "data/eval_qa_continual.jsonl"
FACT_REPETITIONS = 5   # how many times to feed each fact into memory


def load_items(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    if not os.path.exists(QUESTIONS_PATH):
        raise SystemExit(f"No questions file at {QUESTIONS_PATH}")
    items = load_items(QUESTIONS_PATH)
    print(f"Loaded {len(items)} continual-learning eval items from {QUESTIONS_PATH}\n")

    tcfg = TrainConfig()
    device = "cpu"
    tokenizer = load_tokenizer(tcfg.tokenizer_path)
    l1_cfg = Layer1Config(vocab_size=tokenizer.get_vocab_size())

    layer1 = Layer1Foundation(l1_cfg).to(device)
    layer1.load_state_dict(torch.load(f"{tcfg.out_dir}/layer1_final.pt", map_location=device))
    layer1.eval()

    rows = []
    for i, item in enumerate(items):
        fact = item["fact"]
        q = item["question"]
        kws = item["expected_keywords"]

        # Fresh memory stack for every question, so we're really testing
        # "does feeding THIS fact help answer THIS question," with no
        # cross-contamination from previous questions.
        stack = MemoryStack(layer1,
                              l2_cfg=Layer2Config(dim=l1_cfg.n_embd),
                              l3_cfg=Layer3Config(dim=l1_cfg.n_embd),
                              stack_cfg=StackConfig())

        # Feed the fact repeatedly so it consolidates from L2 to L3.
        with torch.no_grad():
            fact_ids = tokenizer.encode(fact).ids
            fact_tensor = torch.tensor([fact_ids], dtype=torch.long, device=device)
            for _ in range(FACT_REPETITIONS):
                hidden = layer1.forward_returning_hidden(fact_tensor)
                for t in range(hidden.size(1)):
                    stack.observe(hidden[0, t])

        # Seed identically before each generation so ON vs OFF differ only in
        # the memory contribution, not in which random tokens got sampled.
        torch.manual_seed(1234 + i)
        ans_on = ask(layer1, stack, tokenizer, device, q, use_memory=True, max_new_tokens=40)
        torch.manual_seed(1234 + i)
        ans_off = ask(layer1, stack, tokenizer, device, q, use_memory=False, max_new_tokens=40)

        s_on = score(ans_on, kws)
        s_off = score(ans_off, kws)
        rows.append((fact, q, kws, ans_on, s_on, ans_off, s_off))

        print(f"[{i+1}/{len(items)}] fact taught: {fact}")
        print(f"  Q: {q}")
        print(f"  expected keywords: {kws}")
        print(f"  memory ON  ({s_on:.2f}): {ans_on[:100]}")
        print(f"  memory OFF ({s_off:.2f}): {ans_off[:100]}")
        print(f"  L2 units: {len(stack.layer2.units)}, L3 units: {len(stack.layer3.units)}")
        print()

    avg_on = sum(r[4] for r in rows) / len(rows)
    avg_off = sum(r[6] for r in rows) / len(rows)
    delta = avg_on - avg_off
    print("=" * 60)
    print(f"Continual-learning eval across {len(rows)} items:")
    print(f"  memory ON:  {avg_on:.3f}")
    print(f"  memory OFF: {avg_off:.3f}")
    print(f"  delta (ON - OFF): {delta:+.3f}   " +
          ("<- continual learning WORKS: taught facts get recalled" if delta > 0.05 else
           "<- continual learning HURT (memory made answers worse)" if delta < -0.05 else
           "<- marginal effect: memory didn't measurably help or hurt"))


if __name__ == "__main__":
    main()
