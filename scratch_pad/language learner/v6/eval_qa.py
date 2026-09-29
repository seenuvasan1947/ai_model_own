"""Q&A evaluation harness. Runs each question in data/eval_qa.jsonl twice
(with memory ON and OFF), scores each generated answer by expected-keyword
overlap, and reports per-question results plus aggregate scores.

This is the ablation test that answers "does the memory stack actually help
generation?" -- something v1-v5 could only guess at.

Usage:
    python eval_qa.py
    python eval_qa.py --questions data/eval_qa.jsonl   # explicit path
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


QUESTIONS_PATH = "data/eval_qa.jsonl"


def load_questions(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def score(answer: str, expected_keywords: list) -> float:
    """Fraction of expected keywords present (case-insensitive substring) in the
    generated answer. Simple but useful first-cut metric."""
    if not expected_keywords:
        return 0.0
    ans = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans)
    return hits / len(expected_keywords)


def main():
    path = QUESTIONS_PATH
    if len(sys.argv) > 2 and sys.argv[1] == "--questions":
        path = sys.argv[2]
    if not os.path.exists(path):
        raise SystemExit(f"No questions file at {path}")

    questions = load_questions(path)
    print(f"Loaded {len(questions)} eval questions from {path}\n")

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
    print(f"Stack: {stack.summary()}\n")

    rows = []
    for i, item in enumerate(questions):
        q = item["question"]
        kws = item["expected_keywords"]

        ans_on = ask(layer1, stack, tokenizer, device, q, use_memory=True, max_new_tokens=40)
        ans_off = ask(layer1, stack, tokenizer, device, q, use_memory=False, max_new_tokens=40)

        s_on = score(ans_on, kws)
        s_off = score(ans_off, kws)
        rows.append((q, kws, ans_on, s_on, ans_off, s_off))

        print(f"[{i+1}/{len(questions)}] Q: {q}")
        print(f"  expected keywords: {kws}")
        print(f"  memory ON  ({s_on:.2f}): {ans_on[:100]}")
        print(f"  memory OFF ({s_off:.2f}): {ans_off[:100]}")
        print()

    avg_on = sum(r[3] for r in rows) / len(rows)
    avg_off = sum(r[5] for r in rows) / len(rows)
    delta = avg_on - avg_off
    print("=" * 60)
    print(f"Aggregate keyword-overlap score across {len(rows)} questions:")
    print(f"  memory ON:  {avg_on:.3f}")
    print(f"  memory OFF: {avg_off:.3f}")
    print(f"  delta (ON - OFF): {delta:+.3f}   " +
          ("<- memory helps" if delta > 0.02 else
           "<- memory hurts" if delta < -0.02 else
           "<- memory has no measurable effect"))


if __name__ == "__main__":
    main()
