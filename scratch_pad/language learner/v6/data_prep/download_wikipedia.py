"""Download Simple English Wikipedia via HuggingFace and clean it into a
subset for Layer 1 training.

Simple English Wikipedia uses a controlled vocabulary and simpler grammar
than full Wikipedia -- much better fit for training a small model on
"neutral factual English." Complements TinyStories: TinyStories teaches
narrative and dialog patterns, Simple Wikipedia teaches factual/expository
patterns.

Output: data/simple_wiki.txt (~TARGET_WORDS words, default 5M).
Run:    python data_prep/download_wikipedia.py
"""

import os
import re

TARGET_WORDS = 5_000_000
OUT_PATH = "data/simple_wiki.txt"


def clean(text: str) -> str:
    # Strip templates, refs, section headers, and collapse whitespace.
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # keep displayed link text
    text = re.sub(r"={2,}[^=]+={2,}", " ", text)  # section headers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Missing dependency: run './.venv/bin/pip install datasets' first."
        )

    print("Loading Simple English Wikipedia from HuggingFace...")
    # The 'wikimedia/wikipedia' repo has simple-english as one of its configs.
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)

    os.makedirs("data", exist_ok=True)
    words = 0
    articles = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in ds:
            text = clean(ex.get("text", ""))
            if len(text.split()) < 20:
                continue  # skip stubs / near-empty articles
            f.write(text + "\n\n")
            words += len(text.split())
            articles += 1
            if articles % 500 == 0:
                print(f"  {articles:,} articles, {words:,} words so far...")
            if words >= TARGET_WORDS:
                break

    print(f"\nWrote {articles:,} articles ({words:,} words) to {OUT_PATH}")


if __name__ == "__main__":
    main()
