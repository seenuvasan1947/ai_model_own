"""Download TinyStories via HuggingFace and subset it to a target token count.

TinyStories (Eldan & Li, 2023) is a public dataset of ~500M tokens of very
simple, grammatically clean English stories, specifically designed to teach
small language models coherent generation. Best fit for Layer 1's role as
domain-neutral English substrate. See V6_CHANGES.md.

Output: data/tinystories.txt (~TARGET_WORDS words, default 10M).
Run:    python data_prep/download_tinystories.py
"""

import os

TARGET_WORDS = 30_000_000  # subset to this many words
OUT_PATH = "data/tinystories.txt"


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Missing dependency: run './.venv/bin/pip install datasets' first."
        )

    print("Loading TinyStories from HuggingFace (streaming to keep memory low)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    os.makedirs("data", exist_ok=True)
    words = 0
    stories = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex.get("text", "").strip()
            if not text:
                continue
            f.write(text.replace("\n", " ") + "\n\n")
            words += len(text.split())
            stories += 1
            if stories % 1000 == 0:
                print(f"  {stories:,} stories, {words:,} words so far...")
            if words >= TARGET_WORDS:
                break

    print(f"\nWrote {stories:,} stories ({words:,} words) to {OUT_PATH}")


if __name__ == "__main__":
    main()
