"""BPE tokenizer trained only on the target language corpus (Phase 1: single-language foundation)."""

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_tokenizer(corpus_path: str, out_path: str, vocab_size: int = 4000):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
    )
    tokenizer.train([corpus_path], trainer)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokenizer.save(out_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    import sys

    corpus = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/tokenizer.json"
    vocab_size = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    tok = train_tokenizer(corpus, out, vocab_size)
    print(f"Trained tokenizer with vocab_size={tok.get_vocab_size()} -> {out}")
