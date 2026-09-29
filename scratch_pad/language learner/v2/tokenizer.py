"""BPE tokenizer trained on the combined book + Q&A-behavior corpus.

Both files are used for vocabulary so the tokenizer never hits [UNK] on
question-format text at inference time, even though the book itself
contains no questions.
"""

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_tokenizer(corpus_paths, out_path: str, vocab_size: int = 4000):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
    )
    tokenizer.train(corpus_paths, trainer)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokenizer.save(out_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


if __name__ == "__main__":
    tok = train_tokenizer(
        ["data/alice_clean.txt", "data/qa_behavior.txt"],
        "data/tokenizer.json",
        vocab_size=4000,
    )
    print(f"Trained tokenizer with vocab_size={tok.get_vocab_size()} -> data/tokenizer.json")
