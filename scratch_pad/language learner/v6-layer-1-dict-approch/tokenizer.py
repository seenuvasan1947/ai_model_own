"""BPE tokenizer trained on the full training corpus + Q&A-behavior corpus.

Both files are used for vocabulary so the tokenizer never hits [UNK] on
question-format text at inference time, even though the book text itself
contains no questions.

v4 fix #2 (two bugs, not one):
  1. v3's __main__ was hardcoded to train the tokenizer on
     ["data/alice_clean.txt", "data/qa_behavior.txt"] -- it never included
     Sherlock Holmes text, even though train_pair1.py's TrainConfig.book_path
     pointed at data/combined_corpus.txt (Alice + Sherlock) for the actual
     language-model training. So the tokenizer's vocabulary was built from a
     smaller subset than the model was trained on -- Sherlock-specific words
     never got dedicated BPE merges, forcing them into awkward subword
     fragments. This is the most likely direct cause of the garbled tokens
     ("crudow", "loul") seen in v3's output. Fixed here by training the
     tokenizer on the SAME path as TrainConfig.book_path, not a hardcoded
     alternate file.
  2. vocab_size was fixed at 4000 regardless of corpus size (see
     suggest_vocab_size() below and config.py's Layer1Config.vocab_size).
"""

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def suggest_vocab_size(corpus_paths, min_vocab=2000, max_vocab=12000, words_per_vocab_unit=16):
    """v4 fix #2: rough heuristic so vocab_size isn't left as a fixed magic
    number as the corpus grows. Not a rigorous formula -- just prevents the
    "vocab size never revisited" bug from recurring silently. Scales
    sub-linearly with corpus word count (bigger corpora need proportionally
    fewer NEW vocab entries per word, since common words repeat)."""
    total_words = 0
    for path in corpus_paths:
        with open(path, "r", encoding="utf-8") as f:
            total_words += len(f.read().split())
    suggested = int((total_words ** 0.5) * words_per_vocab_unit)
    return max(min_vocab, min(max_vocab, suggested))


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
    from config import Layer1Config, TrainConfig

    tcfg = TrainConfig()
    corpus_paths = [tcfg.book_path, tcfg.qa_path]  # matches what train_pair1.py actually trains on

    suggested = suggest_vocab_size(corpus_paths)
    configured = Layer1Config().vocab_size
    print(f"Corpus word count -> suggested vocab_size: {suggested:,}  (config.py currently sets: {configured:,})")

    tok = train_tokenizer(corpus_paths, tcfg.tokenizer_path, vocab_size=configured)
    print(f"Trained tokenizer with vocab_size={tok.get_vocab_size()} -> {tcfg.tokenizer_path}")
