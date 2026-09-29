"""Layer 0 - encoders: text -> unit vector.

HashEncoder : no training, numpy only (Phase A: isolate the memory logic).
GRUEncoder  : small byte-level GRU trained from scratch on next-byte
              prediction (Phase B). Needs torch.
"""
import re
import zlib

import numpy as np

_WORD = re.compile(r"\w+")


class HashEncoder:
    """Signed feature hashing of words, word bigrams and char trigrams."""

    def __init__(self, dim=1024):
        self.dim = dim
        self.name = "hash"

    def _add(self, v, feat, w):
        h = zlib.crc32(feat.encode("utf-8"))
        v[h % self.dim] += w if (h >> 31) & 1 else -w

    def encode(self, text):
        words = _WORD.findall(text.lower())
        v = np.zeros(self.dim, dtype=np.float32)
        for i, w in enumerate(words):
            self._add(v, "w:" + w, 1.0)
            if i + 1 < len(words):
                self._add(v, "b:" + w + " " + words[i + 1], 0.7)
            padded = f"^{w}$"
            for j in range(len(padded) - 2):
                self._add(v, "c:" + padded[j:j + 3], 0.3)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def save(self, path):
        pass

    def load(self, path):
        pass


class GRUEncoder:
    """Byte-level GRU; sentence vector = mean of hidden states."""

    def __init__(self, dim=128, emb=32):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.dim = dim
        self.name = "gru"

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(256, emb)
                s.gru = nn.GRU(emb, dim, batch_first=True)
                s.out = nn.Linear(dim, 256)

            def forward(s, x):
                h, _ = s.gru(s.emb(x))
                return h, s.out(h)

        self.net = Net()

    def _bytes(self, text, max_len=256):
        b = list(text.encode("utf-8", "ignore")[:max_len]) or [32]
        return self.torch.tensor([b], dtype=self.torch.long)

    def train_lm(self, texts, epochs=3, lr=3e-3, log=print):
        torch = self.torch
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for ep in range(epochs):
            total = 0.0
            for t in texts:
                x = self._bytes(t)
                if x.size(1) < 2:
                    continue
                _, logits = self.net(x[:, :-1])
                loss = torch.nn.functional.cross_entropy(logits[0], x[0, 1:])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            log(f"  gru epoch {ep + 1}/{epochs}: loss {total / max(1, len(texts)):.3f}")
        self.net.eval()

    def encode(self, text):
        torch = self.torch
        with torch.no_grad():
            h, _ = self.net(self._bytes(text))
        v = h[0].mean(0).numpy().astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def save(self, path):
        self.torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(self.torch.load(path))
        self.net.eval()


class SentenceEncoder:
    """Pretrained sentence embedder (all-MiniLM-L6-v2, 384-d, ~90 MB).
    Similar MEANING -> nearby vectors, even with different words.
    Not from scratch: it stands in for a pre-built language sense."""

    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = "minilm"

    def encode(self, text):
        return self.model.encode(text, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)

    def save(self, path):
        pass

    def load(self, path):
        pass


def make_encoder(name):
    if name == "minilm":
        return SentenceEncoder()
    return HashEncoder(512)
