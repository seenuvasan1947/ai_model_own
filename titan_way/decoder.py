"""Small byte-level decoder: writes a sentence from (question, memory recall).

Two interchangeable backbones (plan_future item 5):
  gru         - 2-layer GRU (recurrent, streaming)
  transformer - tiny causal transformer with a LOCAL attention window
Trained from scratch on synthetic data (synth.py), then frozen. The memory
(neural_memory.py) keeps learning; the decoder only turns recalled content
into words.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from synth import prompt

EOS = 10  # "\n"


class ByteSeq(nn.Module):
    def __init__(self, backbone="gru", d=128, layers=2, window=48, max_len=256):
        super().__init__()
        self.backbone, self.window = backbone, window
        self.emb = nn.Embedding(256, d)
        if backbone == "gru":
            self.core = nn.GRU(d, d, num_layers=layers, batch_first=True)
        else:
            self.pos = nn.Embedding(max_len, d)
            layer = nn.TransformerEncoderLayer(d, 4, d * 3, 0.0, batch_first=True, norm_first=True)
            self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 256)

    def forward(self, x):
        h = self.emb(x)
        if self.backbone == "gru":
            h, _ = self.core(h)
        else:
            T = x.size(1)
            h = h + self.pos(torch.arange(T))[None]
            i = torch.arange(T)
            mask = (i[None, :] > i[:, None]) | (i[:, None] - i[None, :] >= self.window)
            h = self.core(h, mask=mask)
        return self.out(h)

    @torch.no_grad()
    def generate(self, q, m, max_new=140, temp=0.0):
        self.eval()
        ids = list(prompt(q, m).encode("ascii", "ignore"))
        out = []
        for _ in range(max_new):
            logits = self(torch.tensor([ids]))[0, -1]
            nxt = logits.argmax().item() if temp <= 0 else \
                torch.multinomial(F.softmax(logits / temp, -1), 1).item()
            if nxt == EOS:
                break
            ids.append(nxt); out.append(nxt)
        return bytes(out).decode("ascii", "ignore")


def batchify(rows):
    """rows of (q, m, answer) -> input ids, target ids, loss mask (answer only)."""
    seqs, starts = [], []
    for q, m, a in rows:
        p = list(prompt(q, m).encode("ascii", "ignore"))
        s = p + list(a.encode("ascii", "ignore")) + [EOS]
        seqs.append(s); starts.append(len(p))
    L = max(len(s) for s in seqs)
    x = torch.zeros(len(rows), L, dtype=torch.long)
    mask = torch.zeros(len(rows), L)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        x[i, :len(s)] = torch.tensor(s)
        mask[i, st:len(s)] = 1.0
    return x[:, :-1], x[:, 1:], mask[:, 1:]


def load_decoder(path):
    d = torch.load(path)
    m = ByteSeq(d["backbone"])
    m.load_state_dict(d["state"])
    m.eval()
    return m
