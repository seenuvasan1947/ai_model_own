"""titan_way (neural memory only)  vs  hybrid_way (episodic index + neural memory)
vs plain lookup, on the same taught text and questions.

Refusal thresholds are picked on a SEPARATE calibration topic (never on the
test questions). Neural systems are run over 3 seeds.

  python compare.py [minilm|hash]
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_data as D
from agent import HybridAgent, split_sentences, _ta
from brainlm.encoders import make_encoder

TitanAgent = _ta.TitanAgent
SEEDS = (0, 1, 2)
ENC = sys.argv[1] if len(sys.argv) > 1 else "minilm"


class Plain:
    """Baseline: nearest stored sentence by cosine, nothing else."""
    def __init__(self, enc):
        self.enc = make_encoder(enc); self.S = []; self.T = []
    def teach(self, text):
        for s in split_sentences(text):
            self.T.append(s); self.S.append(self.enc.encode(s))
    def ask(self, q):
        sims = np.stack(self.S) @ self.enc.encode(q); i = int(sims.argmax())
        return self.T[i], float(sims[i])


def calibrate(ask, dev_questions, dev_refuse):
    """Best cutoff on the calibration topic: answer iff score >= cutoff."""
    good = [(ask(q)[1], want in ask(q)[0]) for q, want in dev_questions]
    bad = [ask(q)[1] for q in dev_refuse]
    grid = np.arange(0.0, 1.0, 0.01)
    acc = [sum(c and s >= t for s, c in good) + sum(s < t for s in bad) for t in grid]
    best = [t for t, a in zip(grid, acc) if a == max(acc)]
    return float(np.median(best))


def run_groups(ask, thr):
    def hit(q, want):
        t, s = ask(q); return s >= thr and want in t
    same = np.mean([hit(q, w) for q, w in D.SAME])
    diff = np.mean([hit(q, w) for q, w in D.DIFF])
    ref = np.mean([ask(q)[1] < thr for q in D.REFUSE])
    total = (same * len(D.SAME) + diff * len(D.DIFF) + ref * len(D.REFUSE)) / (len(D.SAME) + len(D.DIFF) + len(D.REFUSE))
    return same, diff, ref, total


def build(kind, seed):
    """Returns (agent, ask_fn, teach_seconds) for the systems that need thresholds."""
    tmp = tempfile.mkdtemp()
    return tmp


def teach_all(ag, kind, texts):
    t0 = time.time()
    for tx in texts:
        if kind == "plain":
            ag.teach(tx)
        else:
            ag.learn_text(tx)
    if kind.startswith("hybrid"):
        ag.sleep(replay_batches=5)
    return time.time() - t0


def make(kind, seed):
    tmp = tempfile.mkdtemp()
    if kind == "plain":
        return Plain(ENC), tmp
    if kind == "titan":
        return TitanAgent(tmp, encoder=ENC, seed=seed), tmp
    return HybridAgent(tmp, encoder=ENC, seed=seed, prune=(kind == "hybrid_pruned")), tmp


def asker(ag, kind):
    if kind == "plain":
        return ag.ask
    if kind == "titan":
        def ask_titan(q):
            ranked, _ = ag.recall(q)
            return ranked[0] if ranked else ("", 0.0)
        return ask_titan
    if kind == "hybrid_index":
        return lambda q: ag.ask_index(q)
    if kind == "hybrid_neural":
        return lambda q: ag.ask_neural(q)


def evaluate(kind, seeds):
    base = "hybrid_index" if kind == "hybrid" else kind
    rows, times, sizes, paths = [], [], [], []
    for s in seeds:
        # calibration on the dev topic with a fresh agent of the same kind
        cal_kind = "hybrid" if kind.startswith("hybrid") and kind not in ("hybrid_index", "hybrid_neural") else kind
        ag_dev, t1 = make("hybrid" if cal_kind == "hybrid" else kind, s)
        teach_all(ag_dev, cal_kind, [D.DEV_TEXT])
        dq = D.DEV_SAME + D.DEV_DIFF
        if cal_kind == "hybrid":
            ti = calibrate(asker(ag_dev, "hybrid_index"), dq, D.DEV_REFUSE)
            tn = calibrate(asker(ag_dev, "hybrid_neural"), dq, D.DEV_REFUSE)
        else:
            thr = calibrate(asker(ag_dev, kind), dq, D.DEV_REFUSE)
        shutil.rmtree(t1, ignore_errors=True)

        ag, tmp = make(kind, s)
        times.append(teach_all(ag, kind, D.TOPICS))
        if cal_kind == "hybrid":                       # combined system: index first, then neural
            ai, an = asker(ag, "hybrid_index"), asker(ag, "hybrid_neural")
            def ask(q, ai=ai, an=an, ti=ti, tn=tn):
                t, sc = ai(q)
                if sc >= ti:
                    return t, 1.0
                t2, s2 = an(q)
                return (t2, 1.0) if s2 >= tn else (t2, 0.0)
            rows.append(run_groups(ask, 0.5))
            n_idx = sum(ai(q)[1] >= ti for q, _ in D.SAME + D.DIFF)
            paths.append(n_idx / len(D.SAME + D.DIFF))
            sizes.append((len(ag.idx), sum(w.numel() for w in ag.mem.M)))
        else:
            rows.append(run_groups(asker(ag, kind), thr))
            if kind == "plain":
                sizes.append((len(ag.T), 0))
            elif kind == "hybrid_index":
                sizes.append((len(ag.idx), 0))        # neural memory unused
            elif kind == "hybrid_neural":
                sizes.append((0, sum(w.numel() for w in ag.mem.M)))   # index unused
            else:
                sizes.append((0, sum(w.numel() for w in ag.mem.M)))
        shutil.rmtree(tmp, ignore_errors=True)
    r = np.mean(rows, 0)
    return r, np.mean(times), sizes[0], (np.mean(paths) if paths else None), np.std([x[3] for x in rows])


SYSTEMS = [
    ("plain lookup (no memory model)", "plain", (0,)),
    ("titan_way (neural memory only)", "titan", SEEDS),
    ("hybrid: index only", "hybrid_index", SEEDS),
    ("hybrid: neural only (after sleep)", "hybrid_neural", SEEDS),
    ("hybrid: index + neural", "hybrid", SEEDS),
    ("hybrid + prune (index shrinks)", "hybrid_pruned", SEEDS),
]
if __name__ == "__main__":
    n = (len(D.SAME), len(D.DIFF), len(D.REFUSE))
    print(f"encoder: {ENC} | taught: 4 topics, {sum(len(split_sentences(t)) for t in D.TOPICS)} sentences | "
          f"questions: {n[0]} same-wording, {n[1]} different-wording, {n[2]} unrelated (should refuse)")
    print(f"{'system':<36}{'same':>6}{'diff':>6}{'refuse':>8}{'overall':>9}{'±':>6}{'teach s':>9}   size")
    for label, kind, seeds in SYSTEMS:
        r, t, size, idxfrac, sd = evaluate(kind, seeds)
        sz = f"{size[0]} vectors + {size[1]} params"
        extra = f"  (answered from index: {idxfrac:.0%})" if idxfrac is not None else ""
        print(f"{label:<36}{r[0]:>6.2f}{r[1]:>6.2f}{r[2]:>8.2f}{r[3]:>9.2f}{sd:>6.2f}{t:>9.1f}   {sz}{extra}", flush=True)
