"""Hebbian language learner.

Everything the model learns is stored in synapses that change by *local* rules: a
synapse only sees the activity of the unit on each side of it (plus a global
competition / normalisation / modulation signal). There is no gradient, no loss
function and no backpropagation. Learning is always on: every token read with
``learn=True`` updates the model, and new words simply recruit new units.

Plastic structures (see DESIGN.md for the full story):

``S1``   word -> next-word synapses (asymmetric, STDP-like Hebb: pre = previous word,
         post = the word that follows)
``P``    one-shot "conjunction cells" (sequence detectors) for the last 2, 3, 4 words -> next
         word; ``PF`` refines each with the sentence's first word (the "frame": a sentence that
         began as a question stays a question); ``PM`` is the sentence-start cell per reading mode
``C``    covariance-Hebbian synapses from a decaying trace of recent content words to the next
         word (the model's topic / fact memory), written with a three-factor rule: plasticity is
         scaled by a global *surprise* signal
``echo_*`` pointer synapses: how often the next word repeats a recently active word that sat in
         a given slot (an answer restating the question's subject)
``U``    Hebbian accumulators of each word's left/right neighbours at distance 1 and 2
         (random-index codes) plus a hashed spelling code; the input to the category layer
``Wc``   competitive category units (instar / Oja-style rule, ART-style recruitment of new
         units when nothing matches, batch settling + merging during "sleep")
``episodes`` a bounded episodic buffer of recent distinct sentences, replayed in "sleep"
         to (re)build the category-level grammar (complementary-learning-systems style)

Category-level grammar (class n-grams up to order 5, class -> word emission) is *read out*
of the episodes through the current word->category assignment, so it is always consistent
with the categories as they stand now.
"""

from __future__ import annotations

import math
import pickle
import zlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .tokenizer import BOS, END_TOKENS, NON_CONTENT, PROTOCOL, split_documents, tokenize


@dataclass
class Config:
    # --- category layer input ---
    index_dim: int = 256          # random-index code length per window slot
    index_nnz: int = 8            # non-zeros per random-index code
    form_dim: int = 64            # hashed character n-gram code (helps rare/new words)
    form_weight: float = 0.5
    window2_weight: float = 0.8   # weight of the distance-2 neighbours relative to distance-1
    # --- competitive category units ---
    max_classes: int = 96
    vigilance: float = 0.3        # cosine below which a new category unit is recruited
    merge_threshold: float = 0.85  # prototypes more similar than this are merged in "sleep"
    batch_iters: int = 3          # batch-Hebbian (fixed point of the instar rule) iterations in "sleep"
    class_lr_min: float = 0.03
    consolidate_every: int = 1500  # sentences between "sleep" replay passes
    consolidate_passes: int = 3
    # --- category-level grammar ---
    class_order: int = 5          # longest category n-gram context
    max_episodes: int = 300000    # bound on the episodic sentence buffer
    # --- topic memory ---
    topic_decay: float = 0.9      # per-token decay of the content-word trace
    topic_min: float = 0.05
    topic_freq_max: float = 0.012  # words more active than this never enter the content trace (homeostasis)
    pmi_eps: float = 1e-3         # smoothing of the association log-ratio
    pmi_shrink: float = 2.0       # evidence is scaled by c / (c + pmi_shrink): few co-occurrences, little trust
    topic_gain: float = 0.75      # strength of the multiplicative topic gate
    topic_cap: float = 12.0
    topic_target_freq_max: float = 0.02  # only words rarer than this are gated by topic (grammar words never are)
    topic_positive_only: bool = True     # gate boosts supported words; it does not suppress the rest
    topic_pool: str = "sum"       # how cue votes combine: 'sum', 'maxmin' (best support + strongest veto), 'max'
    idf_power: float = 0.0        # >0: weight cues by (specificity)^power, specificity = log-inverse activity
    surprise_ref: float = 2.0     # three-factor rule: topic plasticity scales with surprise / surprise_ref
    surprise_max: float = 4.0     #   (clipped to [0.5, surprise_max]); 0 disables the modulation
    lex_surprise: bool = True     # word-level cells are also written harder when the event was surprising (>= 1x)
    min_support: float = 2.5      # abstain ("i do not know") if an uncopied content word of the answer has less evidence
    echo_gain: float = 1.0        # strength of the learned pointer/copy mixture (0 disables)
    echo_min: float = 0.25        # a trace word counts as echoed above this activity
    # --- readout ---
    class_mix: float = 0.7        # weight of the category-level grammar in the readout
    class_trust: float = 1.0      # category evidence is discounted while a context word has been seen fewer times
    cand_trust: float = 8.0       # ... and for a candidate next word that has been seen only a few times
    mix: str = "geometric"        # 'geometric' (product of experts) or 'linear' interpolation
    smoothing: str = "wb"         # back-off rule: 'wb' (Witten-Bell) or 'ad' (absolute discounting)
    wb: float = 1.0               # Witten-Bell back-off scale, word level (small = trust specific contexts)
    wb_class: float = 1.0         # same, category level
    discount: float = 0.75        # absolute discount per observed continuation (smoothing='ad')
    uni_alpha: float = 0.1
    # --- belief revision (explicit instruction only, see tell()) ---
    revise_decay: float = 0.7     # a confident cell contradicted by told input loses this share of its old synapses
    revise_confident: float = 0.6  # a cell is "confident" if one continuation holds at least this share
    revise_novel: float = 0.15    # ... and the told word holds less than this share
    # --- plasticity ---
    recency: float = 0.0          # >0: newer synaptic updates count more (slow forgetting)
    seed: int = 0


class State:
    """Short-term (working-memory) state while reading or generating."""

    __slots__ = ("prev2", "prev1", "first", "hist", "trace", "src", "mode", "at_start", "sent")

    def __init__(self) -> None:
        self.prev2 = -1
        self.prev1 = -1
        self.first = -1               # first word of the current sentence (the "frame" unit)
        self.hist: List[int] = []     # last few words of the sentence (starting with <s>)
        self.trace: Dict[int, float] = {}
        self.src: Dict[int, int] = {}  # word in trace -> the word that preceded it (its "slot")
        self.mode = 0                 # 0 plain text, 1 reading a question, 2 giving an answer
        self.at_start = True
        self.sent: List[int] = []

    def copy(self) -> "State":
        s = State()
        s.prev2, s.prev1, s.first, s.at_start = self.prev2, self.prev1, self.first, self.at_start
        s.mode = self.mode
        s.hist = list(self.hist)
        s.trace = dict(self.trace)
        s.src = dict(self.src)
        s.sent = list(self.sent)
        return s


class HebbianLM:
    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.words: List[str] = []
        self.idx: Dict[str, int] = {}
        self.V = 0
        self.cap = 0
        # conjunction cells for word contexts of length 2..class_order-1 (sequence detectors of rising order)
        self.P: Dict[int, Dict[Tuple[int, ...], list]] = {n: {} for n in range(2, self.cfg.class_order)}
        # sentence-start cells conditioned on the reading mode (plain / question / answer)
        self.PM: Dict[int, list] = {}
        # the same cells refined by the sentence's first word (the "frame": question vs statement ...)
        self.PF: Dict[int, Dict[Tuple[int, ...], list]] = {n: {} for n in range(2, self.cfg.class_order)}
        self.episodes: Dict[Tuple[int, ...], float] = {}
        self._pending: List[Tuple[int, ...]] = []
        # pointer ("echo") synapses: how often the next word repeats a recently active content word
        # that had a given slot (preceding word), in a given context (mode, prev2, prev1)
        self.echo_n: Dict[Tuple[int, int, int, int], float] = {}
        self.echo_e: Dict[Tuple[int, int, int, int], float] = {}
        self.Ntot = 0.0
        self._revise = False
        self.gain = 1.0
        self.sent_count = 0
        self.token_count = 0
        R, F = self.cfg.index_dim, self.cfg.form_dim
        self.dim = 4 * R + F
        self.Wc = np.zeros((self.cfg.max_classes, self.dim), np.float32)
        self.nwin = np.zeros(self.cfg.max_classes)
        self.active = np.zeros(self.cfg.max_classes, bool)
        self.means = np.zeros((4, R), np.float32)
        self._dirty = True
        self._readout: Optional[dict] = None
        self._ro_cls: Optional[np.ndarray] = None
        self._ppmi_cache: Dict[int, np.ndarray] = {}
        self._alloc(256)
        self.bos = self._get_id(BOS)
        self.cls[self.bos] = self.cfg.max_classes - 1   # <s> is a reserved category, never clustered

    # ------------------------------------------------------------------ storage
    def _alloc(self, cap: int) -> None:
        R, F = self.cfg.index_dim, self.cfg.form_dim

        def grow(name: str, shape: tuple, dtype, fill=0):
            old = getattr(self, name, None)
            new = np.full(shape, fill, dtype)
            if old is not None:
                new[tuple(slice(0, s) for s in old.shape)] = old
            setattr(self, name, new)

        grow("RV", (cap, R), np.float32)
        grow("FV", (cap, F), np.float32)
        grow("U", (cap, 4, R), np.float32)    # left1, right1, left2, right2 neighbour accumulators
        grow("S1", (cap, cap), np.float32)
        grow("C", (cap, cap), np.float32)
        grow("cw", (cap,), np.float64)
        grow("cnt", (cap,), np.int64)
        grow("Tf", (cap,), np.float64)
        grow("cls", (cap,), np.int32, fill=-1)
        self.cap = cap

    def _make_index(self, word: str) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(("idx:" + word).encode()))
        v = np.zeros(self.cfg.index_dim, np.float32)
        pos = rng.choice(self.cfg.index_dim, size=self.cfg.index_nnz, replace=False)
        v[pos] = rng.choice(np.array([-1.0, 1.0], np.float32), size=self.cfg.index_nnz)
        return v / math.sqrt(self.cfg.index_nnz)

    def _make_form(self, word: str) -> np.ndarray:
        s = f"^{word}$"
        feats = [s[i:i + 3] for i in range(len(s) - 2)]
        feats += ["suf1:" + word[-1:], "suf2:" + word[-2:], "suf3:" + word[-3:],
                  "pre2:" + word[:2], "len:" + str(min(len(word), 8))]
        v = np.zeros(self.cfg.form_dim, np.float32)
        for g in feats:
            h = zlib.crc32(g.encode())
            v[h % self.cfg.form_dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def _get_id(self, word: str, create: bool = True) -> int:
        i = self.idx.get(word)
        if i is not None:
            return i
        if not create:
            return -1
        if self.V == self.cap:
            self._alloc(self.cap * 2)
        i = self.V
        self.V += 1
        self.words.append(word)
        self.idx[word] = i
        self.RV[i] = self._make_index(word)
        self.FV[i] = self._make_form(word)
        self._dirty = True
        return i

    # ------------------------------------------------------------------ reading / learning
    def new_state(self) -> State:
        return State()

    def _begin_sentence(self, st: State) -> None:
        st.prev2, st.prev1, st.first = -1, self.bos, -1
        st.hist = [self.bos]
        st.at_start = False
        st.sent = []

    def _advance(self, st: State, w: int) -> None:
        if st.first < 0 and st.prev1 == self.bos:
            st.first = w
        before = st.prev1
        st.prev2, st.prev1 = st.prev1, w
        st.hist.append(w)
        if len(st.hist) > self.cfg.class_order:
            del st.hist[0]
        lam, tmin = self.cfg.topic_decay, self.cfg.topic_min
        tr = {f: v * lam for f, v in st.trace.items() if v * lam >= tmin}
        if self.words[w] not in NON_CONTENT and self.cw[w] / max(self.Ntot, 5000.0) < self.cfg.topic_freq_max:
            tr[w] = tr.get(w, 0.0) + 1.0
            st.src[w] = before
        st.trace = tr
        if len(st.src) > len(tr):
            st.src = {f: v for f, v in st.src.items() if f in tr}

    def _learn_step(self, st: State, w: int) -> None:
        """Hebbian updates for one observed transition (pre: context units, post: word w)."""
        prev2, prev1, trace = st.prev2, st.prev1, st.trace
        g = self.gain
        cfg = self.cfg
        # surprise (a global modulatory signal) is read from the word-level synapses *before* they change
        gt = 1.0
        hist = st.hist
        if cfg.surprise_max > 0:
            row_n = float(self.S1[prev1, :self.V].sum())
            p_est = float(self.S1[prev1, w]) / row_n if row_n > 0 else 0.0
            if len(hist) >= 2:
                ent = self.P[2].get(tuple(hist[-2:]))
                if ent is not None and ent[0] > 0:
                    p_est = max(p_est, ent[1].get(w, 0.0) / ent[0])
            surprise = -math.log(max(p_est, 1e-4))
            gt = min(cfg.surprise_max, max(0.5, surprise / cfg.surprise_ref))
            if cfg.lex_surprise:
                g = g * max(1.0, gt)
        self.S1[prev1, w] += g
        ek = (st.mode, prev2, prev1)
        slots: Dict[int, List[int]] = {}
        for f, tau in trace.items():
            if tau >= cfg.echo_min and f in st.src:
                slots.setdefault(st.src[f], []).append(f)
        for slot, fs in slots.items():
            k4 = ek + (slot,)
            self.echo_n[k4] = self.echo_n.get(k4, 0.0) + 1.0
            if w in fs:
                self.echo_e[k4] = self.echo_e.get(k4, 0.0) + 1.0
        jobs = []
        for n in self.P:
            if len(hist) >= n:
                ctx = tuple(hist[-n:])
                jobs.append((self.P[n], ctx))
                if st.first >= 0 and ctx[0] != self.bos:      # window does not already reach <s>
                    jobs.append((self.PF[n], (st.first,) + ctx))
        if prev1 == self.bos:
            jobs.append((self.PM, st.mode))
        for table, key in jobs:
            ent = table.get(key)
            if ent is None:
                ent = [0.0, {}]
                table[key] = ent
            elif self._revise and ent[0] >= 2.0 and ent[0] > 0:
                # heterosynaptic depression: a confident belief contradicted by told input weakens
                top = max(ent[1].values()) / ent[0]
                if top >= cfg.revise_confident and ent[1].get(w, 0.0) / ent[0] < cfg.revise_novel:
                    for k in list(ent[1]):
                        if k != w:
                            ent[1][k] *= 1.0 - cfg.revise_decay
                    ent[0] = sum(ent[1].values())
            ent[0] += g
            ent[1][w] = ent[1].get(w, 0.0) + g
        g = self.gain
        self.cw[w] += g
        self.cnt[w] += 1
        self.Ntot += g
        C, Tf = self.C, self.Tf
        for f, tau in trace.items():
            if self._revise and tau >= 0.5 and self.cls[w] >= 0:
                # told input contradicts a confident association: weaken the words that could fill the
                # same slot (same learned category) as the told word, for this cue
                same = self.cls[:self.V] == self.cls[w]
                row = C[f, :self.V]
                tot = float(row[same].sum())
                if tot > 1.0 and row[same].max() / tot >= cfg.revise_confident and row[w] / tot < cfg.revise_novel:
                    rivals = same.copy()
                    rivals[w] = False
                    row[rivals] *= 1.0 - cfg.revise_decay
            C[f, w] += g * tau * gt
            Tf[f] += g * tau
        # distributional context accumulators (inputs to the category layer)
        U, RV = self.U, self.RV
        U[w, 0] += RV[prev1]
        U[prev1, 1] += RV[w]
        if prev2 >= 0:
            U[w, 2] += RV[prev2]
            U[prev2, 3] += RV[w]
        self.token_count += 1
        if cfg.recency:
            self.gain *= 1.0 + cfg.recency
            if self.gain > 1e6:
                self._renorm()
        self._dirty = True

    def _renorm(self) -> None:
        g = self.gain
        self.S1 /= g
        self.C /= g
        self.Tf /= g
        self.cw /= g
        self.Ntot /= g
        for table in list(self.P.values()) + list(self.PF.values()) + [self.PM]:
            for ent in table.values():
                ent[0] /= g
                for k in ent[1]:
                    ent[1][k] /= g
        self.gain = 1.0

    def feed_tokens(self, st: State, tokens: Iterable[str], learn: bool = False) -> None:
        for t in tokens:
            if t in PROTOCOL:
                if t == "question":
                    st.mode = 1
                elif t == "answer":
                    st.mode = 2
                continue
            if st.at_start:
                self._begin_sentence(st)
            w = self._get_id(t, create=learn)
            if w < 0:
                continue  # unknown word while not learning: ignored
            if learn:
                self._learn_step(st, w)
                st.sent.append(w)
            self._advance(st, w)
            if t in END_TOKENS:
                st.at_start = True
                if learn:
                    self._end_sentence(st)

    def _end_sentence(self, st: State) -> None:
        if st.sent:
            key = tuple(st.sent)
            self.episodes[key] = self.episodes.get(key, 0.0) + 1.0
            self._pending.append(key)
            if len(self.episodes) > self.cfg.max_episodes:
                keys = list(self.episodes.keys())
                for j in self.rng.choice(len(keys), size=len(keys) // 10, replace=False):
                    del self.episodes[keys[j]]
                self._ro_cls = None          # force a full rebuild of the readout
        for i in st.sent:
            self._class_learn(i)
        st.sent = []
        self.sent_count += 1
        if self.cfg.consolidate_every and self.sent_count % self.cfg.consolidate_every == 0:
            self.consolidate()

    def read(self, text: str, learn: bool = True) -> None:
        """Read text (documents separated by blank lines)."""
        for doc in split_documents(text):
            st = self.new_state()
            self.feed_tokens(st, tokenize(doc), learn=learn)
            if learn and st.sent:  # unterminated trailing fragment
                self._end_sentence(st)

    def read_lines(self, lines: Iterable[str], learn: bool = True) -> None:
        """Each item is one document."""
        for line in lines:
            st = self.new_state()
            self.feed_tokens(st, tokenize(line), learn=learn)
            if learn and st.sent:
                self._end_sentence(st)

    # ------------------------------------------------------------------ category layer
    def _block_weights(self) -> np.ndarray:
        w2 = self.cfg.window2_weight
        return np.array([1.0, 1.0, w2, w2], np.float32)

    def _word_vec(self, i: int) -> np.ndarray:
        bw = self._block_weights()
        parts = []
        for b in range(4):
            u = self.U[i, b]
            n = float(np.linalg.norm(u))
            parts.append(bw[b] * ((u / n - self.means[b]) if n > 0 else np.zeros_like(u)))
        parts.append(self.cfg.form_weight * self.FV[i])
        x = np.concatenate(parts)
        n = float(np.linalg.norm(x))
        return (x / n).astype(np.float32) if n > 0 else x.astype(np.float32)

    def _competitive(self, x: np.ndarray) -> int:
        """Winner-take-all Hebbian (instar/Oja-style) step with ART-style recruitment."""
        cfg = self.cfg
        k, s = -1, -2.0
        if self.active.any():
            sims = self.Wc @ x
            sims[~self.active] = -2.0
            k = int(np.argmax(sims))
            s = float(sims[k])
        if s < cfg.vigilance:
            f = self._free_slot()
            if f >= 0:
                self.Wc[f] = x
                self.active[f] = True
                self.nwin[f] = 1
                return f
        if k >= 0:
            lr = max(cfg.class_lr_min, 1.0 / (self.nwin[k] + 2.0))
            self.Wc[k] += lr * (x - self.Wc[k])
            n = float(np.linalg.norm(self.Wc[k]))
            if n > 0:
                self.Wc[k] /= n
            self.nwin[k] += 1
        return k

    def _free_slot(self) -> int:
        free = np.flatnonzero(~self.active[:-1])        # last slot is reserved for <s>
        return int(free[0]) if free.size else -1

    def _class_learn(self, i: int) -> None:
        if i == self.bos:
            return
        x = self._word_vec(i)
        if x.any():
            k = self._competitive(x)
            if self.cls[i] < 0 and k >= 0:   # new word: immediate provisional category
                self.cls[i] = k

    def consolidate(self, passes: Optional[int] = None) -> None:
        """'Sleep': replay stored word signatures through the same competitive rule, settle
        the prototypes (batch-Hebbian: each prototype moves to the mean of its members, the
        fixed point of the instar rule), merge near-duplicate units, refresh the
        word->category assignment and free unused units."""
        V, cfg = self.V, self.cfg
        passes = cfg.consolidate_passes if passes is None else passes
        seen = np.flatnonzero(self.cnt[:V] >= 1)
        if seen.size == 0:
            return
        # centre the context codes (Hebbian PCA assumes zero-mean input)
        for b in range(4):
            vs = []
            for i in seen:
                if self.cnt[i] < 2:
                    continue
                n = np.linalg.norm(self.U[i, b])
                if n > 0:
                    vs.append(self.U[i, b] / n)
            if vs:
                self.means[b] = np.mean(vs, axis=0)
        X = np.stack([self._word_vec(i) for i in range(V)])
        wts = np.sqrt(self.cnt[:V].astype(np.float64))
        if wts.sum() > 0:
            p = wts / wts.sum()
            order = self.rng.choice(V, size=passes * V, p=p)
            for i in order:
                if X[i].any():
                    self._competitive(X[i])
        has = (self.cnt[:V] >= 1) & (np.abs(X).sum(1) > 0)
        has[self.bos] = False
        for _ in range(cfg.batch_iters):
            if not self.active.any():
                break
            sims = X @ self.Wc.T
            sims[:, ~self.active] = -2.0
            best = sims.argmax(1)
            for k in np.flatnonzero(self.active):
                mem = (best == k) & has
                if mem.any():
                    v = (wts[mem, None] * X[mem]).sum(0)
                    n = float(np.linalg.norm(v))
                    if n > 0:
                        self.Wc[k] = v / n
            self._merge_units()
        self._assign_all(X)
        self._dirty = True

    def _merge_units(self) -> None:
        act = np.flatnonzero(self.active)
        if act.size < 2:
            return
        P = self.Wc[act] @ self.Wc[act].T
        for a in range(len(act)):
            if not self.active[act[a]]:
                continue
            for b in range(a + 1, len(act)):
                if self.active[act[b]] and P[a, b] > self.cfg.merge_threshold:
                    self.active[act[b]] = False

    def _assign_all(self, X: np.ndarray) -> None:
        V, cfg = self.V, self.cfg
        if not self.active.any():
            return
        sims = X @ self.Wc.T
        sims[:, ~self.active] = -2.0
        best = sims.argmax(1)
        bs = sims.max(1)
        order = np.argsort(-self.cnt[:V])
        for i in order:  # recruit units for words that nothing matches
            if bs[i] < cfg.vigilance and X[i].any() and i != self.bos:
                k = self._free_slot()
                if k < 0:
                    break
                self.Wc[k] = X[i]
                self.active[k] = True
                self.nwin[k] = 1
                s = X @ self.Wc[k]
                upd = s > bs
                best[upd] = k
                bs[upd] = s[upd]
        self.cls[:V] = best
        self.cls[self.bos] = cfg.max_classes - 1
        used = np.bincount(best, minlength=cfg.max_classes)
        dead = self.active & (used == 0)
        self.active[dead] = False

    # ------------------------------------------------------------------ readout
    @staticmethod
    def _add_count(table: dict, key, c: int, val: float) -> None:
        ent = table.get(key)
        if ent is None:
            ent = [0.0, {}]
            table[key] = ent
        ent[0] += val
        ent[1][c] = ent[1].get(c, 0.0) + val

    def _tally_episode(self, ro: dict, sent: Tuple[int, ...], val: float) -> None:
        """Add one sentence's category-level n-gram counts to the readout tables."""
        clsx, K, N = ro["clsx"], ro["K"], self.cfg.class_order
        seq = [int(clsx[self.bos])] + [int(clsx[i]) for i in sent]
        first = seq[1] if len(seq) > 1 else K
        tables, ftab = ro["ctab"], ro["ftab"]
        for t in range(1, len(seq)):
            c = seq[t]
            if c == K:
                continue
            for n in range(1, N):
                if t - n < 0:
                    break
                ctx = tuple(seq[t - n:t])
                self._add_count(tables[n], ctx, c, val)
                if n >= 2 and t - n > 0:             # window does not already reach <s>
                    self._add_count(ftab[n], (first,) + ctx, c, val)

    def _ensure_readout(self) -> dict:
        if not self._dirty and self._readout is not None:
            return self._readout
        V, cfg = self.V, self.cfg
        K = cfg.max_classes
        cls = self.cls[:V]
        clsx = np.where(cls >= 0, cls, K)
        same = (self._readout is not None and self._ro_cls is not None
                and len(self._ro_cls) == V and bool((self._ro_cls == cls).all()))
        mass = np.zeros((K, V), np.float32)
        ok = cls >= 0
        mass[cls[ok], np.flatnonzero(ok)] = self.cw[:V][ok]
        ms = mass.sum(1)
        Emit = mass / np.maximum(ms, 1e-12)[:, None]
        pcuni = ms / max(ms.sum(), 1e-12)
        uni = self.cw[:V] + cfg.uni_alpha
        uni = (uni / uni.sum()).astype(np.float32)
        if same:
            ro = self._readout
            ro.update(clsx=clsx, Emit=Emit, pcuni=pcuni, uni=uni)
            for sent in self._pending:       # incremental: only the new sentences
                self._tally_episode(ro, sent, 1.0)
        else:
            ro = dict(clsx=clsx, K=K, Emit=Emit, pcuni=pcuni, uni=uni,
                      ctab=[None] + [dict() for _ in range(cfg.class_order - 1)],
                      ftab=[None, None] + [dict() for _ in range(cfg.class_order - 2)])
            for sent, val in self.episodes.items():
                self._tally_episode(ro, sent, val)
        self._pending = []
        self._ro_cls = cls.copy()
        ro["rel_cand"] = (self.cnt[:V] / (self.cnt[:V] + max(cfg.cand_trust, 1e-9))).astype(np.float32) \
            if cfg.cand_trust > 0 else np.ones(V, np.float32)
        ro["content"] = (self.cw[:V] / max(self.Ntot, 5000.0) < cfg.topic_target_freq_max).astype(np.float32)
        self._readout = ro
        self._ppmi_cache = {}
        self._dirty = False
        return ro

    def _ppmi_row(self, f: int) -> np.ndarray:
        row = self._ppmi_cache.get(f)
        if row is None:
            V = self.V
            c = self.C[f, :V]
            expected = self.Tf[f] * self.cw[:V] / max(self.Ntot, 1e-12)
            eps = self.cfg.pmi_eps
            pmi = np.log((c + eps) / (expected + eps))
            pmi = np.where(pmi > 0, pmi * (c / (c + self.cfg.pmi_shrink)), np.maximum(pmi, -2.0))
            row = pmi.astype(np.float32)
            self._ppmi_cache[f] = row
        return row

    def topic_evidence(self, trace: Dict[int, float]) -> np.ndarray:
        """Evidence (log-likelihood-ratio style) that each word follows, given the content trace."""
        cfg = self.cfg
        rows = []
        for f, tau in trace.items():
            if self.Tf[f] > 0:
                w = tau
                if cfg.idf_power:
                    w *= (math.log(1 + self.Ntot / max(self.Tf[f], 1.0)) / math.log(1 + self.Ntot)) ** cfg.idf_power
                rows.append(w * self._ppmi_row(f))
        if not rows:
            return np.zeros(self.V, np.float32)
        R = np.stack(rows)
        if cfg.topic_pool == "maxmin":
            return R.max(0) + R.min(0)
        if cfg.topic_pool == "max":
            return np.maximum(R.max(0), 0)
        return R.sum(0)

    def _blend(self, ent, size: int, lower: np.ndarray, scale: float) -> np.ndarray:
        """Combine a context cell with the distribution backed off to a shorter context."""
        n, dist = ent
        if self.cfg.smoothing == "ad":
            D = self.cfg.discount
            out = lower * (D * len(dist) / n)
            idx = np.fromiter(dist.keys(), np.int64, len(dist))
            out[idx] += np.maximum(np.fromiter(dist.values(), np.float64, len(dist)) - D, 0.0) / n
            return out
        lam = n / (n + scale * len(dist))
        return lam * self._dist_from(ent, size) + (1 - lam) * lower

    @staticmethod
    def _dist_from(ent, size: int) -> np.ndarray:
        n, dist = ent
        d = np.zeros(size, np.float32)
        d[list(dist.keys())] = np.fromiter(dist.values(), np.float32, len(dist)) / n
        return d

    def predict(self, st: State, use_class: bool = True, use_topic: bool = True,
                use_frame: bool = True) -> np.ndarray:
        """Probability distribution over the next word given the working-memory state."""
        ro = self._ensure_readout()
        cfg = self.cfg
        V = self.V
        prev1 = st.prev1 if st.prev1 >= 0 else self.bos
        prev2 = st.prev2
        first = st.first if use_frame else -1
        uni = ro["uni"]
        # --- word-level (lexical) sequence synapses: back-off chain, most specific last ---
        row = self.S1[prev1, :V]
        n2 = float(row.sum())
        lex = uni
        if n2 > 0:
            t2 = int(np.count_nonzero(row))
            if cfg.smoothing == "ad":
                lex = np.maximum(row - cfg.discount, 0.0) / n2 + uni * (cfg.discount * t2 / n2)
            else:
                lam = n2 / (n2 + cfg.wb * t2)
                lex = lam * (row / n2) + (1 - lam) * uni
        hist = st.hist if st.hist else [self.bos]
        chain = []
        for n in sorted(self.P):
            if len(hist) >= n:
                ctx = tuple(hist[-n:])
                chain.append((self.P[n], ctx))
                if first >= 0 and ctx[0] != self.bos:
                    chain.append((self.PF[n], (first,) + ctx))
        if prev1 == self.bos:                      # first word of a sentence: depends on the reading mode
            chain.append((self.PM, st.mode))
        for table, key in chain:
            ent = table.get(key)
            if ent is not None:
                lex = self._blend(ent, V, lex, cfg.wb)
        p = lex
        self._last_E = None
        # --- category-level grammar: class n-grams (order <= 5) + frame (generalises to
        #     word combinations never seen) ---
        if use_class and cfg.class_mix > 0:
            K, clsx = ro["K"], ro["clsx"]
            pc = ro["pcuni"]
            hc = [int(clsx[i]) for i in hist]
            for n in range(1, min(cfg.class_order, len(hc) + 1)):
                ctx = tuple(hc[-n:])
                ent = ro["ctab"][n].get(ctx)
                if ent is not None:
                    pc = self._blend(ent, K, pc, cfg.wb_class)
                if first >= 0 and n >= 2 and hist[-n] != self.bos:
                    ent = ro["ftab"][n].get((int(clsx[first]),) + ctx)
                    if ent is not None:
                        pc = self._blend(ent, K, pc, cfg.wb_class)
            pcls = pc @ ro["Emit"]
            # a word seen only a few times has an unreliable category: trust word-level cells instead
            rho = cfg.class_mix
            for i in hist[-3:]:
                if i != self.bos:
                    c = float(self.cnt[i])
                    rho = min(rho, cfg.class_mix * c / (c + cfg.class_trust)) if c < 10 * cfg.class_trust else rho
            rho = rho * ro["rel_cand"]                       # per candidate word
            if cfg.mix == "geometric":
                p = np.exp((1 - rho) * np.log(np.maximum(lex, 1e-12))
                           + rho * np.log(np.maximum(pcls, 1e-9)))
            else:
                p = (1 - rho) * lex + rho * pcls
        # --- topic gate (multiplicative, like predictive depolarisation) ---
        if use_topic and cfg.topic_gain > 0 and st.trace:
            E = self.topic_evidence(st.trace)
            if cfg.topic_positive_only:
                E = np.maximum(E, 0.0)
            E = E * ro["content"]
            self._last_E = E
            p = p * np.exp(cfg.topic_gain * np.clip(E, -cfg.topic_cap, cfg.topic_cap))
        p = np.asarray(p, np.float64)
        # --- pointer / copy mixture: in contexts where the next word usually repeats a word that
        #     was just said in a given slot (e.g. an answer restating the question's subject) ---
        if cfg.echo_gain > 0 and st.trace:
            ctx = (st.mode, prev2, prev1)
            groups: Dict[int, List[int]] = {}
            for f, tau in st.trace.items():
                if tau >= cfg.echo_min and f in st.src:
                    groups.setdefault(st.src[f], []).append(f)
            ptr: Dict[int, float] = {}
            for slot, fs in groups.items():
                k4 = ctx + (slot,)
                n_k = self.echo_n.get(k4, 0.0)
                rate = self.echo_e.get(k4, 0.0) / (n_k + 1.0) if n_k > 0 else 0.0
                if rate <= 0:
                    continue
                tot = sum(st.trace[f] for f in fs)
                for f in fs:
                    ptr[f] = ptr.get(f, 0.0) + rate * st.trace[f] / tot
            pi = sum(ptr.values())
            if pi > 0:
                pi_c = min(0.98, pi * cfg.echo_gain)
                p = (1.0 - pi_c) * p / max(p.sum(), 1e-300)
                for f, a in ptr.items():
                    p[f] += pi_c * a / pi
        p[self.bos] = 0.0
        s = p.sum()
        return p / s if s > 0 else np.full(V, 1.0 / V)

    # ------------------------------------------------------------------ scoring / generation
    def logprob(self, text_or_tokens, use_class: bool = True, use_topic: bool = True,
                use_frame: bool = True) -> float:
        """Total log-probability of a sentence (or several) under the model; read-only."""
        toks = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else list(text_or_tokens)
        st = self.new_state()
        lp = 0.0
        for t in toks:
            if t in PROTOCOL:
                if t == "question":
                    st.mode = 1
                elif t == "answer":
                    st.mode = 2
                continue
            if st.at_start:
                self._begin_sentence(st)
            w = self.idx.get(t, -1)
            if w < 0:
                lp += math.log(1e-9)
                continue
            p = self.predict(st, use_class, use_topic, use_frame)
            lp += math.log(max(p[w], 1e-12))
            self._advance(st, w)
            if t in END_TOKENS:
                st.at_start = True
        return lp

    def _sample(self, p: np.ndarray, temperature: float, top_k: int, rng) -> int:
        p = p.copy()
        if top_k and top_k < len(p):
            cut = np.partition(p, -top_k)[-top_k]
            p[p < cut] = 0.0
        if temperature != 1.0:
            p = p ** (1.0 / max(temperature, 1e-6))
        s = p.sum()
        p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
        return int(rng.choice(len(p), p=p))

    def continue_from(self, st: State, max_len: int = 20, temperature: float = 0.8,
                      top_k: int = 15, rng=None, use_class: bool = True,
                      use_topic: bool = True, use_frame: bool = True,
                      support: Optional[List[float]] = None) -> List[str]:
        """Generate until the end of the current sentence. If `support` is a list, the topic
        evidence behind each uncopied content word is appended to it (used for confidence)."""
        rng = rng or self.rng
        out: List[str] = []
        st = st.copy()
        for _ in range(max_len):
            if st.at_start:
                self._begin_sentence(st)
            p = self.predict(st, use_class, use_topic, use_frame)
            w = self._sample(p, temperature, top_k, rng)
            tok = self.words[w]
            if support is not None and w not in st.trace:
                ro = self._readout
                if ro is not None and ro["content"][w] > 0 and tok not in NON_CONTENT:
                    support.append(float(self._last_E[w]) if self._last_E is not None else 0.0)
            out.append(tok)
            self._advance(st, w)
            if tok in END_TOKENS:
                break
        return out

    def generate(self, prompt: str = "", **kw) -> List[str]:
        st = self.new_state()
        self.feed_tokens(st, tokenize(prompt), learn=False)
        return self.continue_from(st, **kw)

    def ask(self, question: str, max_len: int = 16, temperature: float = 0.3,
            top_k: int = 5, rng=None, abstain: bool = False, min_support: Optional[float] = None,
            **kw) -> str:
        """Question/answer protocol: 'question : <q> ? answer :' then free generation.

        With ``abstain=True`` the model says "i do not know ." when the question contains a word it
        has never met, or when a content word of its own answer (one it did not copy from the
        question) is not backed by what the question's cue words are associated with."""
        q = tokenize(question)
        if not q or q[-1] not in END_TOKENS:
            q.append("?")
        if abstain and any(t not in self.idx and t not in PROTOCOL for t in q):
            return "i do not know ."           # the question mentions a word it has never met
        st = self.new_state()
        self.feed_tokens(st, ["question", ":"] + q + ["answer", ":"], learn=False)
        support: List[float] = []
        out = self.continue_from(st, max_len=max_len, temperature=temperature, top_k=top_k, rng=rng,
                                 support=support if abstain else None, **kw)
        if abstain:
            thr = self.cfg.min_support if min_support is None else min_support
            if not support or min(support) < thr:
                return "i do not know ."
        return " ".join(out)

    def tell(self, text: str, repeats: int = 1, revise: bool = True) -> None:
        """Learn from a statement (continual, one-shot-capable). With ``revise`` the statement may
        overturn a confident older belief (competing synapses in the contradicted cell are
        depressed); bulk training via read()/read_lines() never does this."""
        self._revise = revise
        try:
            for _ in range(repeats):
                self.read(text, learn=True)
        finally:
            self._revise = False

    def teach_qa(self, question: str, answer: str, revise: bool = True) -> None:
        q = question.strip()
        if not q.endswith("?"):
            q += " ?"
        self._revise = revise
        try:
            self.read(f"question : {q}\nanswer : {answer.strip()}", learn=True)
        finally:
            self._revise = False

    # ------------------------------------------------------------------ introspection / io
    def category_of(self, word: str) -> int:
        i = self.idx.get(word, -1)
        return int(self.cls[i]) if i >= 0 else -1

    def categories(self) -> Dict[int, List[str]]:
        out: Dict[int, List[Tuple[int, str]]] = {}
        for i in range(self.V):
            k = int(self.cls[i])
            if k >= 0:
                out.setdefault(k, []).append((int(self.cnt[i]), self.words[i]))
        return {k: [w for _, w in sorted(v, reverse=True)] for k, v in sorted(out.items())}

    def stats(self) -> dict:
        return dict(words=self.V, tokens=self.token_count, sentences=self.sent_count,
                    categories=int(self.active.sum()), word_cells=sum(len(t) for t in self.P.values()), frame_cells=sum(len(t) for t in self.PF.values()),
                    episodes=len(self.episodes),
                    word_synapses=int(np.count_nonzero(self.S1[:self.V, :self.V])))

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(path: str) -> "HebbianLM":
        with open(path, "rb") as f:
            m = pickle.load(f)
        m._dirty = True
        m._readout = None
        return m
