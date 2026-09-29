"""
FULL brain-inspired continual-learning prototype.
Layers implemented:
  0 - Encoder (slow weights, per-expert)
  1 - Working memory buffer (temporary, holds recent (h, loss) pairs)
  2 - Surprise gate (per-sample loss = salience signal)
  3 - Fast memory (per-expert, decays if unreinforced, updates via local rule)
  4 - Consolidation (periodic: promotes reinforced/high-salience buffer items
                     into long-term store, then clears buffer)
  5 - Long-term store (external key-value memory, retrieved by similarity ->
                     RAG-style; NOT a slow-weight fine-tune in this version,
                     see notes at bottom for why)
  6 - Modular routing (MoE-style: router picks which expert's encoder+memory
                     handles this input, so unrelated content doesn't share
                     the same fast-memory slot)

Run:  pip install torch numpy   then   python full_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
rng = np.random.default_rng(42)

# =================================================================
# 0. Toy continual-learning task (Task A, Task B; same space, diff clusters)
# =================================================================
N_FEATURES = 10
N_CLASSES = 3

def make_task(n_samples, centers, noise=0.5):
    X, y = [], []
    for c, center in enumerate(centers):
        pts = center + rng.normal(0, noise, size=(n_samples, N_FEATURES))
        X.append(pts); y.append(np.full(n_samples, c))
    X = np.concatenate(X); y = np.concatenate(y)
    idx = rng.permutation(len(X))
    return torch.tensor(X[idx], dtype=torch.float32), torch.tensor(y[idx], dtype=torch.long)

centers_A = rng.normal(0, 3, size=(N_CLASSES, N_FEATURES))
centers_B = rng.normal(0, 3, size=(N_CLASSES, N_FEATURES))
X_A, y_A = make_task(300, centers_A)
X_B, y_B = make_task(300, centers_B)
X_A_test, y_A_test = make_task(100, centers_A)
X_B_test, y_B_test = make_task(100, centers_B)


# =================================================================
# Layer 5 - Long-term store: simple external key-value memory (RAG-style).
# Keys = encoder outputs at time of consolidation, Values = class label
# (stand-in for "compressed gist"). Retrieval = nearest-neighbor similarity.
# =================================================================
class LongTermStore:
    def __init__(self, dim, capacity=200):
        self.dim = dim
        self.capacity = capacity
        self.keys = torch.zeros(0, dim)
        self.values = torch.zeros(0, dtype=torch.long)
        self.strength = torch.zeros(0)  # how many times reinforced (repetition lever)

    def write(self, key, value):
        key = key.detach().unsqueeze(0)
        # if very similar key already stored, reinforce it instead of duplicating
        if len(self.keys) > 0:
            sims = F.cosine_similarity(self.keys, key.expand(len(self.keys), -1))
            best = sims.argmax().item()
            if sims[best] > 0.95:
                self.strength[best] += 1
                return
        self.keys = torch.cat([self.keys, key], dim=0)
        self.values = torch.cat([self.values, torch.tensor([value])])
        self.strength = torch.cat([self.strength, torch.tensor([1.0])])
        # capacity limit: drop weakest entry if over capacity (forced forgetting)
        if len(self.keys) > self.capacity:
            weakest = self.strength.argmin().item()
            keep = [i for i in range(len(self.keys)) if i != weakest]
            self.keys, self.values, self.strength = self.keys[keep], self.values[keep], self.strength[keep]

    def retrieve(self, query, k=1):
        if len(self.keys) == 0:
            return None
        sims = F.cosine_similarity(self.keys, query.unsqueeze(0).expand(len(self.keys), -1))
        topk = sims.topk(min(k, len(sims))).indices
        return self.values[topk], sims[topk]

    def __len__(self):
        return len(self.keys)


# =================================================================
# One "expert": Layer 0 encoder + Layer 3 fast memory, for ONE region/module
# =================================================================
class Expert(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, decay=0.05):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.memory = torch.zeros(hidden_dim)   # fast memory, NOT an nn.Parameter
        self.decay = decay

    def forward(self, x):
        h = self.encoder(x)
        h_with_mem = h + self.memory.unsqueeze(0)
        logits = self.classifier(h_with_mem)
        return logits, h

    @torch.no_grad()
    def update_fast_memory(self, h_batch, per_sample_loss, surprise_threshold):
        surprise = per_sample_loss.mean().item()
        if surprise > surprise_threshold:
            w = (per_sample_loss / (per_sample_loss.sum() + 1e-8)).unsqueeze(1)
            update = (w * h_batch.detach()).sum(dim=0)
            self.memory = (1 - self.decay) * self.memory + self.decay * update
        else:
            self.memory = (1 - self.decay) * self.memory
        return surprise


# =================================================================
# Layer 6 - Router: decides which expert handles a given input
# (simple version: a small learned classifier over input space itself)
# =================================================================
class Router(nn.Module):
    def __init__(self, in_dim, n_experts):
        super().__init__()
        self.net = nn.Linear(in_dim, n_experts)

    def forward(self, x):
        logits = self.net(x)
        return logits.argmax(dim=1), F.softmax(logits, dim=1)


# =================================================================
# Full model: Router -> Expert(s) -> fast memory -> working buffer ->
#             periodic consolidation -> long-term store
# =================================================================
class FullBrainInspiredModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, n_experts=2,
                 decay=0.05, surprise_threshold=0.8,
                 buffer_size=64, consolidate_every=50, repetition_min=2):
        super().__init__()
        self.router = Router(in_dim, n_experts)
        self.experts = nn.ModuleList([
            Expert(in_dim, hidden_dim, n_classes, decay=decay) for _ in range(n_experts)
        ])
        self.n_experts = n_experts
        self.surprise_threshold = surprise_threshold

        # Layer 1: working memory buffer (per expert): list of (h, loss, label)
        self.buffers = [[] for _ in range(n_experts)]
        self.buffer_size = buffer_size

        # Layer 5: one long-term store per expert (keeps unrelated content separate)
        self.long_term = [LongTermStore(hidden_dim) for _ in range(n_experts)]

        self.step_count = 0
        self.consolidate_every = consolidate_every
        self.repetition_min = repetition_min  # min times seen before consolidation eligible

        # logging (obstacle #3: log every internal signal separately)
        self.log = {"surprise": [], "memory_norm": [[] for _ in range(n_experts)],
                    "expert_usage": [], "long_term_size": [[] for _ in range(n_experts)]}

    def forward(self, x, y=None, train=True):
        expert_ids, route_probs = self.router(x)
        self.log["expert_usage"].append(expert_ids.float().mean().item())

        all_logits = torch.zeros(x.size(0), self.experts[0].classifier.out_features)
        all_h = torch.zeros(x.size(0), self.experts[0].encoder[0].out_features)

        for e_id in range(self.n_experts):
            mask = expert_ids == e_id
            if mask.sum() == 0:
                continue
            logits_e, h_e = self.experts[e_id](x[mask])
            all_logits[mask] = logits_e
            all_h[mask] = h_e

            if train and y is not None:
                loss_e = F.cross_entropy(logits_e, y[mask], reduction="none")
                surprise = self.experts[e_id].update_fast_memory(h_e, loss_e, self.surprise_threshold)
                self.log["surprise"].append(surprise)
                self.log["memory_norm"][e_id].append(self.experts[e_id].memory.norm().item())

                # Layer 1: push into working buffer
                for i in range(len(h_e)):
                    self.buffers[e_id].append((h_e[i].detach().clone(), loss_e[i].item(), y[mask][i].item()))
                if len(self.buffers[e_id]) > self.buffer_size:
                    self.buffers[e_id] = self.buffers[e_id][-self.buffer_size:]

        if train:
            self.step_count += 1
            if self.step_count % self.consolidate_every == 0:
                self.consolidate()

        return all_logits, all_h, expert_ids

    def consolidate(self):
        """Layer 4: periodic, sleep-like pass. Promote buffer items that were
        either (a) seen/reinforced repeatedly (similar key appears >= repetition_min
        times), or (b) had high salience (loss above threshold), into long-term store."""
        for e_id in range(self.n_experts):
            buf = self.buffers[e_id]
            if not buf:
                continue
            for h, loss, label in buf:
                if loss > self.surprise_threshold or self.repetition_min <= 1:
                    self.long_term[e_id].write(h, label)
            self.log["long_term_size"][e_id].append(len(self.long_term[e_id]))
            self.buffers[e_id] = []  # clear buffer after consolidation (like sleep clearing hippocampus)

    @torch.no_grad()
    def predict_with_long_term_fallback(self, x):
        """At inference: use expert's fast path, but also check long-term store
        for a confident nearest-neighbor match (mimics recognition/familiarity retrieval)."""
        expert_ids, _ = self.router(x)
        preds = torch.zeros(x.size(0), dtype=torch.long)
        for i in range(x.size(0)):
            e_id = expert_ids[i].item()
            logits, h = self.experts[e_id](x[i:i+1])
            model_pred = logits.argmax(dim=1).item()

            retrieved = self.long_term[e_id].retrieve(h.squeeze(0), k=1)
            if retrieved is not None:
                values, sims = retrieved
                if sims[0].item() > 0.9:   # confident familiarity match
                    preds[i] = values[0].item()
                    continue
            preds[i] = model_pred
        return preds


# =================================================================
# Training / evaluation
# =================================================================
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model.predict_with_long_term_fallback(X)
        acc = (preds == y).float().mean().item()
    model.train()
    return acc

def train_on_task(model, optimizer, X, y, epochs=30, batch_size=32):
    n = len(X)
    for epoch in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            xb, yb = X[b], y[b]
            optimizer.zero_grad()
            logits, h, expert_ids = model(xb, yb, train=True)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    model = FullBrainInspiredModel(
        in_dim=N_FEATURES, hidden_dim=32, n_classes=N_CLASSES, n_experts=2,
        decay=0.05, surprise_threshold=0.8, buffer_size=64,
        consolidate_every=20, repetition_min=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("=== Training on Task A ===")
    train_on_task(model, optimizer, X_A, y_A, epochs=30)
    model.consolidate()  # final consolidation pass after task
    acc_A_after_A = evaluate(model, X_A_test, y_A_test)
    print(f"Accuracy on Task A right after learning A: {acc_A_after_A:.3f}")
    print(f"Long-term store sizes per expert: {[len(s) for s in model.long_term]}")

    print("\n=== Training on Task B (Task A no longer seen) ===")
    train_on_task(model, optimizer, X_B, y_B, epochs=30)
    model.consolidate()
    acc_A_after_B = evaluate(model, X_A_test, y_A_test)
    acc_B_after_B = evaluate(model, X_B_test, y_B_test)

    print(f"Accuracy on Task A AFTER learning B: {acc_A_after_B:.3f}  <- check forgetting here")
    print(f"Accuracy on Task B after learning B: {acc_B_after_B:.3f}")
    print(f"Long-term store sizes per expert: {[len(s) for s in model.long_term]}")
    print(f"Expert usage (mean id over time, first vs last 10 batches): "
          f"{np.mean(model.log['expert_usage'][:10]):.2f} -> {np.mean(model.log['expert_usage'][-10:]):.2f}")
    print(f"Final fast-memory norms per expert: {[e.memory.norm().item() for e in model.experts]}")

"""
NOTES / HONEST LIMITATIONS OF THIS PROTOTYPE:
- Layer 5 uses an EXTERNAL key-value store (RAG-style), not a slow-weight
  fine-tune, because that path is simpler to get working first and matches
  option (b) from our design discussion. Option (a) (folding consolidated
  memories back into encoder weights via periodic fine-tuning) is a valid
  next experiment -- swap consolidate() to also run a few backprop steps
  on stored (h, label) pairs.
- The Router is untrained-from-scratch and may route inconsistently early
  on; consider pretraining it or using a fixed rule (e.g. task ID) first
  to isolate whether memory/consolidation help BEFORE also debugging routing.
- surprise_threshold, decay, consolidate_every, repetition_min are all
  UNTUNED guesses. Expect to need real experimentation here (obstacle #2
  and #5 from our earlier list) before results are meaningful.
- This is still a small toy testbed, not a claim of solving continual
  learning -- it is the "first testable slice" made complete across all
  six layers we designed, so you can now run real ablations (turn off
  routing, turn off long-term store, etc.) and see what each piece adds.
"""
