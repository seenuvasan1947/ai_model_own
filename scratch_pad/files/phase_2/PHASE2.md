# Phase 2 — What This Phase Will Build

## Goal
Make the model reason, not just retrieve.
Phase 1 stores facts and generates from them.
Phase 2 makes the model **derive new facts** from stored ones,
**answer complex questions**, and **handle things it was never explicitly told**.

---

## Capabilities to Build

### 1. Multi-hop Graph Reasoning
Follow chains of facts to derive new knowledge.

Example:
```
stored: (hacker) --[exploits]--> (vulnerability)
stored: (vulnerability) --[exists-in]--> (software)
stored: (software) --[runs-on]--> (server)

derived: hacker can compromise server  (3-hop inference)
```

Rules like "if A attacks B and B runs on C, then A threatens C"
are written as Python logic — no ML, no backprop.

### 2. Word-Sense Disambiguation
Separate meanings of the same word based on context.

Example:
```
"bank" near "river"  → bank_nature
"bank" near "money"  → bank_finance
```

Each sense gets its own node in the knowledge graph and
its own Hebbian association weights.
Learning one sense never pollutes the other.

### 3. Sentence-Level Associations
Right now the model learns word→word pairs.
Phase 2 adds sentence→sentence associations.

Example:
```
"A firewall blocks access." followed by
"The network stays secure."

→ model learns that firewall sentences predict security sentences
→ generation becomes paragraph-coherent, not just sentence-coherent
```

### 4. Broader Knowledge Base
Feed larger corpora to the English base:
- WikiText-103 (~100M words)
- Project Gutenberg books
- Simple English Wikipedia

This fills gaps in what the model can answer
without needing a transformer fallback.

### 5. Confidence Scoring
Every answer comes with a confidence score based on:
- How many facts support this answer
- How directly the facts match the question
- How strong the Hebbian associations are

Low-confidence answers are flagged so the user knows
the model is uncertain.

---

## What Phase 2 Will NOT Do
- No backpropagation (stays true to the original idea)
- No pretrained transformer weights
- No internet lookup

---

## Architecture After Phase 2

```
Layer 1: English Base         (Hebbian, frozen)
Layer 2: Knowledge Graph      (triples + multi-hop inference)
Layer 3: Domain Layer         (Hebbian, continuously updated)
Layer 4: Word-Sense Splitter  (context-based disambiguation)
Layer 5: Sentence Associator  (sentence-to-sentence Hebbian)
Layer 6: Generation           (POS-guided, graph-seeded, confidence-scored)
```

---

## How to Trigger Phase 2
Once Phase 1 is working well and you are satisfied with:
- Fact storage and retrieval
- Basic question answering
- Domain learning from text files

Then Phase 2 begins by adding multi-hop reasoning first,
then word-sense disambiguation, then sentence associations.
