# Research notes: Hebbian approaches to learning language

**How these notes were made.** Web search worked; fetching full papers did not (arXiv,
PMC and most publishers are blocked from this sandbox). Everything below comes from
search-result summaries, abstracts and secondary pages, **not** from reading the full
papers. Where a detail matters for the design and I could not confirm it, it says so.
The design in `DESIGN.md` borrows ideas; it does not reproduce any of these papers.

## 1. What "Hebbian" means here

Hebb's rule: a synapse changes using only the activity of the two neurons it connects
("cells that fire together wire together"). Pure Hebb grows without bound, so every
practical variant adds a *local* stabiliser:

| Rule | Idea | Used in this project as |
|---|---|---|
| Oja (1982), Sanger / GHA (1989) | Hebb term plus a decay term; weight vector converges to a principal component, GHA gives several | the "instar" update of the category units (move prototype toward the input, keep it normalised) |
| Covariance / BCM-style sliding threshold | Change is proportional to (pre − mean pre)(post − mean post), so ubiquitous inputs carry no weight | topic associations (PPMI-style weights from co-activity counters) |
| Competitive learning with global inhibition (k-winners-take-all) | Units compete; only the winner learns, so units specialise | category units and the ART-style recruitment of new ones |
| Asymmetric / spike-timing-dependent (STDP-like) Hebb | "pre then post" strengthens, giving directional sequence memory | word → next-word synapses, conjunction cells for word pairs |
| Synaptic scaling / homeostasis | Divisive normalisation keeps total input fixed | row normalisation of every synapse matrix |

Sources: [Generalized Hebbian algorithm (Oja, Sanger)](https://en.wikipedia.org/wiki/Generalized_Hebbian_algorithm);
[Krotov & Hopfield, *Unsupervised learning by competing hidden units*, PNAS 2019](https://arxiv.org/abs/1806.10181)
(local rule + global inhibition learns useful features with no labels, per its abstract).

## 2. Work most relevant to language

**Assembly Calculus / "language organ" (Mitropolsky & Papadimitriou).**
A simulated system of neuron-like units, synapses and brain areas with Hebbian
plasticity and *no backpropagation* learns nouns, verbs and their meanings from a modest
number of grounded sentences; it builds on an earlier biologically plausible English
parser. [The Architecture of a Biologically Plausible Language Organ (2023)](https://arxiv.org/abs/2306.15364),
[A Biologically Plausible Parser (2021)](https://arxiv.org/abs/2108.02189).
*Takeaway:* word learning, binding and syntax can all come from Hebbian plasticity plus
competition. *Not adopted:* their neuron-level simulation and grounded perception; I could
not read the experimental detail and it is a much larger build.

**Grammar from Hebbian learning (Pulvermüller & Knoblauch 2009, *Neural Networks* 22(2):161-172).**
Per the abstract: auto- and hetero-associative connectivity, sequence detectors and
unsupervised Hebbian learning give "discrete combinatorial neuronal assemblies" that
respond the same way to learned strings and to *unseen word sequences that obey a rule*,
because words that substitute for each other in the same context get bound together.
*Takeaway (core of this project):* generalisation to new word combinations comes from
**category units that words get bound to**, plus **sequence synapses between categories**.

**Fly-inspired word embeddings (Liang et al., ICLR 2021).**
A winner-take-all circuit (Kenyon cells + inhibitory APL neuron) learns the correlation
structure between words and their context and yields sparse binary codes; competitive on
similarity tasks at small code sizes.
[Can a Fruit Fly Learn Word Embeddings?](https://arxiv.org/abs/2101.06887)
*Takeaway:* sparse competitive codes learn usable word representations cheaply.
*Caveat:* my search summary did not confirm the exact update rule; I did not copy it.

**Hebbian tokenisation / embeddings (2025 preprint).** A hierarchy of neurons learns to
tokenise text and a second set binds the patterns into embeddings.
[Hebbian learning the local structure of language](https://arxiv.org/abs/2503.02057).
Only the summary was visible to me.

**Sequence memory.** Hawkins & Ahmad's neuron model with many dendritic synapses learns
high-order sequences, with "predictive" depolarisation that biases which cells fire
([Why Neurons Have Thousands of Synapses, 2016](https://arxiv.org/abs/1511.00083)).
Keeler (1987) showed Kanerva's sparse distributed memory can store sequences when
time-delayed connections are added ([NTRS report](https://ntrs.nasa.gov/api/citations/19870017148/downloads/19870017148.pdf)).
Temporally asymmetric Hebbian (STDP-like) rules can learn sequences (Matsumoto & Okada; COSYNE 2022 abstract).
*Takeaway:* directional pre→post Hebbian synapses + a predictive (modulatory) signal
that gates which unit fires. The topic signal in this project plays that modulatory role.

**Willshaw / sparse associative memory.** Efficient but needs very sparse codes, which is
hard on real data; competitive learning can supply them
([Sa-Couto & Wichert](https://arxiv.org/abs/2301.02196)).

## 3. How children find categories and words (inspiration for the unsupervised parts)

* Syntactic categories can be found from local distributional context alone (words in
  similar neighbour contexts cluster into noun/verb-like classes); works best for frequent
  words and struggles with ambiguous ones. Review:
  [Unsupervised Language Acquisition: Theory and Practice](https://arxiv.org/pdf/cs/0212024)
  (describes Finch & Chater / Redington et al.).
* Infants segment speech using transitional probabilities (Saffran, Aslin & Newport 1996);
  a later [meta-analysis](https://www.mpi.nl/publications/item_2475527) found the effect
  smaller and more fragile than the original studies suggested. So I treat it as
  inspiration, not proof.

## 4. Continual learning with Hebbian plasticity

Sparse, k-winner activations limit interference between old and new knowledge, and fast
plastic weights plus slow weights (complementary learning systems) reduce forgetting.
Surveys / examples:
[Hebbian plasticity in sparse and predictive-coding networks (survey)](https://arxiv.org/abs/2407.17305),
[Hebbian Continual Representation Learning](https://arxiv.org/abs/2207.04874),
[CLS-style pattern separation/completion](https://arxiv.org/abs/2507.11393).
*Takeaway:* keep learning always on, keep activity sparse, keep a fast (one-shot) path
for new items and a slower consolidation path for categories. That is the structure used here.

## 5. What this means for the design (and honest limits)

1. Local Hebbian-family rules can learn word → word order, word categories and
   category → category order, which is exactly what "basic words + grammar" needs.
2. The published systems that go further (parsing, grounded meaning) use much richer
   machinery than I can validate here. This project is a **small, testable** cousin:
   a Hebbian *sequence + category + association* model, not a full language organ.
3. A model like this is statistically close to a class-based n-gram model with learned
   classes and a topic gate. It can generalise to new word combinations and keep
   learning; it will **not** reason, handle long-range dependencies, or match modern
   neural LMs. The evaluation in `README.md` measures what it actually does.

## 6. Addendum: why a dictionary, a semantic memory and a rule engine (the "workspace")

The language layer is good at grammar and word categories but stores facts only diffusely, and it can only ever
know the words it has seen. Three classical ideas fill those gaps. As above, these notes come from search
summaries, not full papers.

* **A dictionary supplies meaning.** WordNet groups words into synonym sets, each with a gloss and links such as
  hypernym ("is a kind of") and antonym; created by George Miller's group at Princeton and later directed by
  Christiane Fellbaum ([WordNet](https://en.wikipedia.org/wiki/WordNet),
  [Fellbaum, *WordNet: An Electronic Lexical Database*](https://pages.cs.wisc.edu/~lingfeng/bn_pages/wordnet-christine-fellbaum.html)).
  It is used here through its plain-text files; it has known inconsistencies, so it is a resource, not ground truth.
* **A semantic network with spreading activation stores and retrieves meaning.** Retrieval is activation of a
  concept's representation, which spreads to related concepts, and residual activation makes recent items easier to
  retrieve: Collins & Loftus, *A spreading activation theory of semantic processing*, Psychological Review 82(6),
  1975, after Quillian's 1967 model ([overview](https://en.wikipedia.org/wiki/Spreading_activation)). The semantic
  memory in this project is a small version: edges strengthened by repetition, rivals inhibited, recency, climbing
  "is a" links when nothing is stored.
* **Two memory systems with different speeds.** McClelland, McNaughton & O'Reilly (*Why there are complementary
  learning systems in the hippocampus and neocortex*, Psychological Review 102(3), 1995) argue for a fast,
  sparse, pattern-separated store for new episodes and a slow, overlapping system that gradually extracts structure
  ([paper](https://stanford.edu/%7Ejlmcc/papers/McCMcNaughtonOReilly95.pdf)). In the workspace, the semantic memory
  is the fast exact store and the Hebbian language layer is the slow statistical one.

What is original here is only the particular combination and the engineering, not the ideas; the closest relatives
are classical semantic-network and neuro-symbolic systems. Whether it generalises to open-ended English is exactly
what the coverage numbers in README.md are meant to show honestly (it does not).
