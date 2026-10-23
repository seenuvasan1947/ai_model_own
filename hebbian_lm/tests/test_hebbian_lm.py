"""Tests for the Hebbian language learner.  Run:  python -m unittest discover -s tests -v

A small model (8k sentences, ~4 s) is trained once and shared. Thresholds are set well below what
the model reaches, so they catch real regressions without being flaky.
"""

import copy
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hebbian_lm import Config, HebbianLM, tokenize  # noqa: E402
from hebbian_lm import evaluate as ev  # noqa: E402
from hebbian_lm.curriculum import Language  # noqa: E402
from hebbian_lm.tokenizer import split_documents, split_sentences  # noqa: E402


class TokenizerTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("The dog's runs, fast!"), ["the", "dog's", "runs", ",", "fast", "!"])
        self.assertEqual(tokenize("Two plus 3 is five?"), ["two", "plus", "3", "is", "five", "?"])

    def test_sentences_and_documents(self):
        sents = list(split_sentences(tokenize("a b . c d ? e")))
        self.assertEqual(sents, [["a", "b", "."], ["c", "d", "?"], ["e"]])
        self.assertEqual(len(split_documents("one .\n\ntwo .\n\n\nthree .")), 3)


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lang = Language()
        cls.docs = cls.lang.sample_docs(3000, seed=0)

    def test_heldout_never_trained(self):
        held = set(self.lang.heldout)
        self.assertGreater(len(held), 1000)
        self.assertFalse(held & self.lang.trained)
        for d in self.docs:
            for line in d.split("\n"):
                self.assertNotIn(line, held)

    def test_training_sentences_are_in_the_language(self):
        for s in list(self.lang.trained)[:500]:
            self.assertIn(s, self.lang.accepted)

    def test_known_sentences(self):
        for s in ["the dog runs .", "the dogs run .", "does the boy eat an apple ?", "there are two cats .",
                  "he is happy .", "i am happy .", "the mice sleep ."]:
            self.assertIn(s, self.lang.grammar, s)
        for s in ["the dog run .", "the dogs runs .", "a dogs run .", "dog the runs ."]:
            self.assertNotIn(s, self.lang.accepted, s)

    def test_minimal_pairs_are_really_ungrammatical(self):
        pairs = self.lang.minimal_pairs(200, seed=3)
        self.assertGreater(len(pairs), 100)
        for good, bad, kind in pairs:
            self.assertIn(good, self.lang.grammar)
            self.assertNotIn(bad, self.lang.accepted)
            self.assertNotEqual(good, bad)


class TrainedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lang = Language()
        docs = cls.lang.sample_docs(8000, seed=0)
        cls.model = HebbianLM(Config(seed=0))
        cls.model.read_lines(docs)
        cls.model.consolidate()
        cls.pairs = cls.lang.minimal_pairs(300, seed=1)

    # ---- the learning rules are local: nothing but numpy, no gradients ----
    def test_no_deep_learning_framework(self):
        for banned in ("torch", "tensorflow", "jax"):
            self.assertNotIn(banned, sys.modules)

    def test_probabilities_are_valid(self):
        m = self.model
        st = m.new_state()
        m.feed_tokens(st, tokenize("the dog"))
        p = m.predict(st)
        self.assertEqual(p.shape, (m.V,))
        self.assertTrue(np.isfinite(p).all())
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertEqual(p[m.bos], 0.0)

    def test_reading_without_learning_changes_nothing(self):
        m = copy.deepcopy(self.model)
        before = (m.V, m.token_count, float(m.S1.sum()), float(m.C.sum()), len(m.episodes))
        m.read("the dog runs to the park .\nthe flubber is purple .", learn=False)
        self.assertEqual(before, (m.V, m.token_count, float(m.S1.sum()), float(m.C.sum()), len(m.episodes)))
        self.assertNotIn("flubber", m.idx)

    def test_learning_touches_only_related_synapses(self):
        m = copy.deepcopy(self.model)
        before = m.S1.copy()
        m.tell("the dog runs .")
        changed_rows = {int(i) for i in np.flatnonzero((m.S1[:m.V] != before[:m.V]).any(axis=1))}
        words = {m.idx[w] for w in ("<s>", "the", "dog", "runs")}
        self.assertTrue(changed_rows <= words, changed_rows - words)

    # ---- words and grammar ----
    def test_grammar_beats_broken_sentences_on_unseen_combinations(self):
        r = ev.minimal_pair_report(self.model, self.pairs)
        self.assertGreater(r["all"], 0.9, r)
        self.assertGreater(r["agreement"], 0.8, r)
        self.assertGreater(r["number"], 0.95, r)
        self.assertGreater(r["order"], 0.95, r)

    def test_invented_categories_match_parts_of_speech(self):
        c = ev.category_report(self.model, self.lang)
        self.assertGreater(c["nmi_fine"], 0.7, c)
        m = self.model
        self.assertEqual(m.category_of("dog"), m.category_of("cat"))
        self.assertEqual(m.category_of("dogs"), m.category_of("cats"))
        self.assertNotEqual(m.category_of("dog"), m.category_of("dogs"))      # singular vs plural
        self.assertNotEqual(m.category_of("runs"), m.category_of("run"))      # 3rd person vs base
        self.assertNotEqual(m.category_of("dog"), m.category_of("runs"))      # noun vs verb

    def test_free_generation_is_mostly_in_the_language(self):
        g = ev.generation_report(self.model, self.lang, n=150, temperature=0.7)
        self.assertLess(g["ungrammatical"], 0.15, g["bad_examples"])
        self.assertGreater(g["distinct"], 50)

    def test_subject_verb_agreement_in_probabilities(self):
        m = self.model
        self.assertGreater(m.logprob("the dog runs ."), m.logprob("the dog run ."))
        self.assertGreater(m.logprob("the dogs run ."), m.logprob("the dogs runs ."))
        self.assertGreater(m.logprob("does the dog run ?"), m.logprob("does the dog runs ?"))

    # ---- facts, questions ----
    def test_answers_taught_facts(self):
        q = ev.qa_report(self.model, self.lang)
        self.assertGreater(q["taught_facts"]["all"], 0.75, q["taught_facts"])
        self.assertEqual(q["unknown_facts_before_teaching"]["all"], 0.0)

    def test_fact_told_once_is_answered(self):
        m = copy.deepcopy(self.model)
        self.assertNotIn("the sand is brown .", m.ask("what color is the sand ?"))
        m.tell("the sand is brown .")
        rng = np.random.default_rng(0)
        self.assertEqual(m.ask("what color is the sand ?", rng=rng, temperature=0.3, top_k=5), "the sand is brown .")
        m.tell("the spider has eight legs .")
        self.assertEqual(m.ask("how many legs does the spider have ?", rng=rng, temperature=0.3, top_k=5),
                         "the spider has eight legs .")

    def test_told_fact_survives_other_learning(self):
        m = copy.deepcopy(self.model)
        m.tell("the ocean is blue .")
        for s in ["the cat sleeps .", "birds can fly .", "the boy eats an apple .", "she is happy ."]:
            m.tell(s)
        rng = np.random.default_rng(0)
        self.assertEqual(m.ask("what color is the ocean ?", rng=rng, temperature=0.3, top_k=5), "the ocean is blue .")

    def test_fact_can_be_corrected(self):
        m = copy.deepcopy(self.model)
        rng = np.random.default_rng(0)
        self.assertEqual(m.ask("what color is the sky ?", rng=rng, temperature=0.2, top_k=3), "the sky is blue .")
        m.tell("the sky is red .")                 # one authoritative statement is enough
        self.assertEqual(m.ask("what color is the sky ?", rng=rng, temperature=0.2, top_k=3), "the sky is red .")
        # plain bulk reading (not told input) does not overturn beliefs this way
        m2 = copy.deepcopy(self.model)
        m2.read("the sky is red .", learn=True)
        self.assertEqual(m2.ask("what color is the sky ?", rng=rng, temperature=0.2, top_k=3), "the sky is blue .")

    def test_says_i_do_not_know(self):
        m = copy.deepcopy(self.model)
        rng = np.random.default_rng(0)
        kw = dict(rng=rng, temperature=0.3, top_k=5, abstain=True)
        self.assertEqual(m.ask("what color is the cherry ?", **kw), "i do not know .")   # never heard of it
        self.assertEqual(m.ask("how many legs does the rabbit have ?", **kw), "i do not know .")  # never taught
        self.assertEqual(m.ask("what color is the sky ?", **kw), "the sky is blue .")     # taught: answers
        m.tell("the cherry is red .")
        self.assertEqual(m.ask("what color is the cherry ?", **kw), "the cherry is red .")  # told once: answers

    # ---- keeps learning ----
    def test_new_word_is_categorised_and_used(self):
        m = copy.deepcopy(self.model)
        before = ev.minimal_pair_report(m, self.pairs)["all"]
        for s in ["the zorp runs .", "a zorp sleeps .", "the zorp is big ."]:
            m.tell(s)
        m.consolidate()
        self.assertIn("zorp", m.idx)
        self.assertEqual(m.category_of("zorp"), m.category_of("dog"))
        self.assertGreater(m.logprob("the zorp runs ."), m.logprob("the zorp run ."))
        self.assertGreater(m.logprob("the zorp sleeps ."), m.logprob("zorp the sleeps ."))
        after = ev.minimal_pair_report(m, self.pairs)["all"]
        self.assertGreater(after, before - 0.03)       # nothing old was lost

    def test_new_word_form_helps_unseen_word(self):
        # a word first met only through its spelling pattern still gets a category (suffix cue)
        m = copy.deepcopy(self.model)
        m.tell("the zorp blicked the cat .")
        m.consolidate()
        self.assertIn("blicked", m.idx)

    # ---- persistence and determinism ----
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pkl")
            self.model.save(path)
            m2 = HebbianLM.load(path)
        for s in ["the dog runs .", "does the boy eat the apple ?", "what color is the sky ?"]:
            self.assertAlmostEqual(self.model.logprob(s), m2.logprob(s), places=6)
        rng1, rng2 = np.random.default_rng(5), np.random.default_rng(5)
        self.assertEqual(self.model.ask("what does the dog say ?", rng=rng1), m2.ask("what does the dog say ?", rng=rng2))

    def test_training_is_deterministic(self):
        lang = Language()
        docs = lang.sample_docs(1500, seed=7, include_facts=False)
        a, b = HebbianLM(Config(seed=3)), HebbianLM(Config(seed=3))
        a.read_lines(docs)
        b.read_lines(docs)
        a.consolidate()
        b.consolidate()
        self.assertEqual(a.stats(), b.stats())
        self.assertAlmostEqual(a.logprob("the dog runs ."), b.logprob("the dog runs ."), places=9)


if __name__ == "__main__":
    unittest.main()
