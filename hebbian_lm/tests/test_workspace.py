"""Tests for the dictionary/rule engine, comprehension, semantic memory and the workspace.
They use the bundled mini dictionary, so they run offline; WordNet tests are skipped unless data/wordnet exists."""

import copy
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hebbian_lm import Config, HebbianLM  # noqa: E402
from hebbian_lm.comprehension import ClauseParser, Discourse, tag_sentence, tokenize_cased  # noqa: E402
from hebbian_lm.curriculum import Language  # noqa: E402
from hebbian_lm.lexicon import Lexicon, inflect_noun, inflect_verb  # noqa: E402
from hebbian_lm.workspace import Workspace  # noqa: E402

WN = os.path.join(ROOT, "data", "wordnet")


def parse(text, lex, disc=None):
    toks = tokenize_cased(text)
    tag_sentence(toks, lex)
    p = ClauseParser(lex, disc or Discourse())
    return toks, p


def props(text, lex, disc=None):
    toks, p = parse(text, lex, disc)
    return [(x.subj, x.rel, x.obj, x.qty) for x in p.parse_declarative(toks)]


class MorphologyTests(unittest.TestCase):
    def test_inflection_rules(self):
        self.assertEqual(inflect_noun("dog", "pl"), "dogs")
        self.assertEqual(inflect_noun("box", "pl"), "boxes")
        self.assertEqual(inflect_noun("baby", "pl"), "babies")
        self.assertEqual(inflect_noun("mouse", "pl"), "mice")
        self.assertEqual(inflect_noun("wolf", "pl"), "wolves")
        self.assertEqual(inflect_verb("run", "3sg"), "runs")
        self.assertEqual(inflect_verb("fly", "3sg"), "flies")
        self.assertEqual(inflect_verb("watch", "3sg"), "watches")
        self.assertEqual(inflect_verb("hop", "ing"), "hopping")
        self.assertEqual(inflect_verb("make", "ing"), "making")
        self.assertEqual(inflect_verb("eat", "past"), "ate")
        self.assertEqual(inflect_verb("play", "past"), "played")

    def test_analysis_of_word_forms(self):
        lx = Lexicon()
        a = lx.analyze("zebras")[0]
        self.assertEqual((a.lemma, a.pos, a.feats["number"], a.known), ("zebra", "n", "pl", True))
        a = lx.analyze("mice")[0]
        self.assertEqual((a.lemma, a.feats["number"]), ("mouse", "pl"))
        a = lx.analyze("ran")[0]
        self.assertEqual((a.lemma, a.pos, a.feats["form"]), ("run", "v", "past"))
        a = lx.analyze("hopping")[0]
        self.assertEqual((a.lemma, a.feats["form"]), ("hop", "ing"))
        a = lx.analyze("bigger")[0]
        self.assertEqual((a.lemma, a.pos, a.feats["degree"]), ("big", "a", "comp"))

    def test_unknown_words_are_guessed_from_their_shape(self):
        lx = Lexicon()
        for word, pos, key, val in [("wugs", "n", "number", "pl"), ("blicked", "v", "form", "past"), ("flib", "n", "number", "sg"),
                                    ("glorply", "r", None, None)]:
            a = lx.analyze(word)[0]
            self.assertFalse(a.known, word)
            self.assertEqual(a.pos, pos, word)
            if key:
                self.assertEqual(a.feats[key], val)

    def test_meaning_queries(self):
        lx = Lexicon()
        self.assertTrue(lx.is_a("zebra", "animal"))
        self.assertTrue(lx.is_a("puppy", "canine"))
        self.assertFalse(lx.is_a("dog", "bird"))
        self.assertTrue(lx.isa_any_pos("blue", "color"))
        self.assertIn("small", lx.antonyms("big"))
        self.assertIn("cold", lx.antonyms("hot"))
        self.assertIn("eight legs", lx.define("spider"))

    def test_runtime_learning(self):
        lx = Lexicon()
        self.assertFalse(lx.knows("flib", "n"))
        lx.learn("flib", "n", kind="animal", hypernyms=["animal"])
        self.assertTrue(lx.is_a("flib", "animal"))
        self.assertTrue(lx.analyze("flibs")[0].known)

    @unittest.skipUnless(os.path.exists(os.path.join(WN, "index.noun")), "WordNet not downloaded")
    def test_wordnet_backend(self):
        lx = Lexicon(wordnet=WN)
        self.assertTrue(lx.is_a("zebra", "animal"))
        self.assertTrue(lx.isa_any_pos("blue", "color"))
        self.assertEqual(lx.analyze("sparrows")[0].lemma, "sparrow")
        self.assertEqual(lx.analyze("legs")[0].lemma, "leg")
        self.assertIn("equine", lx.best_sense("zebra", "n").hypernyms)
        self.assertGreater(len(lx.define("giraffe")), 10)
        self.assertFalse(lx.analyze("blicked")[0].known)


class ComprehensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lx = Lexicon()

    def test_declaratives(self):
        lx = self.lx
        self.assertEqual(props("The sky is blue.", lx), [("sky", "attr", "blue", "")])
        self.assertEqual(props("A zebra is an animal.", lx), [("zebra", "isa", "animal", "")])
        self.assertEqual(props("The dogs eat bones.", lx), [("dog", "eat", "bone", "")])
        self.assertEqual(props("The cow lives in the farm.", lx), [("cow", "live_in", "farm", "")])
        self.assertEqual(props("The bird has two legs.", lx), [("bird", "have", "leg", "two")])
        self.assertEqual(props("There are three cats.", lx), [("cat", "count", "", "three")])
        self.assertEqual(props("The opposite of big is small.", lx), [("big", "antonym", "small", "")])
        self.assertEqual(props("The dog says woof.", lx), [("dog", "say", "woof", "")])
        self.assertEqual(props("The bird can fly.", lx), [("bird", "can", "fly", "")])

    def test_negation_and_adjectives_in_noun_phrases(self):
        toks, p = parse("The dog does not eat the apple.", self.lx)
        pr = p.parse_declarative(toks)
        self.assertTrue(pr[0].neg)
        self.assertEqual(props("Tom has a red ball.", self.lx), [("tom", "have", "ball", "one"), ("ball", "attr", "red", "")])

    def test_pronouns_follow_the_discourse(self):
        d = Discourse()
        props("The boy runs.", self.lx, d)
        self.assertEqual(props("He sees the dog.", self.lx, d)[0][:3], ("boy", "see", "dog"))
        d2 = Discourse()
        props("The zebra is big.", self.lx, d2)
        self.assertEqual(props("It eats grass.", self.lx, d2)[0][:3], ("zebra", "eat", "grass"))

    def test_questions(self):
        lx = self.lx
        cases = {
            "What color is the sky?": ("fact", "sky", "attr"), "What does the dog say?": ("fact", "dog", "say"),
            "How many legs does the bird have?": ("fact", "bird", "have"), "What is a zebra?": ("define", "zebra", ""),
            "What is the opposite of big?": ("opposite", "big", ""), "Is the sky blue?": ("yesno", "sky", "attr"),
            "Is a zebra an animal?": ("yesno", "zebra", "isa"), "What is two plus three?": ("arithmetic", "", ""),
        }
        for q, (kind, subj, rel) in cases.items():
            toks, p = parse(q, lx)
            qq = p.parse_question(toks)
            self.assertIsNotNone(qq, q)
            self.assertEqual((qq.kind, qq.subj, qq.rel), (kind, subj, rel), q)


class WorkspaceTests(unittest.TestCase):
    def ws(self):
        return Workspace(model=None, lexicon=Lexicon())

    def test_store_and_retrieve_exactly(self):
        w = self.ws()
        w.read("the sky is blue . the dog says woof . the cow lives in the farm . the bird has two legs .")
        self.assertEqual(w.ask("what color is the sky ?").text, "the sky is blue .")
        self.assertEqual(w.ask("what does the dog say ?").text, "the dog says woof .")
        self.assertEqual(w.ask("where does the cow live ?").text, "the cow lives in the farm .")
        self.assertEqual(w.ask("how many legs does the bird have ?").text, "the bird has two legs .")

    def test_many_facts_do_not_interfere(self):
        from hebbian_lm.evaluate_workspace import make_facts
        facts = make_facts(20, seed=3)
        w = self.ws()
        for f in facts:
            w.read(f["statement"])
        wrong = [f for f in facts if f["value"] not in w.ask(f["question"]).text.split()]
        self.assertEqual(wrong, [])

    def test_i_do_not_know_and_unknown_word_prompt(self):
        w = self.ws()
        self.assertEqual(w.ask("what color is the sand ?").text, "i do not know .")
        r = w.ask("what color is the glorp ?")
        self.assertEqual(r.kind, "unknown_word")
        self.assertIn("glorp", r.text)

    def test_unknown_word_learned_from_a_definition(self):
        w = self.ws()
        w.read("a flib is an animal .")
        self.assertEqual(w.ask("is a flib an animal ?").text.split()[0], "yes")
        self.assertEqual(w.ask("is a flib a bird ?").text.split()[0], "no")
        self.assertIn("kind of animal", w.ask("what is a flib ?").text)
        w.read("the flib eats grass .")
        self.assertEqual(w.ask("what does the flib eat ?").text, "the flib eats the grass .")

    def test_dictionary_gives_meaning_without_being_told(self):
        w = self.ws()
        self.assertEqual(w.ask("is a zebra an animal ?").text, "yes , the zebra is an animal .")
        self.assertIn("kind of equine", w.ask("what is a zebra ?").text)
        self.assertEqual(w.ask("what is the opposite of big ?").text.split()[-2], "little")
        self.assertEqual(w.ask("is a dog a bird ?").text.split()[0], "no")

    def test_inheritance_through_is_a(self):
        w = self.ws()
        w.read("animals eat food . ")      # generic statement about a kind
        w.read("the zebra is big .")
        r = w.ask("what does the zebra eat ?")
        # nothing is stored about zebras eating; it inherits from "animal"
        self.assertTrue(r.inferred or r.kind == "idk")

    def test_correction_replaces_old_belief(self):
        w = self.ws()
        w.read("the sky is blue .")
        self.assertEqual(w.ask("what color is the sky ?").text, "the sky is blue .")
        w.read("the sky is red .")
        self.assertEqual(w.ask("what color is the sky ?").text, "the sky is red .")
        self.assertEqual(w.ask("is the sky blue ?").text.split()[0], "no")
        self.assertEqual(w.ask("is the sky red ?").text.split()[0], "yes")

    def test_pronoun_story(self):
        w = self.ws()
        w.read("the zebra is brown . it eats the grass . the sparrow is small . it lives in the tree .")
        self.assertEqual(w.ask("what does the zebra eat ?").text, "the zebra eats the grass .")
        self.assertEqual(w.ask("where does the sparrow live ?").text, "the sparrow lives in the tree .")
        self.assertEqual(w.ask("what color is the zebra ?").text, "the zebra is brown .")

    def test_arithmetic_by_rules(self):
        w = self.ws()
        self.assertEqual(w.ask("what is five plus eight ?").text, "five plus eight is thirteen .")
        self.assertEqual(w.ask("what is nine minus four ?").text, "nine minus four is five .")

    def test_grammar_check_and_correction(self):
        w = self.ws()
        cases = {"the dogs runs .": "the dogs run .", "the dog run .": "the dog runs .", "a dogs run .": "a dog runs .",
                 "does the cat runs ?": "does the cat run ?", "the dogs is big .": "the dogs are big .",
                 "an dog runs .": "a dog runs .", "these dog run .": "these dogs run ."}
        for bad, good in cases.items():
            self.assertTrue(w.check(bad), bad)
            self.assertEqual(w.correct(bad), good, bad)
        for fine in ["the dog runs .", "the dogs run .", "she is happy .", "i am happy .", "does the boy eat an apple ?"]:
            self.assertEqual([i for i in w.check(fine) if i.kind != "unknown"], [], fine)

    def test_save_and_reopen(self):
        w = self.ws()
        w.read("a flib is an animal . the flib is green . the sky is blue .")
        with tempfile.TemporaryDirectory() as d:
            w.save(d)
            w2 = Workspace.open(d, lexicon=Lexicon())
        self.assertEqual(w2.ask("what color is the flib ?").text, "the flib is green .")
        self.assertEqual(w2.ask("is a flib an animal ?").text.split()[0], "yes")


class DictionaryInTheLanguageModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lang = Language()
        docs = lang.sample_docs(6000, seed=0)
        cls.lex = Lexicon()
        cls.lm = HebbianLM(Config(), lexicon=cls.lex)
        cls.lm.read_lines(docs)
        cls.lm.consolidate()
        cls.plain = HebbianLM(Config())
        cls.plain.read_lines(docs)
        cls.plain.consolidate()

    def test_new_words_get_the_right_category_before_any_sentence(self):
        m, p = copy.deepcopy(self.lm), copy.deepcopy(self.plain)
        hits_dict = sum(m.seed_category(w) == m.category_of("dog") for w in ["zebra", "sparrow", "kitten", "wolf", "tiger"])
        hits_plain = sum(p.seed_category(w) == p.category_of("dog") for w in ["zebra", "sparrow", "kitten", "wolf", "tiger"])
        self.assertGreaterEqual(hits_dict, 4)
        self.assertLessEqual(hits_plain, 1)
        plural = [m.seed_category(w) == m.category_of("dogs") for w in ["zebras", "wolves", "tigers"]]
        self.assertGreaterEqual(sum(plural), 2)

    def test_deepcopy_keeps_the_dictionary(self):
        self.assertIs(copy.deepcopy(self.lm).lexicon, self.lex)

    def test_workspace_with_language_model(self):
        ws = Workspace(model=copy.deepcopy(self.lm), lexicon=self.lex)
        ws.read("the zebra is brown . the zebra eats grass .")
        r = ws.ask("what color is the zebra ?")
        self.assertEqual(r.text, "the zebra is brown .")
        self.assertIsNotNone(r.fluency)
        self.assertIn("zebra", ws.model.idx)
        # the language layer keeps learning from what was read
        self.assertGreater(ws.model.logprob("the zebra eats grass ."), ws.model.logprob("grass the eats zebra ."))


class EntityTests(unittest.TestCase):
    def ws(self, instances=True):
        return Workspace(model=None, lexicon=Lexicon(), instances=instances)

    def test_two_owners_two_balls(self):
        ws = self.ws()
        ws.read("tom has a red ball . anna has a blue ball .")
        self.assertEqual(len(ws.mem.ents.of("ball")), 2)
        self.assertEqual(ws.ask("what color is tom's ball ?").text, "tom's ball is red .")
        self.assertEqual(ws.ask("what color is anna's ball ?").text, "anna's ball is blue .")
        self.assertEqual(ws.ask("what does tom have ?").text, "tom has a red ball .")

    def test_ambiguous_the_asks_which_and_takes_the_answer(self):
        ws = self.ws()
        ws.read("tom has a red ball . anna has a blue ball .")
        r = ws.converse("what color is the ball ?")
        self.assertEqual(r.kind, "clarify")
        self.assertIn("tom's ball", r.text)
        self.assertEqual(ws.converse("anna's").text, "anna's ball is blue .")
        ws.converse("what color is the ball ?")
        self.assertEqual(ws.converse("the red one").text, "tom's ball is red .")

    def test_single_instance_resolves_and_pronouns_follow_it(self):
        ws = self.ws()
        ws.read("tom has a red ball . it is big .")
        self.assertEqual(ws.ask("what color is the ball ?").text, "the ball is red .")
        self.assertEqual(ws.ask("is the ball big ?").text, "yes , the ball is big .")

    def test_reading_the_definite_picks_the_latest_mentioned(self):
        ws = self.ws()
        ws.read("tom has a red ball . anna has a blue ball . the ball is big .")
        self.assertEqual(ws.ask("is anna's ball big ?").text, "yes , anna's ball is big .")
        self.assertNotEqual(ws.ask("is tom's ball big ?").text.split(" ,")[0], "yes")

    def test_possessive_pronoun(self):
        ws = self.ws()
        ws.read("tom has a dog . his dog is brown .")
        self.assertEqual(ws.ask("what color is tom's dog ?").text, "the dog is brown .")

    def test_telling_the_same_thing_again_does_not_duplicate(self):
        ws = self.ws()
        ws.read("tom has a red ball . tom has a red ball .")
        self.assertEqual(len(ws.mem.ents.of("ball")), 1)
        ws.read("tom has a blue ball .")
        self.assertEqual(ws.ask("how many balls does tom have ?").text, "tom has two balls .")

    def test_generic_statements_stay_at_the_kind_level(self):
        ws = self.ws()
        ws.read("the zebra is brown . a bird has a beak . the dog eats bones .")
        self.assertEqual(len(ws.mem.ents), 0)
        self.assertEqual(ws.ask("what color is the zebra ?").text, "the zebra is brown .")

    def test_instance_inherits_from_its_kind(self):
        ws = self.ws()
        ws.read("the dog barks . tom has a dog .")
        self.assertEqual(ws.ask("is tom's dog an animal ?").text, "yes , the dog is an animal .")

    def test_without_instances_the_old_behaviour_collapses(self):
        ws = self.ws(instances=False)
        ws.read("tom has a red ball . anna has a blue ball .")
        self.assertEqual(len(ws.mem.ents), 0)
        self.assertEqual(ws.ask("what color is the ball ?").kind, "answer")

    def test_instances_survive_save_and_load(self):
        ws = self.ws()
        ws.read("tom has a red ball . anna has a blue ball .")
        with tempfile.TemporaryDirectory() as d:
            ws.save(d)
            w2 = Workspace.open(d, lexicon=Lexicon())
        self.assertEqual(w2.ask("what color is anna's ball ?").text, "anna's ball is blue .")
        self.assertEqual(w2.ask("what color is the ball ?").kind, "clarify")

    def test_possessive_token_and_grammar_check(self):
        toks = tokenize_cased("tom's red ball is big .")
        tag_sentence(toks, Lexicon())
        self.assertEqual([t.tag for t in toks][:4], ["PN", "POSS", "ADJ", "N"])
        ws = self.ws()
        self.assertEqual(ws.check("tom's ball is red ."), [])


class ReaderTests(unittest.TestCase):
    def ws(self, **kw):
        return Workspace(model=None, lexicon=Lexicon(), take_notes=False, **kw)

    def props(self, text):
        return {(p.subj, p.rel, p.obj) for p in self.ws().read(text).props}

    def test_verb_phrase_coordination_shares_the_subject(self):
        p = self.props("Tom sat down and scratched his head .")
        self.assertIn(("tom", "sit", ""), p)
        self.assertIn(("tom", "scratch", "head#1"), p)               # "his head" is tom's own head: an instance

    def test_clauses_after_but_and_because(self):
        p = self.props("The cat ran , but the dog slept because it was tired .")
        self.assertIn(("cat", "run", ""), p)
        self.assertIn(("dog", "sleep", ""), p)

    def test_relative_clause_and_complement_clause(self):
        p = self.props("There was a crash which made everybody turn . Anna knew that the dog barked .")
        self.assertIn(("crash", "make", "everybody"), p)
        self.assertIn(("anna", "know", ""), p)
        self.assertIn(("dog", "bark", ""), p)

    def test_infinitive_and_causative(self):
        p = self.props("Tom wanted to sing . It made him feel queer .")
        self.assertIn(("tom", "want", ""), p)
        self.assertIn(("tom", "sing", ""), p)
        self.assertIn(("it", "make", "tom"), p)                      # "him" is Tom
        self.assertIn(("tom", "feel", "queer"), p)

    def test_passive(self):
        p = self.props("The nut was eaten by the squirrel . The statue was unveiled .")
        self.assertIn(("squirrel", "eat", "nut"), p)
        self.assertIn(("statue", "attr", "unveiled"), p)

    def test_quoted_words_are_not_facts_but_the_speaker_is_noted(self):
        p = self.props('"The moon is made of cheese ," said Anna .')
        self.assertIn(("anna", "say", ""), p)
        self.assertFalse(any(x[0] == "moon" for x in p))

    def test_full_name_and_short_name_are_one_character(self):
        ws = self.ws()
        ws.read("Buster Bear lives in the forest . Buster ate the berries .")
        self.assertEqual({p.subj for p in ws.read("Buster slept .").props}, {"buster bear"})
        self.assertIn("buster bear", ws.ask("what did buster do ?").text)

    def test_pronoun_prefers_the_subject_of_the_last_sentence(self):
        ws = self.ws()
        ws.read("Tom saw Ben . He ran .")
        r = ws.ask("what did tom do ?").text
        self.assertIn("ran", r)
        self.assertIn("saw", r)
        self.assertEqual(ws.ask("what did ben do ?").text, "i do not know .")

    def test_what_did_x_do_summarises(self):
        ws = self.ws()
        ws.read("Anna jumped on the box . Anna ate an apple .")
        r = ws.ask("what did anna do ?").text
        self.assertIn("jumped", r)
        self.assertIn("ate", r)

    def test_real_text_gold_sets(self):
        from hebbian_lm import evaluate_real as er
        lex = Lexicon()
        v1, v2 = er.extractors(lex)
        res = er.report(v2)
        self.assertGreaterEqual(res["dev"]["recall"], 0.6)           # bundled dictionary (390 words)
        self.assertGreaterEqual(res["test"]["recall"], 0.55)
        self.assertGreaterEqual(res["test"]["coverage"], 0.85)
        self.assertGreater(res["test"]["recall"], er.report(v1)["test"]["recall"] + 0.25)

    @unittest.skipUnless(os.path.isdir(WN), "WordNet not downloaded")
    def test_real_text_gold_sets_with_wordnet(self):
        from hebbian_lm import evaluate_real as er
        _, v2 = er.extractors(Lexicon.default())
        res = er.report(v2)
        self.assertGreaterEqual(res["dev"]["recall"], 0.85)
        self.assertGreaterEqual(res["test"]["recall"], 0.75)


class MemoryAndNotesTests(unittest.TestCase):
    def ws(self, **kw):
        return Workspace(model=None, lexicon=Lexicon(), **kw)

    def test_facts_fade_and_rehearsal_slows_it(self):
        ws = self.ws(take_notes=False)
        ws.read("the zebra is brown . the lion lives in the zoo .")
        for _ in range(4):
            ws.sleep(100, replay=False)
            ws.ask("what color is the zebra ?")                    # used again and again
        ws.sleep(700, replay=False)
        self.assertEqual(ws.ask("what color is the zebra ?").text, "the zebra is brown .")
        self.assertEqual(ws.ask("where does the lion live ?").text, "i do not know .")      # never used: gone

    def test_forgotten_fact_is_found_again_in_the_notes(self):
        ws = self.ws()
        ws.read("the zebra is brown .")
        ws.sleep(3000, replay=False)
        self.assertEqual(ws.mem.size(), 0)
        r = ws.ask("what color is the zebra ?")
        self.assertIn("the zebra is brown .", r.text)
        self.assertIn("notes", r.text + r.note)
        self.assertEqual(ws.mem.size(), 1)                             # reading the note again re-learned it

    def test_notes_do_not_invent_answers(self):
        ws = self.ws()
        ws.read("the zebra is brown . the lion eats meat .")
        self.assertEqual(ws.ask("what color is the sparrow ?").text.split(" ")[0:3], ["i", "do", "not"])
        self.assertEqual(ws.ask("what color is the lion ?").text, "i do not know .")

    def test_correction_still_replaces_after_time_passes(self):
        ws = self.ws(take_notes=False)
        ws.read("the sky is blue .")
        ws.sleep(100, replay=False)
        ws.read("the sky is gray .")
        self.assertEqual(ws.ask("what color is the sky ?").text, "the sky is gray .")

    def test_note_keeper_cue_search_and_learning_from_use(self):
        ws = self.ws()
        ws.note("The meeting with the dentist is on Friday .")
        ws.note("Buy milk and bread .")
        hits = ws.recall("when is the dentist meeting ?")
        self.assertEqual(hits[0].text, "The meeting with the dentist is on Friday .")
        self.assertEqual(ws.recall("something about the moon"), [])

    def test_the_same_note_twice_is_one_stronger_note(self):
        ws = self.ws()
        ws.read("the cat sleeps .")
        ws.read("the cat sleeps .")
        self.assertEqual(len(ws.notes), 1)
        self.assertEqual(next(iter(ws.notes.notes.values())).recalls, 1)

    def test_notes_are_selective_not_a_transcript(self):
        ws = self.ws()
        ws.read("the zebra is brown . the lion lives in the zoo .")
        before = len(ws.notes)
        ws.read("the zebra is brown . the lion lives in the zoo . the zebra is brown . the lion lives in the zoo .")
        self.assertEqual(len(ws.notes), before)                    # nothing new, nothing new noted
        self.assertEqual(ws.notes.notes[1].recalls, 2)             # but it was strengthened by the repeats

    def test_long_text_keeps_only_a_share_and_long_sentences_become_gists(self):
        ws = self.ws()
        animals = ["zebra", "sparrow", "kitten", "puppy", "wolf", "tiger", "giraffe", "goat", "fox", "snake", "turtle", "eagle",
                   "squirrel", "chicken", "dolphin", "whale"]
        ws.read(" ".join(f"the {a} says moo ." for a in animals))
        self.assertLessEqual(len(ws.notes), 7)
        ws2 = self.ws()
        long_sentence = "the zebra that lived near the old river with the sparrow and the wolf was brown and ate the grass every morning before anyone woke ."
        ws2.read(long_sentence)
        text = next(iter(ws2.notes.notes.values())).text
        self.assertLess(len(text), len(long_sentence))
        self.assertIn("zebra", text)

    def test_attention_decides_what_is_worth_noting(self):
        ws = self.ws()
        ws.read("the zebra is brown .")
        ws.ask("where does the zebra live ?")
        animals = ["sparrow", "kitten", "puppy", "wolf", "tiger", "giraffe", "goat", "fox", "snake", "turtle", "eagle", "squirrel",
                   "chicken", "dolphin", "whale", "shark"]
        ws.read("the zebra lives in the farm . " + " ".join(f"the {a} says moo ." for a in animals))
        self.assertTrue(any(n.text.startswith("the zebra lives") for n in ws.notes.notes.values()))

    def test_flagged_sentences_are_always_noted(self):
        ws = self.ws()
        ws.read("the cat is black . the cat is black . remember : the key is under the mat .")
        self.assertTrue(any("key" in n.words for n in ws.notes.notes.values()))

    def test_notes_and_clock_survive_save_and_load(self):
        ws = self.ws()
        ws.read("the zebra is brown .")
        ws.sleep(50)
        with tempfile.TemporaryDirectory() as d:
            ws.save(d)
            w2 = Workspace.open(d, lexicon=Lexicon())
        self.assertEqual(len(w2.notes), 1)
        self.assertEqual(w2.mem.clock, 50)
        self.assertEqual(w2.ask("what color is the zebra ?").text, "the zebra is brown .")

    def test_varied_style_keeps_the_content_and_varies_the_wording(self):
        ws = self.ws(style="varied", seed=1)
        ws.read("the zebra is brown .")
        outs = {ws.ask("what color is the zebra ?").text for _ in range(12)}
        self.assertGreater(len(outs), 1)
        self.assertTrue(all("zebra is brown" in o for o in outs))

    def test_faded_memory_is_hedged(self):
        ws = self.ws(style="varied", seed=3, take_notes=False)
        ws.read("the zebra is brown .")
        ws.sleep(330, replay=False)
        outs = [ws.ask("what color is the zebra ?").text for _ in range(1)]
        self.assertTrue(any(w in outs[0] for w in ("think", "believe", "remember", "not sure")), outs)


if __name__ == "__main__":
    unittest.main()
