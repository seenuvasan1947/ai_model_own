"""hash vs minilm on the SAME paragraph: questions that reuse its words,
questions with DIFFERENT words (same meaning), and questions it should refuse."""
import shutil, sys, tempfile, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import TitanAgent, MIN_SIM

PARA = open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/parl_full.txt").read()
KEY = {  # question -> a phrase the right sentence contains
    "same words": [
        ("what happens if parliament passes a vote of no confidence?", "vote of no confidence"),
        ("who forms the government?", "forms the government"),
        ("what does parliament do?", "makes laws"),
        ("which countries are examples of parliamentary democracies?", "Canada"),
        ("who is the head of state?", "ceremonial"),
    ],
    "different words": [
        ("How can lawmakers force a leader out?", "vote of no confidence"),
        ("Who picks the people that run the country?", "Voters elect"),
        ("Name some nations that rule in that manner?", "Canada"),
        ("What does the assembly do about spending plans?", "approves the budget"),
        ("How is a setup with a directly chosen chief different?", "presidential system"),
        ("Who performs only symbolic roles?", "ceremonial"),
        ("What ends a leader's time in office when lawmakers lose faith in them?", "vote of no confidence"),
    ],
    "should refuse": [
        ("What is the speed of light?", None),
        ("How do I bake a chocolate cake?", None),
        ("Who won the football world cup in 2010?", None),
    ],
}
for enc in ("hash", "minilm"):
    tmp = tempfile.mkdtemp(); ag = TitanAgent(tmp, encoder=enc)
    ag.learn_text(PARA)
    print(f"\n=== encoder: {enc} ===")
    for group, qs in KEY.items():
        ok = 0
        for q, want in qs:
            ranked, _ = ag.recall(q)
            top, sim = ranked[0] if ranked else ("", 0.0)
            ans = ag.answer(q)["text"]
            answered = ans != "I don't know."
            good = (want in ans) if want else (not answered)
            ok += good
            print(f"  {'OK ' if good else 'BAD'} [{group}] sim {sim:.2f} {'(answered)' if answered else '(refused) '} {q}")
        print(f"  -> {group}: {ok}/{len(qs)}")
    shutil.rmtree(tmp)
