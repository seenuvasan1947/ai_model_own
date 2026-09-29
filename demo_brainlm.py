"""Checks the plan_current success criteria for the text agent."""
import shutil
import tempfile

from brainlm import BrainAgent

d = tempfile.mkdtemp()
a = BrainAgent(d)
a.learn("The capital of France is Paris. Mitochondria produce energy for the cell. "
        "Python is a programming language created by Guido van Rossum. "
        "The capital of Australia is Canberra. Water boils at 100 degrees Celsius.")
a.learn("Sydney is the largest city in Australia and a popular tourist destination.")

q = "What is the capital of Australia?"
first = a.ask(q)
print("1. recall        :", first.text)

# repetition reinforces instead of duplicating
n = len(a.store); a.learn("The capital of France is Paris.")
print("2. no duplicate  :", len(a.store) == n)

# teach a wrong belief repeatedly so it wins recall, then correct it
for _ in range(6):
    a.learn("The capital of Australia is Sydney.")
bad = a.ask(q)
print("3. before fix    :", bad.text)
err = a.correct("The capital of Australia is Canberra.", q)
print(f"4. error signal  : {err:.2f}")
print("5. after fix     :", a.ask(q).text)
print("6. paraphrase    :", a.ask("Which city is Australia's capital?").text)

# persistence
a.save(); b = BrainAgent(d)
print("7. after restart :", b.ask(q).text, "| memories:", len(b.store))
print("8. unknown       :", b.ask("Who won the 2050 chess olympiad?").text)
shutil.rmtree(d)
