# Release 1.0.0 — the trained workspace

`workspace/` is the output of `python train.py` (seed 0, 30,000 synthetic child-English sentences, bundled dictionary) with the
code of this release. It contains the semantic memory (`memory.json`, 68 background facts), the dictionary additions, the notes
(`notes.json`), and the trained Hebbian language layer (`language_model.pkl`, 12 MB, a Python pickle).

> Only load `language_model.pkl` from a source you trust: pickle files can run code when loaded. This one was produced by
> `train.py` in this repository; `SHA256SUMS` lets you check it was not altered.

## Use it without training

```bash
git clone https://github.com/seenuvasan1947/ai_model_own
cd ai_model_own && git checkout v1.0.0        # or the hebbian-v2 branch
cd hebbian_lm
pip install -r requirements.txt               # numpy only
sha256sum -c release/v1.0/SHA256SUMS --ignore-missing 2>/dev/null || (cd release/v1.0 && sha256sum -c SHA256SUMS)
python chat.py --workspace release/v1.0/workspace
```

```python
from hebbian_lm.lexicon import Lexicon
from hebbian_lm.workspace import Workspace
ws = Workspace.open("release/v1.0/workspace", lexicon=Lexicon())     # Lexicon.default() uses WordNet if you downloaded it
ws.read("The Sepoy Mutiny began in 1857. Mangal Pandey attacked British officers at Barrackpore.")
print(ws.ask("what did Mangal Pandey do ?").text)
ws.sleep(100)                      # time passes: unused facts fade, notes remain
ws.save("my_workspace")            # keep what it learned
```

Do not edit the files in `release/v1.0/`; `chat.py` saves to a different directory with `/save my_workspace`.
Full instructions: `../../USAGE.md`. What is in this version and what is not: `../../V2.md` and `../../RELEASE.md`.
