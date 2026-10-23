#!/usr/bin/env python3
"""Download Princeton WordNet 3.0 (the plain-text database NLTK distributes) into data/wordnet/.

    python fetch_wordnet.py

Optional: the system works without it (a small bundled dictionary covers the demo world), but with it
the vocabulary engine knows ~150,000 English words: definitions, word kinds, hypernyms, synonyms, antonyms.
WordNet is (c) Princeton University and is distributed under its own permissive licence (see data/wordnet/LICENSE).
"""

import io
import os
import sys
import urllib.request
import zipfile

URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip"


def main(dest: str = "data/wordnet") -> None:
    if os.path.exists(os.path.join(dest, "index.noun")):
        print(f"WordNet already present in {dest}")
        return
    print("downloading", URL)
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            name = info.filename.split("/", 1)[-1]
            if name and not info.is_dir():
                with open(os.path.join(dest, name), "wb") as f:
                    f.write(z.read(info))
    print(f"WordNet unpacked into {dest}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
