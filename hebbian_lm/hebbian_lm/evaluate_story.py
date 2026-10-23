"""Reading a short factual passage and answering questions about it from different angles.

Two passages with hand-written questions. Each question lists the answers that count as right: every keyword of one alternative must
appear in the reply. ``idk_ok`` marks a question the passage does not answer (``i do not know`` is then correct). Scored per question as

* ``right``  - the reply contains the keywords,
* ``idk``    - "i do not know" / "i did not understand" although the passage answers it,
* ``wrong``  - something else (a confident wrong or irrelevant reply).

``sepoy`` was used while building (development). ``fire`` was written before the story work and looked at as an aggregate, then
used to fix general mechanisms, so it is development too. ``titanic`` was written before the later fixes but its failures were then inspected, so it is development too. ``apollo`` was the held-out check: it was run once (9/16, against 8/16 for the code before this stage) and then also used to fix general
problems, so it is development too. ``wright`` was written before those fixes and run once first (9/15 against 5/15), then also used for fixes. ``everest`` was written next and run once (7/14 against 5/14), then used for fixes. ``curie`` was written last, before the Everest fixes, and
is run once at the end: it is the clean held-out result.
"""

from __future__ import annotations

from typing import Callable, Dict, List

SEPOY = dict(
    name="sepoy",
    text=("The Sepoy Mutiny began in 1857. Sepoys were Indian soldiers who served the British East India Company. "
          "The mutiny started in Meerut on 10 May 1857. The soldiers were angry about the new rifle cartridges. "
          "Rumours said that the cartridges were greased with cow fat and pig fat. Hindu soldiers and Muslim soldiers refused to use them. "
          "Mangal Pandey attacked British officers at Barrackpore. The rebels marched to Delhi and made Bahadur Shah Zafar their leader. "
          "Rani Lakshmibai led the rebels in Jhansi. The British took Delhi back in September 1857. The rebellion ended in 1858. "
          "After the mutiny the British Crown took control of India from the Company."),
    questions=[
        ("what were sepoys ?", [["soldier"]]),
        ("who were the sepoys ?", [["soldier"]]),
        ("which company did the sepoys serve ?", [["company"]]),
        ("when did the mutiny begin ?", [["1857"]]),
        ("where did the mutiny start ?", [["meerut"]]),
        ("why were the soldiers angry ?", [["cartridge"]]),
        ("what were the soldiers angry about ?", [["cartridge"]]),
        ("what were the cartridges greased with ?", [["fat"]]),
        ("who refused to use the cartridges ?", [["soldier"]]),
        ("what did Mangal Pandey do ?", [["attack"]]),
        ("who attacked British officers ?", [["pandey"]]),
        ("where did Mangal Pandey attack British officers ?", [["barrackpore"]]),
        ("where did the rebels march ?", [["delhi"]]),
        ("who did the rebels make their leader ?", [["bahadur"]]),
        ("who led the rebels in Jhansi ?", [["lakshmibai"]]),
        ("who was the leader in Delhi ?", [["bahadur"]]),
        ("what did the British take back ?", [["delhi"]]),
        ("when did the British take Delhi back ?", [["september"]]),
        ("when did the rebellion end ?", [["1858"]]),
        ("what happened in 1858 ?", [["end"]]),
        ("what happened after the mutiny ?", [["crown"]]),
        ("who took control of India ?", [["crown"]]),
        ("did the mutiny start in Meerut ?", [["yes"]]),
        ("did the mutiny start in Delhi ?", [["no"]]),
        ("is Mangal Pandey a British officer ?", "idk_ok"),
        ("who is Rani Lakshmibai ?", [["rebel"], ["lakshmibai", "led"]]),
        ("what is the Sepoy Mutiny ?", [["began"], ["rebellion"], ["1857"]]),
    ])

FIRE = dict(
    name="fire",
    text=("The Great Fire of London began on 2 September 1666. It started in a bakery on Pudding Lane. "
          "The baker, Thomas Farriner, had forgotten to put out his oven. Strong winds spread the fire quickly across the wooden houses. "
          "The fire burned for four days. It destroyed about 13000 houses and 87 churches, including St Paul's Cathedral. "
          "Samuel Pepys wrote about the fire in his diary. King Charles ordered people to pull down houses to stop the flames. "
          "After the fire, Christopher Wren designed a new St Paul's Cathedral. Only six people are known to have died."),
    questions=[
        ("when did the fire begin ?", [["1666"], ["september"]]),
        ("where did the fire start ?", [["pudding"], ["bakery"]]),
        ("who was the baker ?", [["farriner"]]),
        ("what did the baker forget ?", [["oven"]]),
        ("what spread the fire ?", [["wind"]]),
        ("how long did the fire burn ?", [["four"]]),
        ("what did the fire destroy ?", [["house"], ["church"]]),
        ("how many churches did the fire destroy ?", [["87"], ["eighty seven"]]),
        ("who wrote about the fire ?", [["pepys"]]),
        ("where did Pepys write about the fire ?", [["diary"]]),
        ("what did King Charles order ?", [["people"], ["pull"]]),
        ("who designed the new cathedral ?", [["wren"]]),
        ("what happened after the fire ?", [["wren"], ["design"]]),
        ("did the fire start in a bakery ?", [["yes"]]),
        ("did the fire start in a church ?", [["no"]]),
        ("did Pepys write a diary ?", [["yes"]]),
        ("who was Thomas Farriner ?", [["baker"]]),
        ("what did Christopher Wren design ?", [["cathedral"]]),
        ("how many people died ?", [["six"]]),
        ("what is Pudding Lane ?", "idk_ok"),
    ])
TITANIC = dict(
    name="titanic",
    text=("The Titanic was a large British passenger ship. It left Southampton on 10 April 1912 on its first voyage to New York. "
          "Captain Edward Smith commanded the ship. On the night of 14 April the Titanic hit an iceberg in the North Atlantic Ocean. "
          "The ship sank in about three hours. More than 1500 people died. The liner Carpathia rescued about 700 survivors. "
          "The disaster led to new safety rules, and ships now carry enough lifeboats for everyone."),
    questions=[
        ("what was the Titanic ?", [["ship"]]),
        ("when did the Titanic leave Southampton ?", [["1912"], ["april"]]),
        ("where did the Titanic go ?", [["new york"]]),
        ("who commanded the ship ?", [["smith"]]),
        ("what did the Titanic hit ?", [["iceberg"]]),
        ("where did the Titanic hit an iceberg ?", [["atlantic"]]),
        ("when did the Titanic hit an iceberg ?", [["april"], ["night"]]),
        ("how long did the ship take to sink ?", [["three"]]),
        ("how many people died ?", [["1500"]]),
        ("who rescued the survivors ?", [["carpathia"]]),
        ("how many survivors did the Carpathia rescue ?", [["700"]]),
        ("what did the disaster lead to ?", [["safety"], ["rule"]]),
        ("did the Titanic leave Southampton ?", [["yes"]]),
        ("did the Titanic hit a rock ?", "idk_ok"),
        ("who was Edward Smith ?", [["captain"]]),
        ("what sank ?", [["ship"], ["titanic"]]),
    ])
APOLLO = dict(
    name="apollo",
    text=("Apollo 11 was the first mission to land people on the Moon. It was launched from Florida on 16 July 1969. "
          "Neil Armstrong, Buzz Aldrin and Michael Collins flew in the spacecraft. The rocket was called Saturn V. "
          "Armstrong and Aldrin landed the lunar module Eagle in the Sea of Tranquility on 20 July. "
          "Collins stayed in orbit around the Moon in the command module. Armstrong stepped onto the surface first and said that it was "
          "one small step for man. The astronauts collected rocks and took photographs. They returned to Earth on 24 July and landed in the "
          "Pacific Ocean. About 600 million people watched the landing on television."),
    questions=[
        ("what was Apollo 11 ?", [["mission"]]),
        ("when was Apollo 11 launched ?", [["1969"], ["july"]]),
        ("where was Apollo 11 launched from ?", [["florida"]]),
        ("who flew in the spacecraft ?", [["armstrong"], ["collins"], ["aldrin"]]),
        ("what was the rocket called ?", [["saturn"]]),
        ("who landed the lunar module ?", [["armstrong"], ["aldrin"]]),
        ("where did Armstrong and Aldrin land ?", [["tranquility"]]),
        ("when did they land on the Moon ?", [["20 july"], ["july"]]),
        ("who stayed in orbit ?", [["collins"]]),
        ("what did the astronauts collect ?", [["rock"]]),
        ("where did the astronauts land when they returned ?", [["pacific"]]),
        ("when did they return to Earth ?", [["24 july"], ["july"]]),
        ("how many people watched the landing ?", [["600"]]),
        ("did Collins land on the Moon ?", "idk_ok"),
        ("who stepped onto the surface first ?", [["armstrong"]]),
        ("what did the astronauts take ?", [["photograph"]]),
    ])
WRIGHT = dict(
    name="wright",
    text=("Orville and Wilbur Wright were two brothers from Ohio. They built and flew the first successful airplane. "
          "On 17 December 1903 Orville made the first flight at Kitty Hawk in North Carolina. The flight lasted twelve seconds and covered 120 feet. "
          "Wilbur flew the same machine later that day for 59 seconds. The brothers had worked as bicycle makers before they became inventors. "
          "They tested gliders on the sand dunes because the winds there were strong. The airplane was called the Wright Flyer. "
          "It had a petrol engine and two propellers. Today the Wright Flyer is displayed in a museum in Washington."),
    questions=[
        ("who were Orville and Wilbur Wright ?", [["brother"]]),
        ("what did the brothers build ?", [["airplane"]]),
        ("when did Orville make the first flight ?", [["1903"], ["december"]]),
        ("where did Orville make the first flight ?", [["kitty"]]),
        ("how long did the flight last ?", [["twelve"]]),
        ("what did the flight cover ?", [["120"]]),
        ("who flew the same machine ?", [["wilbur"]]),
        ("how long did Wilbur fly ?", [["59"]]),
        ("where did the brothers test gliders ?", [["dune"]]),
        ("what was the airplane called ?", [["flyer"]]),
        ("what did the Wright Flyer have ?", [["engine"]]),
        ("where is the Wright Flyer displayed ?", [["museum"]]),
        ("did Wilbur make the first flight ?", "idk_ok"),
        ("did the flight last twelve seconds ?", [["yes"]]),
        ("who built the airplane ?", [["brother"], ["wright"]]),
    ])
EVEREST = dict(
    name="everest",
    text=("Mount Everest is the highest mountain in the world. It lies on the border between Nepal and Tibet. "
          "In 1953 Edmund Hillary from New Zealand and Tenzing Norgay from Nepal climbed to the top. They reached the summit on 29 May. "
          "Hillary and Tenzing carried oxygen tanks because the air is thin at great heights. The climb took about two months. "
          "Colonel John Hunt led the expedition. After the climb Queen Elizabeth gave Hillary a knighthood. "
          "Thousands of people have climbed the mountain since then."),
    questions=[
        ("what is Mount Everest ?", [["mountain"]]),
        ("where does Mount Everest lie ?", [["border"], ["nepal"]]),
        ("who climbed to the top ?", [["hillary"], ["tenzing"]]),
        ("when did they reach the summit ?", [["29 may"], ["may"]]),
        ("who reached the summit ?", [["hillary"], ["tenzing"]]),
        ("who led the expedition ?", [["hunt"]]),
        ("what did Hillary and Tenzing carry ?", [["oxygen"]]),
        ("how long did the climb take ?", [["months"]]),
        ("who gave Hillary a knighthood ?", [["elizabeth"]]),
        ("what did Queen Elizabeth give Hillary ?", [["knighthood"]]),
        ("did Tenzing carry oxygen tanks ?", [["yes"]]),
        ("did Hillary lead the expedition ?", "idk_ok"),
        ("what happened in 1953 ?", [["climb"], ["reach"]]),
        ("what happened after the climb ?", [["elizabeth"], ["knighthood"]]),
    ])
CURIE = dict(
    name="curie",
    text=("Marie Curie was born in Warsaw in 1867. She moved to Paris in 1891 to study physics. There she met Pierre Curie, and they married in 1895. "
          "Together they discovered two new elements, polonium and radium. In 1903 Marie Curie won the Nobel Prize in Physics with her husband. "
          "After Pierre died in a street accident in 1906, she continued their work. She won a second Nobel Prize, in Chemistry, in 1911. "
          "Marie Curie died in 1934."),
    questions=[
        ("where was Marie Curie born ?", [["warsaw"]]),
        ("when was Marie Curie born ?", [["1867"]]),
        ("where did Marie move in 1891 ?", [["paris"]]),
        ("why did Marie move to Paris ?", [["physics"], ["study"]]),
        ("who did Marie meet in Paris ?", [["pierre"]]),
        ("when did they marry ?", [["1895"]]),
        ("what did they discover ?", [["polonium"], ["radium"], ["element"]]),
        ("how many elements did they discover ?", [["two"]]),
        ("what did Marie Curie win in 1903 ?", [["nobel"]]),
        ("when did Pierre die ?", [["1906"]]),
        ("where did Pierre die ?", [["street"], ["accident"]]),
        ("what happened in 1911 ?", [["nobel"], ["second"]]),
        ("what happened after Pierre died ?", [["continue"], ["work"]]),
        ("when did Marie Curie die ?", [["1934"]]),
        ("did Marie Curie win two Nobel Prizes ?", [["yes"]]),
    ])
STORIES = [SEPOY, FIRE, TITANIC, APOLLO, WRIGHT, EVEREST, CURIE]


def judge(reply_text: str, kind: str, spec) -> str:
    t = reply_text.lower()
    saidno = kind in ("idk", "not_understood") or "do not know" in t or "did not understand" in t
    if spec == "idk_ok":
        return "right" if saidno or t.startswith("no") else "wrong"
    if saidno:
        return "idk"
    return "right" if any(all(k in t for k in alt) for alt in spec) else "wrong"


def run(make_workspace: Callable[[], object], stories: List[dict] = None, verbose: bool = False) -> Dict[str, dict]:
    out = {}
    for st in stories or STORIES:
        ws = make_workspace()
        ws.read(st["text"])
        tally = {"right": 0, "idk": 0, "wrong": 0}
        rows = []
        for q, spec in st["questions"]:
            r = ws.ask(q)
            j = judge(r.text, r.kind, spec)
            tally[j] += 1
            rows.append((j, q, r.text))
            if verbose:
                print(f"  [{j:5}] {q:50} -> {r.text}")
        out[st["name"]] = dict(total=len(st["questions"]), **tally, rows=rows)
    return out


# ----------------------------------------------------------------------------------------------- retelling
KEY_FACTS = {
    "sepoy": [["mutiny", "1857"], ["meerut"], ["angry", "cartridge"], ["fat"], ["pandey", "attack"], ["delhi", "march"], ["bahadur"],
              ["lakshmibai"], ["september", "1857"], ["rebellion", "1858"], ["crown"]],
    "fire": [["began", "1666"], ["bakery"], ["oven"], ["wind", "spread"], ["four", "days"], ["13000"], ["87"], ["pepys", "diary"],
             ["charles"], ["wren"], ["six"]],
    "titanic": [["ship"], ["southampton"], ["smith"], ["iceberg"], ["three", "hours"], ["carpathia"], ["safety"], ["lifeboat"], ["1500"]],
    "curie": [["born", "warsaw"], ["paris"], ["pierre"], ["1895"], ["radium"], ["1903"], ["1906"], ["1934"]],
    "everest": [["highest", "mountain"], ["nepal"], ["hillary", "tenzing"], ["29 may"], ["hunt"], ["oxygen"], ["elizabeth"]],
    "wright": [["brother"], ["airplane"], ["1903"], ["kitty"], ["twelve"], ["wilbur", "59"], ["glider"], ["flyer"], ["engine"], ["museum"]],
    "apollo": [["first", "moon"], ["florida"], ["armstrong"], ["saturn"], ["tranquility"], ["collins", "orbit"], ["rock"], ["pacific"], ["600"]],
}


def retell_report(make_workspace: Callable[[], object], stories: List[dict] = None, limit: int = 16) -> Dict[str, dict]:
    """Retell each passage from the stored facts and measure: how many of the key facts appear (all their keywords in one sentence),
    how many sentences contain a content word that is not in the passage (invented), how many sentences are word for word a sentence of
    the passage (copied), and how many grammar problems the rule checker finds (names unknown to the dictionary are ignored)."""
    import re
    out = {}
    for st in stories or STORIES:
        ws = make_workspace()
        ws.read(st["text"])
        text = ws.retell("", limit=limit)
        sents = [x.strip() for x in text.split(" . ") if x.strip()]
        sents = [x[:-2] if x.endswith(" .") else x for x in sents]
        src = " ".join(re.findall(r"[a-z0-9']+", st["text"].lower()))
        src_lemmas = {ws.lex.analyze(w)[0].lemma for w in src.split()}
        keys = KEY_FACTS.get(st["name"], [])
        hit = sum(any(all(k in s for k in kw) for s in sents) for kw in keys)
        invented = copied = 0
        for s in sents:
            words = [w for w in re.findall(r"[a-z0-9']+", s) if w not in {"the", "a", "an", "was", "were", "is", "are", "by", "it", "they", "he", "she", "to", "in", "of", "on", "at", "and"}]
            if any(ws.lex.analyze(w)[0].lemma not in src_lemmas and w not in src for w in words):
                invented += 1
            if s in src:
                copied += 1
        issues = [i for i in ws.check(text) if i.kind != "unknown"]
        out[st["name"]] = dict(sentences=len(sents), key_facts=f"{hit}/{len(keys)}", invented_sentences=invented, copied_word_for_word=copied,
                               grammar_issues=len(issues), text=text)
    return out
