# Kokoro voices and languages

Script: `kokoro_voice.py` (model: Kokoro-82M, plain text-to-speech, no cloning).
Chosen voice: **`af_heart`** (American female, the default in the script).

## Where to find the voice list

- Model page: https://huggingface.co/hexgrad/Kokoro-82M
- All voice files: the `voices/` folder in that repo (each voice is one `.pt` file, the file name is the voice id).
- Voice list from Python (needs internet):
  ```python
  from huggingface_hub import list_repo_files
  print(sorted(f[7:-3] for f in list_repo_files("hexgrad/Kokoro-82M") if f.startswith("voices/")))
  ```

## Voice id naming

`<language letter><gender letter>_<name>`, for example `af_heart` = **a**merican **f**emale "heart".

| Prefix | Language | Female voices |
|---|---|---|
| `af_` | American English | alloy, aoede, bella, **heart**, jessica, kore, nicole, nova, river, sarah, sky |
| `bf_` | British English | alice, emma, isabella, lily |
| `hf_` | Hindi | alpha, beta |
| `ef_` / `ff_` / `if_` / `pf_` | Spanish / French / Italian / Portuguese | dora / siwis / sara / dora |
| `jf_` / `zf_` | Japanese / Chinese | several |

Male voices use `am_`, `bm_`, etc. Tamil is not available.

## Language code

`kokoro_voice.py` picks the language from the voice id: `bf_`/`bm_` use British English (`b`), everything else uses American English (`a`).
Other codes in Kokoro: `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese, `p` Portuguese, `z` Chinese.
To use a non-English voice the script would need a change (the `lang` line), plus text in that language.

## How to use

```
cd ~/Documents/own_projects/ai_project/voice_clone
../venv/bin/python kokoro_voice.py --file upsc_text.txt --out upsc_female.wav
../venv/bin/python kokoro_voice.py --file upsc_text.txt --voice af_nicole --out upsc_nicole.wav
../venv/bin/python kokoro_voice.py --file upsc_text.txt --speed 0.9
```

- `--voice` picks the voice, `--speed` sets the pace (default 0.95, 1.0 is normal).
- Blank lines in the text add a 0.35 s pause. Commas and full stops give natural pauses inside a paragraph.

## Notes

- The voice descriptions in the script (warm, soft and so on) are approximate. Listen to decide.
- Keep the GPU free: do not run this together with the image or Qwen scripts (4 GB VRAM).
- Two voices in this project: `qwen_clone.py` for your own cloned voice, `kokoro_voice.py` for the female narration.
