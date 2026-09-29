import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import list_repo_files
from kokoro import KPipeline

SAMPLE_RATE = 24000
# About 10 seconds of speech, with commas and full stops like your real script
SAMPLE_TEXT = ("Welcome back to the channel. Today we are talking about the UPSC Civil Services Examination, "
               "one of the toughest and most respected exams in India.")

parser = argparse.ArgumentParser()
parser.add_argument("--speeds", default="0.9,1.0", help="comma-separated speed variants")
parser.add_argument("--gender", choices=["f", "m", "both"], default="f", help="f = female voices (default)")
parser.add_argument("--voices", help="comma-separated voice ids, instead of all English voices")
parser.add_argument("--text", default=SAMPLE_TEXT)
parser.add_argument("--out", default="voice_samples")
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
args = parser.parse_args()

t_start = time.time()


def log(step, msg):
    print(f"[{time.time() - t_start:6.1f}s] {step}: {msg}", flush=True)


device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
if args.voices:
    voices = [v.strip() for v in args.voices.split(",")]
else:
    files = list_repo_files("hexgrad/Kokoro-82M")
    voices = sorted(f[7:-3] for f in files if f.startswith("voices/") and f[7] in "ab"  # American and British English
                    and (args.gender == "both" or f[8] == args.gender))
speeds = [float(s) for s in args.speeds.split(",")]
os.makedirs(args.out, exist_ok=True)
log("1/3 setup", f"device={device}, {len(voices)} voice(s) x {len(speeds)} speed(s) = {len(voices) * len(speeds)} file(s)")

log("2/3 model", "loading Kokoro-82M")
pipes = {}
for lang in sorted({v[0] for v in voices}):
    pipes[lang] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)

log("3/3 generate", f"saving to {args.out}/")
for voice in voices:
    for speed in speeds:
        t = time.time()
        parts = [np.asarray(a, dtype=np.float32) for _, _, a in pipes[voice[0]](args.text, voice=voice, speed=speed)]
        wav = np.concatenate(parts)
        name = f"{args.out}/{voice}_speed{speed:.2f}.wav"
        sf.write(name, wav, SAMPLE_RATE)
        log("3/3 generate", f"{name} ({len(wav) / SAMPLE_RATE:.1f}s audio in {time.time() - t:.1f}s)")
log("done", f"total {time.time() - t_start:.1f}s")
