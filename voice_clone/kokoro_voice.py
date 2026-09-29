import argparse
import gc
import re
import time

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

SAMPLE_RATE = 24000
# American English female voices (af_ = American female); af_heart is the warmest, most natural one
VOICES = {"af_heart": "warm and sweet", "af_bella": "bright and friendly", "af_nicole": "soft and calm (default)",
          "af_sky": "light and gentle", "af_sarah": "clear and neutral", "bf_emma": "British, calm"}

parser = argparse.ArgumentParser()
parser.add_argument("--voice", default="af_nicole", help=f"one of: {', '.join(VOICES)} (or any other Kokoro voice id)")
parser.add_argument("--list-voices", action="store_true")
parser.add_argument("--speed", type=float, default=0.95, help="1.0 = normal, slightly below sounds softer")
parser.add_argument("--text", help="narrate this text instead of the file")
parser.add_argument("--file", default="upsc_text.txt", help="text file to narrate (use --text for a short line instead)")
parser.add_argument("--out", default="upsc_female.wav")
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
args = parser.parse_args()

if args.list_voices:
    for v, d in VOICES.items():
        print(f"{v:10s} {d}")
    raise SystemExit(0)

t_start = time.time()


def log(step, msg):
    print(f"[{time.time() - t_start:6.1f}s] {step}: {msg}", flush=True)


text = args.text if args.text else open(args.file, encoding="utf-8").read()
paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
lang = "b" if args.voice.startswith("b") else "a"


def run(device):
    log("1/4 setup", f"device={device}, voice={args.voice}, speed={args.speed}" + (f", gpu={torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    log("2/4 model", "loading Kokoro-82M (first run downloads about 350 MB)")
    t = time.time()
    pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=device)
    log("2/4 model", f"loaded in {time.time() - t:.1f}s")

    log("3/4 generate", f"{len(paragraphs)} paragraph(s)")
    pieces, pause = [], np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
    for i, para in enumerate(paragraphs, 1):
        t = time.time()
        # Kokoro splits long text into natural sentence groups on its own
        audio = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(para, voice=args.voice, speed=args.speed)]
        wav = np.concatenate(audio)
        pieces += [wav, pause]
        dur, took = len(wav) / SAMPLE_RATE, time.time() - t
        log("3/4 generate", f"{i}/{len(paragraphs)} -> {dur:.1f}s audio in {took:.1f}s (speed {dur / max(took, 1e-6):.1f}x realtime)")
    return np.concatenate(pieces[:-1])


attempts = ["cpu"] if args.device == "cpu" or not torch.cuda.is_available() else ["cuda", "cpu"]
for n, device in enumerate(attempts):
    try:
        audio = run(device)
        break
    except Exception as e:
        log("error", f"{device} failed: {type(e).__name__}: {str(e)[:200]}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if n == len(attempts) - 1:
            raise
        log("fallback", "trying cpu")

sf.write(args.out, audio, SAMPLE_RATE)
log("4/4 save", f"{args.out} ({len(audio) / SAMPLE_RATE:.1f}s audio)")
log("done", f"total {time.time() - t_start:.1f}s")
