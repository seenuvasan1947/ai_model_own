import argparse
import gc
import re
import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Path to your recorded voice sample and the exact words spoken in that sample
ref_audio_path = "my_voice_ref.wav"
ref_transcript = 'Hello, my name is Seenuvasan, and this is my voice. I am recording a short sample so that a computer can learn how I speak. I will talk slowly and clearly, in a calm and natural way, just like I would when explaining something to a friend. Good technology should be simple and useful, and it should help people save time.'

# Text you want your video pipeline to speak
default_text = "Welcome back to the channel. Today we are testing our local AI voice clone pipeline."

parser = argparse.ArgumentParser()
parser.add_argument("--text", default=default_text)
parser.add_argument("--batch", type=int, default=4, help="sentences generated at once (halved automatically on out-of-memory)")
parser.add_argument("--chunk-chars", type=int, default=220, help="merge sentences into chunks up to this many characters (fewer, smoother pieces)")
parser.add_argument("--file", help="read the text from this file instead of --text")
parser.add_argument("--out", default="qwen_output_voice.wav")
parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
parser.add_argument("--dtype", choices=["float32", "bfloat16"], default=None,
                    help="default: bfloat16 on GPU (then float32, then CPU as fallbacks), float32 on CPU")
args = parser.parse_args()

t_start = time.time()


def log(step, msg):
    print(f"[{time.time() - t_start:6.1f}s] {step}: {msg}", flush=True)


text = open(args.file, encoding="utf-8").read() if args.file else args.text
# Merge neighbouring sentences into longer chunks so the model keeps natural flow between them
sentences, cur = [], ""
for s_ in re.split(r"(?<=[.!?])\s+", text.strip()):
    if cur and len(cur) + 1 + len(s_) > args.chunk_chars:
        sentences.append(cur)
        cur = s_
    else:
        cur = f"{cur} {s_}".strip()
if cur:
    sentences.append(cur)


def run(device, dtype_name):
    """Load model, encode voice and generate all sentences with one device/dtype setting."""
    log("1/5 setup", f"device={device}, dtype={dtype_name}" + (f", gpu={torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    log("2/5 model", f"loading {MODEL_ID} (first run downloads it)")
    t = time.time()
    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device, dtype=getattr(torch, dtype_name))
    log("2/5 model", f"loaded in {time.time() - t:.1f}s")

    # Encode the reference voice once and reuse it for every sentence
    log("3/5 voice", f"encoding reference '{ref_audio_path}'")
    t = time.time()
    prompt = model.create_voice_clone_prompt(ref_audio=ref_audio_path, ref_text=ref_transcript)
    log("3/5 voice", f"done in {time.time() - t:.1f}s")

    log("4/5 generate", f"{len(sentences)} sentence(s), batch size {args.batch}")
    chunks, sr = [], None
    bs, i = args.batch, 0
    while i < len(sentences):
        group = sentences[i:i + bs]
        t = time.time()
        try:
            wavs, sr = model.generate_voice_clone(text=group, language="English", voice_clone_prompt=prompt)
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            gc.collect()
            torch.cuda.empty_cache()
            log("4/5 generate", f"out of GPU memory, batch size reduced to {bs}")
            continue
        chunks.extend(wavs)
        dur = sum(len(w) for w in wavs) / sr
        took = time.time() - t
        i += len(group)
        log("4/5 generate", f"{i}/{len(sentences)} done -> {dur:.1f}s audio in {took:.1f}s (speed {dur / took:.2f}x realtime)")
    return chunks, sr


# Try the fastest setting first and fall back automatically if it crashes or runs out of memory
if args.device == "cpu":
    attempts = [("cpu", args.dtype or "float32")]
elif args.device == "cuda" or torch.cuda.is_available():
    attempts = [("cuda", args.dtype or "bfloat16"), ("cuda", "float32"), ("cpu", "float32")]
else:
    attempts = [("cpu", args.dtype or "float32")]
attempts = list(dict.fromkeys(attempts))

for n, (device, dtype_name) in enumerate(attempts):
    try:
        chunks, sr = run(device, dtype_name)
        break
    except Exception as e:
        log("error", f"{device}/{dtype_name} failed: {type(e).__name__}: {str(e)[:200]}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if n == len(attempts) - 1:
            raise
        log("fallback", f"trying {attempts[n + 1][0]}/{attempts[n + 1][1]}")

# Join sentences with a very short pause
gap = np.zeros(int(0.15 * sr), dtype=chunks[0].dtype)
audio = np.concatenate([x for c in chunks for x in (c, gap)][:-1])
sf.write(args.out, audio, sr)
log("5/5 save", f"{args.out} ({len(audio) / sr:.1f}s audio)")
log("done", f"total {time.time() - t_start:.1f}s")
