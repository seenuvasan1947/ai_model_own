import time
import numpy as np
import sounddevice as sd
import soundfile as sf

# Settings for voice cloning: mono, 24 kHz
samplerate = 24000
channels = 1
duration = 40  # seconds; the script below takes about 30 s to read
filename = "my_voice_ref.wav"

# Read this EXACT text. Use the same string as ref_transcript in qwen_clone.py
script = (
    'Hello, my name is Seenuvasan, and this is my voice. I am recording a short sample so that a computer can learn how I speak. I will talk slowly and clearly, in a calm and natural way, just like I would when explaining something to a friend. Good technology should be simple and useful, and it should help people save time. If this recording sounds clean, the result should sound very close to me.'
)

# sd.default.device = None  # set an input device index here if needed (see sd.query_devices())
print(sd.query_devices(kind="input")["name"], "<- input device\n")
print("Read this at a natural pace, 15 cm from the mic, in a quiet room:\n")
print(script, "\n")

for i in (3, 2, 1):
    print(f"Starting in {i}...")
    time.sleep(1)

print(f"RECORDING for {duration} s. Speak now, then stay quiet when done.")
audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="float32")
sd.wait()
audio = audio[:, 0]

# Trim leading/trailing silence, keeping a short margin
loud = np.where(np.abs(audio) > 0.02)[0]
if len(loud) == 0:
    raise SystemExit("No speech detected. Check your microphone and try again.")
margin = int(0.3 * samplerate)
audio = audio[max(loud[0] - margin, 0): loud[-1] + margin]

peak = float(np.abs(audio).max())
print(f"\nLength: {len(audio) / samplerate:.1f} s, peak level: {peak:.2f}")
if peak >= 0.99:
    print("WARNING: clipping. Move back from the mic or lower the input volume, then re-record.")
elif peak < 0.1:
    print("WARNING: too quiet. Move closer or raise the input volume, then re-record.")

sf.write(filename, audio, samplerate)
print(f"Saved as {filename}")
