#!/usr/bin/env python3
"""
Calming Piano Sleep Music Generator
Additive-synthesis piano:
  - slightly inharmonic partials with per-partial decay (higher = faster fade)
  - soft hammer transient, low velocity (gentle touch)
  - slow, sparse D minor pentatonic phrases with long pedal-like reverb tails
  - occasional low-octave anchor notes, arcing dynamics, fade in/out
"""

import numpy as np
from scipy.signal import butter, lfilter
import wave
import time
import os
from datetime import datetime

SAMPLE_RATE = 44100
DURATION_SECONDS = 300  # 5 minutes


def piano_note(freq, duration, sr=SAMPLE_RATE, velocity=0.5):
    """One piano note: decaying inharmonic partials plus a soft hammer thump."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    B = 0.0004  # inharmonicity coefficient
    sig = np.zeros(n)
    for k in range(1, 9):
        f = freq * k * np.sqrt(1 + B * k * k)
        if f > 5000:
            break
        amp = (1.0 / k ** 1.3) * (velocity ** (0.6 * (k - 1) + 0.4))
        decay = 0.9 + 0.7 * k + 0.4 * (freq / 261.0)  # 1/s, higher partials die first
        sig += amp * np.sin(2 * np.pi * f * t + np.random.uniform(0, 0.4)) * np.exp(-decay * t / 2.2)
    # soft hammer: brief lowpassed noise burst
    hl = int(sr * 0.03)
    burst = np.random.normal(0, 1, hl) * np.linspace(1, 0, hl) ** 2
    b, a = butter(2, 900 / (sr / 2), btype='low')
    sig[:hl] += lfilter(b, a, burst) * 0.05 * velocity
    # de-click attack and release
    a_n = int(sr * 0.004)
    sig[:a_n] *= np.linspace(0, 1, a_n)
    r_n = min(int(sr * 0.3), n)
    sig[-r_n:] *= np.linspace(1, 0, r_n)
    return sig


def simple_reverb(audio, sr=SAMPLE_RATE, wet=0.4):
    """Multi-tap reverb, long tail like a sustain pedal in a hall."""
    out = np.copy(audio)
    taps = [(0.037, 0.5), (0.061, 0.4), (0.089, 0.32), (0.13, 0.25),
            (0.19, 0.2), (0.27, 0.15), (0.37, 0.1), (0.5, 0.07)]
    for delay_s, decay in taps:
        d = int(sr * delay_s)
        delayed = np.zeros_like(audio)
        delayed[d:] = audio[:-d] * decay
        out += delayed * wet
    return out / (np.max(np.abs(out)) + 1e-9)


def smooth_lowpass(audio, cutoff_hz, sr=SAMPLE_RATE):
    b, a = butter(2, cutoff_hz / (sr / 2), btype='low')
    return lfilter(b, a, audio)


def generate_piano_sleep_music(duration=DURATION_SECONDS, sr=SAMPLE_RATE, seed=7):
    print(f"🎹 Composing {duration}s calming piano piece...")
    start = time.time()
    rng = np.random.default_rng(seed)

    # D minor pentatonic: D F G A C (equal temperament)
    def midi(m):
        return 440.0 * 2 ** ((m - 69) / 12)

    # D3 F3 G3 A3 C4 D4 F4 G4 A4 C5 D5
    order = [50, 53, 55, 57, 60, 62, 65, 67, 69, 72, 74]
    total = int(sr * duration)
    music = np.zeros(total)

    def place(sig, at):
        s = int(sr * at)
        e = min(s + len(sig), total)
        if e > s:
            music[s:e] += sig[:e - s]

    idx = 5  # D4
    cursor = 1.5
    print("  • Playing melody...")
    while cursor < duration - 12:
        frac = cursor / duration
        dyn = 0.35 + 0.2 * np.sin(np.pi * frac)  # gentle arc

        # every so often, a deep anchor note (D2 / A2) with a soft chord bloom
        if rng.random() < 0.18:
            root = rng.choice([38, 45])
            place(piano_note(midi(root), 9.0, sr, velocity=dyn * 0.9), cursor)

        step = rng.choice([-2, -1, 0, 1, 2], p=[0.12, 0.33, 0.1, 0.33, 0.12])
        idx = int(np.clip(idx + step, 0, len(order) - 1))
        vel = dyn * rng.uniform(0.8, 1.0)
        place(piano_note(midi(order[idx]), 7.0, sr, velocity=vel), cursor)

        # occasional soft 5th/octave above, sparse and consonant
        if rng.random() < 0.2 and idx + 3 < len(order):
            place(piano_note(midi(order[idx + 3]), 6.0, sr, velocity=vel * 0.6), cursor + 0.05)

        cursor += rng.choice([2.0, 2.5, 3.0, 3.5, 4.5, 6.0], p=[0.15, 0.25, 0.25, 0.15, 0.12, 0.08])

    # closing resolution: D minor triad-ish, spread wide
    print("  • Resolving to tonic...")
    for m, off, v in [(38, 0.0, 0.4), (50, 0.4, 0.35), (57, 0.8, 0.3), (62, 1.3, 0.28)]:
        place(piano_note(midi(m), 10.0, sr, velocity=v), cursor + off)

    music = smooth_lowpass(music, 3500, sr=sr)
    print("  • Applying hall reverb...")
    music = simple_reverb(music, sr=sr, wet=0.4)

    fade = int(sr * 3)
    music[:fade] *= np.linspace(0, 1, fade)
    music[-fade:] *= np.linspace(1, 0, fade)
    music = music / (np.max(np.abs(music)) + 1e-9) * 0.85

    print(f"✓ Composed in {time.time() - start:.2f}s")
    return music


def save_wav(audio, filename, sr=SAMPLE_RATE):
    print(f"💾 Saving {filename}...")
    delay = int(sr * 0.012)
    right = np.zeros_like(audio)
    right[delay:] = audio[:-delay]
    stereo = np.column_stack([audio, right * 0.9 + audio * 0.1])
    audio_i16 = np.int16(np.clip(stereo, -1, 1) * 32767)
    with wave.open(filename, 'w') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio_i16.tobytes())
    print(f"✓ Saved {os.path.getsize(filename) / (1024 ** 2):.2f}MB")


if __name__ == "__main__":
    print(f"\n🎹 CALMING PIANO SLEEP MUSIC GENERATOR  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    music = generate_piano_sleep_music()
    os.makedirs("output", exist_ok=True)
    save_wav(music, "output/piano_sleep_music.wav")
    print("Output: output/piano_sleep_music.wav")
