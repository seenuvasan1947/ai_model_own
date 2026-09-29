#!/usr/bin/env python3
"""
Calming Flute Sleep Music Generator
Physical-modeling-inspired bamboo flute (bansuri-style) synthesis:
  - breath noise filtered through a resonant bandpass (the "air" in the tone)
  - fundamental + weak harmonics (real flutes are near-sinusoidal, not buzzy)
  - slow vibrato (frequency LFO) like a human breath wavering
  - legato portamento between notes (no hard note-to-note jumps)
  - free-flowing, unmetered phrasing over a pentatonic/raga scale
    (minimal cognitive load -> proven most effective for sleep onset)
"""

import numpy as np
from scipy.signal import butter, lfilter
import wave
import time
import os
from datetime import datetime

SAMPLE_RATE = 44100
DURATION_SECONDS = 300  # 5 minutes


def bandpass_noise(duration, center_freq, q, sr=SAMPLE_RATE, amp=1.0):
    """Breath/air noise shaped by a resonant bandpass filter around center_freq."""
    n = int(sr * duration)
    noise = np.random.normal(0, 1, n)
    bw = max(center_freq / q, 20)
    low = max(center_freq - bw / 2, 20) / (sr / 2)
    high = min(center_freq + bw / 2, sr / 2 - 100) / (sr / 2)
    if low >= high:
        low, high = high * 0.5, high
    b, a = butter(2, [low, high], btype='band')
    filtered = lfilter(b, a, noise)
    filtered = filtered / (np.max(np.abs(filtered)) + 1e-9)
    return filtered * amp


def adsr(n, sr, attack, release, sustain_level=1.0):
    env = np.ones(n)
    a = min(int(sr * attack), n // 2)
    r = min(int(sr * release), n - a)
    if a > 0:
        env[:a] = np.linspace(0, sustain_level, a) ** 1.5
    if r > 0:
        env[-r:] *= np.linspace(1, 0, r) ** 1.5
    env[a:n - r] = sustain_level
    return env


def flute_tone(freq_start, freq_end, duration, sr=SAMPLE_RATE,
               vibrato_rate=4.5, vibrato_depth=0.006, breathiness=0.12,
               amp=0.3, attack=0.18, release=0.25):
    """
    One flute note/phrase-segment with portamento from freq_start to freq_end,
    slight vibrato, breathy noise component, and gentle harmonic content.
    """
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Portamento: smooth glide (most of it happens early, like real fingering)
    glide = 1 - np.exp(-t / max(duration * 0.15, 0.02))
    freq = freq_start + (freq_end - freq_start) * glide

    # Vibrato ramps in after the note settles (real players add vibrato after onset)
    vibrato_env = np.clip((t - duration * 0.25) / (duration * 0.3), 0, 1)
    vibrato = 1 + vibrato_depth * vibrato_env * np.sin(2 * np.pi * vibrato_rate * t)
    inst_freq = freq * vibrato

    phase = 2 * np.pi * np.cumsum(inst_freq) / sr

    # Near-sinusoidal tone, essentially no upper harmonics (avoids sharpness)
    tone = np.sin(phase)
    tone += 0.02 * np.sin(2 * phase)

    # Breath noise layer, filtered around the fundamental only (softer, less hiss)
    breath = bandpass_noise(duration, np.mean(freq), q=5.0, sr=sr, amp=1.0)

    signal = tone * (1 - breathiness) + breath * breathiness

    env = adsr(n, sr, attack, release, sustain_level=1.0)
    # subtle amplitude breathing (players naturally swell/fade within a held note)
    swell = 1 + 0.06 * np.sin(2 * np.pi * 0.15 * t)

    return signal * env * swell * amp


def simple_reverb(audio, sr=SAMPLE_RATE, room=2.2, wet=0.35):
    """Lightweight multi-tap reverb for a sense of space (temple/hall)."""
    out = np.copy(audio)
    taps = [(0.031, 0.5), (0.053, 0.35), (0.078, 0.25), (0.11, 0.18), (0.15, 0.12)]
    for delay_s, decay in taps:
        d = int(sr * delay_s)
        delayed = np.zeros_like(audio)
        delayed[d:] = audio[:-d] * decay
        out += delayed * wet
    return out / (np.max(np.abs(out)) + 1e-9)


def pad_drone(freq, duration, sr=SAMPLE_RATE, amp=0.06):
    """Very soft, pure single-tone drone bed under the flute for warmth."""
    t = np.arange(int(sr * duration)) / sr
    d = np.sin(2 * np.pi * freq * t)
    env = np.ones(len(t))
    fade = int(sr * 4)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return d * env * amp


def smooth_lowpass(audio, cutoff_hz, sr=SAMPLE_RATE):
    """Gentle low-pass to remove any residual harshness/sharpness from the mix."""
    b, a = butter(2, cutoff_hz / (sr / 2), btype='low')
    return lfilter(b, a, audio)


def generate_flute_sleep_music(duration=DURATION_SECONDS, sr=SAMPLE_RATE, seed=42):
    print(f"🎶 Composing {duration}s calming flute piece...")
    start = time.time()
    rng = np.random.default_rng(seed)

    # D minor pentatonic (raga-like, evening/calming): D F G A C
    # Dropped a full octave down from before (warm/low register, not piercing)
    base = 146.83  # D3
    ratios = {
        'D4': 1.0, 'F4': 1.189, 'G4': 1.335, 'A4': 1.498, 'C5': 1.782,
        'D5': 2.0, 'C4': 0.891, 'A3': 0.749, 'G3': 0.667,
    }
    scale = {k: base * v for k, v in ratios.items()}
    order = ['A3', 'G3', 'C4', 'D4', 'F4', 'G4', 'A4', 'C5', 'D5']

    total_samples = int(sr * duration)
    music = np.zeros(total_samples)

    # ---- Drone bed: very quiet, single sustained low D (no beating/clash) ----
    print("  • Laying drone bed...")
    drone_note = scale['D4'] / 2  # one octave below the flute's tonic
    drone = pad_drone(drone_note, duration, sr=sr, amp=0.025)
    music[:len(drone)] += drone[:total_samples]

    # ---- Free-flowing flute phrase (unmetered, like an alaap) ----
    print("  • Improvising flute phrase...")
    cursor = 4.0  # let the drone breathe alone for 4s first
    prev_note = 'D4'
    phrase_amp_curve = lambda frac: 0.55 + 0.25 * np.sin(np.pi * frac)  # gentle arc over piece

    while cursor < duration - 3:
        frac = cursor / duration
        # weighted random walk favoring stepwise motion (smoother, less "melody-like")
        idx = order.index(prev_note)
        step = rng.choice([-2, -1, -1, 0, 1, 1, 2], p=[0.08, 0.22, 0.22, 0.06, 0.22, 0.14, 0.06])
        new_idx = int(np.clip(idx + step, 0, len(order) - 1))
        note = order[new_idx]

        note_dur = rng.uniform(2.5, 6.5)
        note_dur = min(note_dur, duration - cursor)
        if note_dur < 1.0:
            break

        amp = phrase_amp_curve(frac) * rng.uniform(0.85, 1.0)
        tone = flute_tone(
            scale[prev_note], scale[note], note_dur, sr=sr,
            vibrato_rate=rng.uniform(3.2, 4.0),
            vibrato_depth=0.003,
            breathiness=rng.uniform(0.05, 0.10),
            amp=amp * 0.32,
            attack=rng.uniform(0.2, 0.35),
            release=rng.uniform(0.4, 0.7),
        )

        start_i = int(sr * cursor)
        end_i = min(start_i + len(tone), total_samples)
        music[start_i:end_i] += tone[:end_i - start_i]

        # small breathing gap between phrase notes (natural breath pause), occasionally
        gap = rng.choice([0.0, 0.0, 0.3, 0.6], p=[0.55, 0.2, 0.15, 0.1])
        cursor += note_dur + gap
        prev_note = note

    # ---- Final long resolving note back to the tonic ----
    print("  • Resolving to tonic...")
    tail_dur = min(8.0, duration - cursor)
    if tail_dur > 1:
        tail = flute_tone(scale[prev_note], scale['D4'], tail_dur, sr=sr,
                           vibrato_depth=0.004, breathiness=0.1, amp=0.18,
                           attack=0.4, release=3.0)
        start_i = int(sr * cursor)
        end_i = min(start_i + len(tail), total_samples)
        music[start_i:end_i] += tail[:end_i - start_i]

    # ---- Smooth out any residual sharpness before reverb ----
    print("  • Smoothing tone (low-pass)...")
    music = smooth_lowpass(music, cutoff_hz=1800, sr=sr)

    # ---- Space / reverb ----
    print("  • Applying hall reverb...")
    music = simple_reverb(music, sr=sr, room=2.2, wet=0.32)

    # ---- Overall fade in/out ----
    fade = int(sr * 3)
    music[:fade] *= np.linspace(0, 1, fade)
    music[-fade:] *= np.linspace(1, 0, fade)

    # ---- Normalize (gentle, keep dynamics — sleep music shouldn't be loud/compressed) ----
    peak = np.max(np.abs(music))
    if peak > 0:
        music = music / peak * 0.85

    print(f"✓ Composed in {time.time() - start:.2f}s")
    return music


def save_wav(audio, filename, sr=SAMPLE_RATE):
    print(f"💾 Saving {filename}...")
    t0 = time.time()
    stereo = np.column_stack([audio, audio])
    # tiny stereo width via micro-delay on right channel
    delay = int(sr * 0.012)
    right = np.zeros_like(audio)
    right[delay:] = audio[:-delay]
    stereo[:, 1] = right * 0.9 + audio * 0.1

    audio_i16 = np.int16(np.clip(stereo, -1, 1) * 32767)
    with wave.open(filename, 'w') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio_i16.tobytes())
    size_mb = os.path.getsize(filename) / (1024 ** 2)
    print(f"✓ Saved {size_mb:.2f}MB in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎋 CALMING FLUTE SLEEP MUSIC GENERATOR")
    print("=" * 60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {DURATION_SECONDS}s ({DURATION_SECONDS/60:.0f} min)")
    print("=" * 60 + "\n")

    t0 = time.time()
    music = generate_flute_sleep_music(DURATION_SECONDS, SAMPLE_RATE)

    os.makedirs("output", exist_ok=True)
    save_wav(music, "output/flute_sleep_music.wav", sr=SAMPLE_RATE)

    total = time.time() - t0
    print("\n" + "=" * 60)
    print(f"✨ DONE in {total:.2f}s ({DURATION_SECONDS/total:.1f}x realtime)")
    print("Output: output/flute_sleep_music.wav")
    print("=" * 60 + "\n")
