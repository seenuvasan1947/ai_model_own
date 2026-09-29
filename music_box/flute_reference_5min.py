#!/usr/bin/env python3
"""5-min flute piece modeled on the analyzed reference (30s-120s clip):
drone under everything, ~6.5s phrase cycle, 1-3s breath gaps, signature
rise from ~400-550Hz to a held ~800Hz (G5), lower wandering phrases that
end in a downward slide, slow swell/fade on held notes."""
import numpy as np, wave, random
from flute_sleep_music import flute_tone, pad_drone, simple_reverb, SAMPLE_RATE as SR

random.seed(7); np.random.seed(7)
TOTAL = 300
N = np.zeros(int(SR * TOTAL) + SR * 10)

def put(sig, start):
    i = int(start * SR); N[i:i + len(sig)] += sig[:len(N) - i]

def note(f0, f1, d, amp=0.3, **k):
    return flute_tone(f0, f1, d, amp=amp, **k)

def phrase_rise(t):
    """signature: slide up from low note to held G5, then release."""
    lo = random.choice([392, 440, 494, 523])
    hi = random.choice([784, 800, 830])
    t0 = t
    put(note(lo, lo, random.uniform(.6, 1.0), attack=.2), t); t += random.uniform(.7, 1.0)
    put(note(lo, hi, random.uniform(1.0, 1.6), amp=.32, attack=.1, release=.05), t); t += random.uniform(1.0, 1.6)
    hold = random.uniform(2.5, 4.0)
    put(note(hi, hi, hold, amp=.34, attack=.3, release=1.0, vibrato_depth=.004), t); t += hold
    return t - t0

def phrase_wander(t):
    """shorter notes 520-660Hz, ends with downward slide to ~400Hz."""
    t0 = t
    pool = [523, 587, 659, 587, 523, 494]
    for _ in range(random.randint(3, 5)):
        f = random.choice(pool); d = random.uniform(.6, 1.4)
        put(note(f, f * random.choice([1, 1, 1.03, .97]), d, amp=.26, attack=.15), t); t += d
    f = random.choice([523, 587])
    d = random.uniform(1.5, 2.2)
    put(note(f, random.choice([392, 415, 440]), d, amp=.28, attack=.15, release=.6), t); t += d
    return t - t0

# drone: C3 + G3 (+ octave), constant like a tanpura
for f, a in [(130.8, .07), (196.0, .05), (261.6, .03)]:
    N[:int(SR * TOTAL)] += pad_drone(f, TOTAL, amp=a)[:int(SR * TOTAL)]

t = 1.0
order = ['w', 'r', 'w', 'w', 'r', 'r', 'w']   # reference: mix, rises recurring
i = 0
while t < TOTAL - 12:
    kind = order[i % len(order)]; i += 1
    t += (phrase_rise if kind == 'r' else phrase_wander)(t)
    t += random.uniform(1.5, 3.0)   # breath gap; drone continues

# gentle overall fade in/out
env = np.ones(len(N)); f = SR * 6
env[:f] = np.linspace(0, 1, f); env[int(SR*TOTAL)-f:int(SR*TOTAL)] = np.linspace(1, 0, f); env[int(SR*TOTAL):] = 0
out = simple_reverb(N * env)[:int(SR * TOTAL)]
out = out / np.max(np.abs(out)) * 0.85
with wave.open("output/flute_reference_5min.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes())
print("done", len(out) / SR, "s")
