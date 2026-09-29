#!/usr/bin/env python3
"""
Premium Sleep Music Generator v2
Creates beautiful, relaxing ambient sleep music with proper instruments
"""

import numpy as np
import wave
import time
from datetime import datetime

SAMPLE_RATE = 44100
DURATION_SECONDS = 300  # 5 minutes
CHANNELS = 2

class InstrumentSynthesizer:
    """Synthesize realistic instrument sounds"""

    @staticmethod
    def pad_synth(frequency, duration, sr=SAMPLE_RATE):
        """Ambient pad - warm, evolving texture like string ensemble"""
        t = np.arange(int(sr * duration)) / sr

        # Multiple detuned oscillators for richness
        osc1 = np.sin(2 * np.pi * frequency * t)
        osc2 = np.sin(2 * np.pi * frequency * 1.002 * t)  # Slight detuning
        osc3 = np.sin(2 * np.pi * frequency * 0.998 * t)

        # Blend oscillators
        wave_data = (osc1 + osc2 * 0.7 + osc3 * 0.7) / 2.4

        # Smooth, long envelope
        attack = int(sr * 2)
        decay = int(sr * duration * 0.3)
        sustain_level = 0.7
        release = int(sr * 3)

        envelope = np.ones(int(sr * duration))
        envelope[:attack] = np.linspace(0, 1, attack)

        sustain_start = attack
        sustain_end = max(attack, int(sr * duration) - release)
        if sustain_start < sustain_end:
            envelope[sustain_start:sustain_end] = sustain_level

        envelope[-release:] = np.linspace(sustain_level, 0, release)

        return wave_data * envelope * 0.25

    @staticmethod
    def soft_bell(frequency, duration, sr=SAMPLE_RATE):
        """Soft, mellow bell-like tones"""
        t = np.arange(int(sr * duration)) / sr

        # Base tone with slight harmonics
        fundamental = np.sin(2 * np.pi * frequency * t)
        harmonic2 = np.sin(2 * np.pi * frequency * 2 * t) * 0.3
        harmonic3 = np.sin(2 * np.pi * frequency * 3 * t) * 0.15

        wave_data = (fundamental + harmonic2 + harmonic3) / 1.45

        # Quick attack, long decay for bell quality
        attack = int(sr * 0.05)
        decay_start = int(sr * 0.2)

        envelope = np.ones(int(sr * duration))
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[decay_start:] = np.exp(-3.5 * (t[decay_start:] - t[decay_start]))

        return wave_data * envelope * 0.25

    @staticmethod
    def warm_bass(frequency, duration, sr=SAMPLE_RATE):
        """Deep, warm bass foundation"""
        t = np.arange(int(sr * duration)) / sr

        # Low frequency sine with sub-bass
        wave_data = np.sin(2 * np.pi * frequency * t)
        wave_data += np.sin(2 * np.pi * frequency * 0.5 * t) * 0.5

        # Very smooth envelope
        attack = int(sr * 3)
        release = int(sr * 4)

        envelope = np.ones(int(sr * duration))
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)

        return wave_data * envelope * 0.15 / 1.5

    @staticmethod
    def apply_reverb(audio, decay_time=2.0, sr=SAMPLE_RATE):
        """Add space and reverb to audio"""
        delay_samples = int(sr * 0.05)  # 50ms delay
        reverb = np.zeros(len(audio) + int(sr * decay_time))

        reverb[:len(audio)] = audio

        for i in range(1, 4):
            delay = delay_samples * i
            decay = 0.5 ** i
            if delay < len(reverb) - len(audio):
                reverb[delay:delay + len(audio)] += audio * decay

        return reverb[:len(audio)]

def generate_premium_sleep_music(duration=300):
    """Generate high-quality sleep music"""
    print(f"🎵 Generating premium {duration}s sleep music...")
    start_time = time.time()

    sr = SAMPLE_RATE
    total_samples = int(sr * duration)
    synth = InstrumentSynthesizer()

    # Create high-res audio
    music = np.zeros(total_samples)

    # Sleep music in A minor (very soothing)
    # A = 110 Hz (lower octave for calming effect)
    notes = {
        'A2': 110,
        'C3': 130.81,
        'E3': 164.81,
        'G3': 196.00,
        'A3': 220,
        'C4': 261.63,
        'E4': 329.63,
        'A4': 440,
        'E5': 659.25,
    }

    # ===== Section 1: Pad Foundation (0-60s) =====
    print("  • Section 1: Establishing atmosphere...")
    # Start with low pad to create foundation
    pad_duration = 60
    for i in range(4):
        offset = i * 15
        if offset >= pad_duration:
            break
        # Play held chords
        chord_freqs = [
            [notes['A2'], notes['C3'], notes['E3']],  # Am chord
            [notes['A2'], notes['E3'], notes['A3']],
        ]
        freq = chord_freqs[i % 2]

        for f in freq:
            pad = synth.pad_synth(f, 15, sr=sr)
            start_sample = int(sr * offset)
            pad_len = min(len(pad), total_samples - start_sample)
            music[start_sample:start_sample + pad_len] += pad[:pad_len]

    # ===== Section 2: Add Gentle Melody (60-180s) =====
    print("  • Section 2: Adding gentle melody...")
    melody_section = [
        ('A4', 3), ('G3', 2), ('E4', 4), ('C4', 3),
        ('A4', 3), ('E4', 2), ('A3', 5), ('C4', 3),
        ('E4', 3), ('G3', 2), ('A4', 4), ('E3', 3),
        ('A3', 3), ('C4', 2), ('E4', 5), ('A4', 3),
    ]

    current_time = 60
    for note_name, duration_s in melody_section:
        if current_time >= 180:
            break

        start_sample = int(sr * current_time)
        if start_sample >= total_samples:
            break

        # Mix bell sounds for melody
        bell = synth.soft_bell(notes[note_name], duration_s, sr=sr)
        bell_len = min(len(bell), total_samples - start_sample)
        music[start_sample:start_sample + bell_len] += bell[:bell_len] * 0.6

        current_time += duration_s

    # ===== Section 3: Build Warmth (180-240s) =====
    print("  • Section 3: Building warmth...")
    warm_chords = [
        (notes['A2'], 10),
        (notes['E3'], 10),
        (notes['C3'], 10),
        (notes['A2'], 10),
        (notes['G3'], 10),
        (notes['E3'], 10),
    ]

    current_time = 180
    for freq, dur in warm_chords:
        if current_time >= 240:
            break

        start_sample = int(sr * current_time)
        if start_sample >= total_samples:
            break

        bass = synth.warm_bass(freq, dur, sr=sr)
        bass_len = min(len(bass), total_samples - start_sample)
        music[start_sample:start_sample + bass_len] += bass[:bass_len]

        current_time += dur

    # ===== Section 4: Fade Out (240-300s) =====
    print("  • Section 4: Gentle fade out...")
    for i in range(6):
        offset = 240 + (i * 10)
        if offset >= duration:
            break

        freq = notes['A2'] if i % 2 == 0 else notes['E3']
        pad = synth.pad_synth(freq, 10, sr=sr)

        start_sample = int(sr * offset)
        pad_len = min(len(pad), total_samples - start_sample)
        music[start_sample:start_sample + pad_len] += pad[:pad_len] * (1 - i/6)

    # Normalize
    max_val = np.max(np.abs(music))
    if max_val > 0:
        music = music / (max_val * 1.2)

    # Add subtle reverb for space
    print("  • Adding reverb and spatial effects...")
    music = synth.apply_reverb(music, decay_time=1.5, sr=sr)[:total_samples]

    # Final normalization
    max_val = np.max(np.abs(music))
    if max_val > 0:
        music = music / (max_val * 1.1)

    elapsed = time.time() - start_time
    print(f"✓ Generation complete in {elapsed:.2f}s")

    return music

def save_wav(audio_data, filename, sr=SAMPLE_RATE, channels=2):
    """Save audio to WAV file"""
    print(f"💾 Saving to {filename}...")
    save_start = time.time()

    # Stereo
    if len(audio_data.shape) == 1:
        stereo_audio = np.column_stack([audio_data, audio_data])
    else:
        stereo_audio = audio_data

    # 16-bit PCM
    audio_int16 = np.int16(stereo_audio * 32767 * 0.95)  # Leave headroom

    # Write
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_int16.tobytes())

    file_size = np.round(np.prod(audio_int16.shape) * 2 / (1024**2), 2)
    elapsed = time.time() - save_start
    print(f"✓ Saved {file_size}MB in {elapsed:.2f}s")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌙 PREMIUM SLEEP MUSIC GENERATOR v2 🌙")
    print("="*60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: 5 minutes of premium ambient sleep music")
    print("="*60 + "\n")

    total_start = time.time()

    music = generate_premium_sleep_music(DURATION_SECONDS)

    import os
    os.makedirs("output", exist_ok=True)
    save_wav(music, "output/sleep_music_premium.wav", sr=SAMPLE_RATE, channels=CHANNELS)

    total_time = time.time() - total_start

    print("\n" + "="*60)
    print("✨ PREMIUM SLEEP MUSIC READY!")
    print(f"Total time: {total_time:.2f}s ({DURATION_SECONDS/total_time:.1f}x realtime)")
    print(f"Output: output/sleep_music_premium.wav")
    print("🎵 Enjoy peaceful sleep! 🎵")
    print("="*60 + "\n")
