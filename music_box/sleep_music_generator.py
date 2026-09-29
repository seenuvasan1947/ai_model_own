#!/usr/bin/env python3
"""
Sleep Music Generator - Creates ambient sleep music using instruments
Generates 5 minutes of relaxing instrumental music
"""

import numpy as np
import wave
import time
from datetime import datetime

# Audio parameters
SAMPLE_RATE = 44100  # 44.1kHz CD quality
DURATION_SECONDS = 300  # 5 minutes
TEMPO_BPM = 60  # Slow, relaxing tempo
CHANNELS = 2  # Stereo
BITS_PER_SAMPLE = 16

def sine_wave(frequency, duration, amplitude=0.3, sr=SAMPLE_RATE):
    """Generate a sine wave at given frequency"""
    t = np.arange(int(sr * duration)) / sr
    return amplitude * np.sin(2 * np.pi * frequency * t)

def apply_envelope(wave_data, duration, sr=SAMPLE_RATE, attack=0.05, decay=0.1):
    """Apply ADSR envelope to smooth note transitions"""
    total_samples = int(sr * duration)
    attack_samples = int(sr * attack)
    decay_samples = int(sr * decay)

    envelope = np.ones(total_samples)
    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Decay
    if total_samples > attack_samples + decay_samples:
        envelope[-decay_samples:] = np.linspace(1, 0.7, decay_samples)

    return wave_data * envelope

def generate_chord(frequencies, duration, sr=SAMPLE_RATE):
    """Generate a chord from multiple frequencies"""
    chord = np.zeros(int(sr * duration))
    for freq in frequencies:
        chord += sine_wave(freq, duration, amplitude=0.2, sr=sr)
    return chord / len(frequencies)  # Normalize

def generate_sleep_music(duration=300, sr=SAMPLE_RATE):
    """
    Generate 5 minutes of ambient sleep music
    Uses:
    - Slow tempo (60 BPM = 1 beat per second)
    - Major/minor chords in A (soothing key)
    - Layered instruments (pad, melody, bass)
    - Gradual transitions
    """
    print(f"🎵 Generating {duration}s ({duration//60}min) sleep music...")
    start_time = time.time()

    total_samples = int(sr * duration)
    music = np.zeros(total_samples)

    # Sleep music key: A major scale
    # Frequencies (A3 = 220Hz as base)
    notes = {
        'A3': 220,
        'B3': 246.94,
        'C#4': 277.18,
        'D4': 293.66,
        'E4': 329.63,
        'F#4': 369.99,
        'G#4': 415.30,
        'A4': 440,
        'B4': 493.88,
        'C#5': 554.37,
        'E5': 659.25,
    }

    # Create ambient pad (low, sustained chords)
    pad_chords = [
        [notes['A3'], notes['C#4'], notes['E4']],      # A minor
        [notes['A3'], notes['C#4'], notes['E4']],      # Same chord, sustained
        [notes['D4'], notes['F#4'], notes['A4']],      # D major
        [notes['E4'], notes['G#4'], notes['B4']],      # E major
    ]

    chord_duration = 30  # Each chord lasts 30 seconds
    for chord_idx, chord_freqs in enumerate(pad_chords * 2):  # Play twice for 4min
        start_sample = int(sr * chord_duration * chord_idx)
        if start_sample >= total_samples:
            break
        chord_samples = min(int(sr * chord_duration), total_samples - start_sample)
        pad = apply_envelope(
            generate_chord(chord_freqs, chord_samples/sr, sr=sr),
            chord_samples/sr,
            sr=sr
        )
        music[start_sample:start_sample + len(pad)] += pad * 0.4

    # Create gentle melody (higher notes)
    melody_notes = [
        'A4', 'B4', 'C#5', 'A4', 'B4', 'A4', 'E4', 'A4',
        'A4', 'A4', 'B4', 'C#5', 'B4', 'A4', 'E4', 'A4',
    ]

    note_duration = 4  # Each note lasts 4 seconds
    for note_idx, note in enumerate(melody_notes):
        if note_idx * note_duration >= duration:
            break
        start_sample = int(sr * note_idx * note_duration)
        if start_sample >= total_samples:
            break

        note_samples = min(int(sr * note_duration), total_samples - start_sample)
        melody = apply_envelope(
            sine_wave(notes[note], note_samples/sr, amplitude=0.25, sr=sr),
            note_samples/sr,
            sr=sr,
            attack=0.1,
            decay=0.5
        )
        music[start_sample:start_sample + len(melody)] += melody * 0.3

    # Normalize to prevent clipping
    max_val = np.max(np.abs(music))
    if max_val > 0:
        music = music / (max_val * 1.1)  # Slight headroom

    elapsed = time.time() - start_time
    print(f"✓ Generation complete in {elapsed:.2f} seconds")

    return music

def save_wav(audio_data, filename, sr=SAMPLE_RATE, channels=2):
    """Save audio data to WAV file"""
    print(f"💾 Saving to {filename}...")
    save_start = time.time()

    # Convert to stereo if mono
    if len(audio_data.shape) == 1:
        stereo_audio = np.column_stack([audio_data, audio_data])
    else:
        stereo_audio = audio_data

    # Convert to 16-bit PCM
    audio_int16 = np.int16(stereo_audio * 32767)

    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 2 bytes for 16-bit
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_int16.tobytes())

    file_size_mb = os.path.getsize(filename) / (1024**2)
    save_time = time.time() - save_start
    print(f"✓ Saved {file_size_mb:.2f}MB in {save_time:.2f} seconds")

if __name__ == "__main__":
    print("\n🎼 SLEEP MUSIC GENERATOR 🎼")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target: 5 minutes (300s) of sleep music")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print("=" * 50 + "\n")

    total_start = time.time()

    # Generate music
    music = generate_sleep_music(DURATION_SECONDS, SAMPLE_RATE)

    # Save to file
    import os
    os.makedirs("output", exist_ok=True)
    output_file = "output/sleep_music_5min.wav"
    save_wav(music, output_file, sr=SAMPLE_RATE, channels=CHANNELS)

    total_time = time.time() - total_start
    print("\n" + "=" * 50)
    print(f"✨ COMPLETE!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Efficiency: {DURATION_SECONDS/total_time:.2f}x realtime")
    print(f"Output: {output_file}")
    print("=" * 50)
