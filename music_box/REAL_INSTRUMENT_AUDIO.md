# Real Piano / Flute Audio (Instead of Synthesized)

The current scripts (`piano_sleep_music.py`, `flute_sleep_music.py`) build sound with math (additive synthesis). It works, but it sounds artificial. Real recorded samples sound much better. There are three ways to get them.

## Option 1: SoundFont + FluidSynth (Recommended)

Keep the melody logic, write notes as MIDI, and let FluidSynth render them with a real instrument recording.

**Pipeline:** Python melody -> `.mid` file -> FluidSynth + `.sf2` SoundFont -> `.wav`

### Setup

```bash
sudo apt install fluidsynth
pip install mido            # to write MIDI files
```

Download a free SoundFont (`.sf2`) and place it in `music_box/soundfonts/`:

| SoundFont | Good for | Notes |
|-----------|----------|-------|
| GeneralUser GS | piano + flute + more | small, good all-rounder |
| FluidR3_GM | piano + flute + more | installable via `sudo apt install fluid-soundfont-gm` |
| Salamander Grand Piano (sf2 build) | piano only | very high quality, large file |

### Render

```bash
fluidsynth -ni soundfonts/GeneralUser_GS.sf2 output/melody.mid \
  -F output/piano_real.wav -r 44100
```

General MIDI program numbers: Acoustic Grand Piano = 0, Flute = 73.

### Code change needed

Replace the synthesis functions with a MIDI writer:
- keep the note-choice logic (D minor pentatonic random walk)
- write each note as `note_on` / `note_off` with a velocity and a time
- add reverb in FluidSynth (`-R 1`) or afterwards with ffmpeg

## Option 2: AI Music Models

Generate audio directly from a text prompt, for example "soft solo piano, slow, sleep music".

| Model | Notes |
|-------|-------|
| MusicGen (Meta, via `audiocraft` or `transformers`) | easy, runs locally, works best with a GPU |
| Stable Audio Open | longer clips, needs more VRAM |

**Pros:** no melody code, and the sound is natural.
**Cons:** less control over the melody and key, and clips are short (10-30 s), so they must be stitched or looped.

## Option 3: Raw Sample Packs

Play individual recorded notes (`.wav`) yourself from Python, pitched and timed by the script.

| Source | Content |
|--------|---------|
| Salamander Grand Piano | multi-velocity piano samples |
| Philharmonia Orchestra samples | flute and other orchestral instruments |
| Freesound.org | mixed, check the licence on each file |
| VSCO 2 Community Orchestra | flute, strings and more |

**Pros:** full control and the most realistic sound.
**Cons:** more code, because you handle note pitch, looping and crossfades yourself.

## Comparison

| | Realism | Control over melody | Effort |
|---|---|---|---|
| Current synthesis | low | full | done |
| SoundFont + FluidSynth | high | full | low |
| AI model | high | low | low |
| Raw samples | highest | full | high |

## Suggested Next Steps

1. Install `fluidsynth` and download one SoundFont.
2. Add `piano_midi_music.py` that reuses the melody logic and writes a `.mid` file.
3. Render with FluidSynth and compare against `output/piano_sleep_music.wav`.
4. Try the flute program (73) once the piano works.

## Licence Note

Check the licence of every SoundFont or sample you download, especially if you plan to publish the music.
