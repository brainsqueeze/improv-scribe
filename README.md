# improv_scribe

Play your guitar or bass into your Mac and get back sheet music and guitar tablature as a PDF — automatically.

`improv_scribe` listens to your instrument through a microphone or USB audio interface, figures out what notes you're playing, and transcribes them into standard notation and tab. When you're done playing, export a PDF you can print, read on a tablet, or import into your DAW.

---

## What it does for musicians

- **Records a performance** from any macOS-compatible audio input (built-in mic, USB interface, etc.)
- **Detects notes automatically** using a high-accuracy pitch detection model tuned for guitar and bass frequencies
- **Produces standard sheet music** with the correct clef, time signature, and tempo marking
- **Produces guitar/bass tablature** alongside the sheet music in the same PDF — showing which string and fret to play each note on, using a position-optimizing algorithm that minimizes hand shifts
- **Exports MIDI** for importing into Logic, GarageBand, Ableton, or any DAW
- **Supports guitar and bass** out of the box (standard tuning, monophonic playing)

---

## Requirements

- macOS 13 Ventura or later
- [MuseScore 4](https://musescore.org) — for PDF rendering (`brew install musescore`)
- [Anaconda](https://www.anaconda.com) or Miniconda — for the Python environment

---

## Setup

```bash
# 1. Install system dependencies
brew install musescore portaudio

# 2. Create the Python environment
conda env create -f envionment.yaml
conda activate auto-sheet-music

# 3. Generate test fixtures (synthetic audio files used by the test suite)
python scripts/generate_fixtures.py

# 4. Run the test suite
conda run -n auto-sheet-music pytest

# 5. Launch the app
python -m improv_scribe
```

---

## Headless / CLI use

Useful for batch transcription or scripting:

```bash
python -m improv_scribe.cli \
  --device 0 \
  --instrument guitar \
  --duration 30 \
  --output ~/Music/transcriptions/
```

This captures 30 seconds from device 0, transcribes it, and writes `transcription.pdf` and `transcription.mid` to the output directory.

---

## Instruments supported

| Instrument | Range | Clef |
|---|---|---|
| Guitar (standard tuning) | E2 – D6 | Treble (8vb) |
| Bass guitar (standard tuning) | E1 – D4 | Bass |

Chord detection and alternate tunings are planned for future releases.

---

## How tablature fret positions are chosen

For each note, the app finds every (string, fret) combination that could play it, then uses dynamic programming to choose the sequence that minimizes total left-hand position shifts across the entire phrase. Open strings are preferred when position cost is equal. This produces tab that sits naturally under the hand rather than jumping arbitrarily across the neck.

---

## Developer setup

See [CLAUDE.md](CLAUDE.md) for the full module map, architecture overview, code style guide, and debugging tips.

```bash
conda run -n auto-sheet-music ruff check src/ tests/   # lint
conda run -n auto-sheet-music pytest --cov=improv_scribe  # test with coverage
```
