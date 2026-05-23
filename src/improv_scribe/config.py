"""
config.py — Global configuration and environment variable resolution.

All tuneable constants live here. Override any value via environment variables
(prefix ATS_) without touching source code. Example:
    ATS_SAMPLE_RATE=48000 python -m improv_scribe
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Audio capture defaults
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = int(os.getenv("ATS_SAMPLE_RATE", "44100"))
CHANNELS: int = 1
BLOCK_SIZE: int = int(os.getenv("ATS_BLOCK_SIZE", "512"))
RING_BUFFER_SECONDS: float = float(os.getenv("ATS_RING_BUFFER_SECONDS", "10.0"))

# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------

PITCH_BACKEND: str = os.getenv("ATS_PITCH_BACKEND", "basic_pitch")
FRAME_LENGTH: int = int(os.getenv("ATS_FRAME_LENGTH", "2048"))
HOP_LENGTH: int = int(os.getenv("ATS_HOP_LENGTH", "512"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("ATS_CONFIDENCE_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Polyphonic detection (Phase 1+) — calibrated from prerequisite probe findings
# ---------------------------------------------------------------------------

# Absolute amplitude floor for basic-pitch events. Below this, a detection is
# dropped at the backend boundary. Calibrated against the four mono integration
# samples (see Task 9 of Phase 1): genuine notes register at >= 0.68 amplitude;
# spurious detections (harmonics, sympathetic resonance) cluster at 0.30 - 0.55.
# 0.65 keeps all real notes on guitar samples; bass samples retain one
# sympathetic-resonance detection that can't be filtered without losing real notes.
POLYPHONIC_AMPLITUDE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_AMPLITUDE_FLOOR", "0.65"))

# Drop basic-pitch events shorter than this duration. Attack-transient
# fragments are typically < 50 ms. Defaults to 50 ms.
MIN_NOTE_DURATION_S: float = float(os.getenv("ATS_MIN_NOTE_DURATION_S", "0.050"))

# Window for grouping basic-pitch's flat note events into chord NoteEvents.
# Calibrated against real strum recordings (spec §13.1): actual onset spreads
# within dyads are 0-35 ms; 100 ms provides comfortable headroom.
ONSET_GROUPING_WINDOW_MS: float = float(os.getenv("ATS_ONSET_GROUPING_WINDOW_MS", "100.0"))

# Within a cluster, drop members whose amplitude is below this fraction of the
# cluster's max amplitude. Defends against basic-pitch's loud-note-dominates
# behaviour where the strongest member registers at 0.85 but a quiet member
# is at 0.42. Phase 4 may revisit; default is conservative (0.5).
POLYPHONIC_RELATIVE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_RELATIVE_FLOOR", "0.5"))

# Tighter merge threshold for chord events. Eighth-note repeated chords at
# 100 BPM are 300 ms apart and must NOT merge into one held chord. Singletons
# keep the existing 600 ms threshold (defined in note_tracker.py).
MERGE_GAP_CHORD_MS: float = float(os.getenv("ATS_MERGE_GAP_CHORD_MS", "200.0"))

# ---------------------------------------------------------------------------
# Noise gate
# ---------------------------------------------------------------------------

NOISE_GATE_RMS_THRESHOLD: float = float(os.getenv("ATS_NOISE_GATE_RMS", "0.01"))
NOISE_GATE_HOLD_MS: float = float(os.getenv("ATS_NOISE_GATE_HOLD_MS", "80.0"))

# ---------------------------------------------------------------------------
# Rhythm quantization
# ---------------------------------------------------------------------------

TEMPO_MIN_BPM: float = float(os.getenv("ATS_TEMPO_MIN_BPM", "40.0"))
TEMPO_MAX_BPM: float = float(os.getenv("ATS_TEMPO_MAX_BPM", "240.0"))

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

MUSESCORE_PATH: str = os.getenv(
    "ATS_MUSESCORE_PATH",
    "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
)

DEFAULT_OUTPUT_DIR: Path = Path(
    os.getenv("ATS_OUTPUT_DIR", str(Path.home() / "Music" / "AudioToSheet"))
)

# ---------------------------------------------------------------------------
# Debug flags
# ---------------------------------------------------------------------------

DEBUG_PITCH: bool = os.getenv("ATS_DEBUG_PITCH", "0") == "1"
DEBUG_PITCH_CSV: Path = Path("/tmp/ats_pitch_debug.csv")


# ---------------------------------------------------------------------------
# AppConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Snapshot of runtime configuration. Pass this to subsystems instead of
    importing module-level constants directly — makes testing and mocking easy."""

    sample_rate: int = field(default_factory=lambda: SAMPLE_RATE)
    channels: int = field(default_factory=lambda: CHANNELS)
    block_size: int = field(default_factory=lambda: BLOCK_SIZE)
    ring_buffer_seconds: float = field(default_factory=lambda: RING_BUFFER_SECONDS)

    pitch_backend: str = field(default_factory=lambda: PITCH_BACKEND)
    frame_length: int = field(default_factory=lambda: FRAME_LENGTH)
    hop_length: int = field(default_factory=lambda: HOP_LENGTH)
    confidence_threshold: float = field(default_factory=lambda: CONFIDENCE_THRESHOLD)

    polyphonic_amplitude_floor: float = field(default_factory=lambda: POLYPHONIC_AMPLITUDE_FLOOR)
    min_note_duration_s: float = field(default_factory=lambda: MIN_NOTE_DURATION_S)

    onset_grouping_window_ms: float = field(default_factory=lambda: ONSET_GROUPING_WINDOW_MS)
    polyphonic_relative_floor: float = field(default_factory=lambda: POLYPHONIC_RELATIVE_FLOOR)
    merge_gap_chord_ms: float = field(default_factory=lambda: MERGE_GAP_CHORD_MS)

    noise_gate_rms: float = field(default_factory=lambda: NOISE_GATE_RMS_THRESHOLD)
    noise_gate_hold_ms: float = field(default_factory=lambda: NOISE_GATE_HOLD_MS)

    tempo_min: float = field(default_factory=lambda: TEMPO_MIN_BPM)
    tempo_max: float = field(default_factory=lambda: TEMPO_MAX_BPM)

    musescore_path: str = field(default_factory=lambda: MUSESCORE_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    debug_pitch: bool = field(default_factory=lambda: DEBUG_PITCH)
    debug_pitch_csv: Path = field(default_factory=lambda: DEBUG_PITCH_CSV)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
