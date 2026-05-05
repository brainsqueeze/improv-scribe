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

PITCH_BACKEND: str = os.getenv("ATS_PITCH_BACKEND", "crepe")
FRAME_LENGTH: int = int(os.getenv("ATS_FRAME_LENGTH", "2048"))
HOP_LENGTH: int = int(os.getenv("ATS_HOP_LENGTH", "512"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("ATS_CONFIDENCE_THRESHOLD", "0.5"))

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
