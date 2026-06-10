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

# Amplitude floor for SINGLETON clusters on the basic-pitch path. Calibrated
# against the mono integration samples: genuine isolated notes register at
# >= 0.68 amplitude; isolated ghost re-detections of ringing strings cluster
# at 0.29 - 0.43. Applied at the cluster stage (note_tracker), not the
# backend — chord members are allowed below this when corroborated by
# cluster context (see POLYPHONIC_MULTI_FLOOR).
POLYPHONIC_AMPLITUDE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_AMPLITUDE_FLOOR", "0.65"))

# Permissive pre-filter applied at the backend boundary. Real chord members
# register as low as 0.29 (weak high strings in a strum); model hallucinations
# below 0.25 carry no information. The strict filtering happens later with
# cluster context.
POLYPHONIC_PRE_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_PRE_FLOOR", "0.25"))

# Absolute amplitude floor for members of MULTI-member clusters. Weak chord
# members are corroborated by the cluster's stronger members, so this sits
# well below the singleton floor. (2026-06 precision audit, docs/
# precision_audit_basic_pitch.md.)
POLYPHONIC_MULTI_FLOOR: float = float(os.getenv("ATS_POLYPHONIC_MULTI_FLOOR", "0.30"))

# Drop basic-pitch events shorter than this duration. Attack-transient
# fragments are typically < 50 ms. Defaults to 50 ms.
MIN_NOTE_DURATION_S: float = float(os.getenv("ATS_MIN_NOTE_DURATION_S", "0.050"))

# basic-pitch splits one re-articulated note into a short attack fragment
# followed by the sustained event with ~0 gap. Fragments shorter than
# FRAGMENT_MAX_DURATION_S are absorbed into the adjacent same-pitch event
# when the gap is at most FRAGMENT_MAX_GAP_S. Long events are never merged
# with each other (those are genuine re-articulations).
FRAGMENT_MAX_DURATION_S: float = float(os.getenv("ATS_FRAGMENT_MAX_DURATION_S", "0.18"))
FRAGMENT_MAX_GAP_S: float = float(os.getenv("ATS_FRAGMENT_MAX_GAP_S", "0.10"))

# Window for grouping basic-pitch's flat note events into chord NoteEvents.
# basic-pitch emits weak strings late (activation crosses threshold mid-
# attack): measured start-time spreads within one strum run to ~280 ms on the
# open-chord samples, while eighth-note strums at 100 BPM are 300 ms apart.
# 250 ms captures the former and separates the latter.
ONSET_GROUPING_WINDOW_MS: float = float(os.getenv("ATS_ONSET_GROUPING_WINDOW_MS", "250.0"))

# Ring-over suppression: a non-max cluster member whose pitch was already
# detected recently (chain of same-pitch events with gaps <= RING_CHAIN_GAP_S)
# at >= its amplitude / RING_SUPPRESSION_RATIO is a re-detection of a
# still-ringing string, not a newly played note. Measured ratios: ghosts sit
# at 0.40-0.55 of their source's amplitude; re-strummed chord members at
# >= 0.62.
RING_SUPPRESSION_RATIO: float = float(os.getenv("ATS_RING_SUPPRESSION_RATIO", "0.60"))
RING_CHAIN_GAP_S: float = float(os.getenv("ATS_RING_CHAIN_GAP_S", "3.0"))

# Weak clusters (max amplitude below POLYPHONIC_AMPLITUDE_FLOOR) must align
# with a librosa-detected onset to survive: a real strum has an attack, a
# decay-phase ghost cluster does not. Strong clusters are exempt because
# librosa misses some genuine strums (measured: 2 of 6 on the open-A sample).
# Tolerances are relative to the cluster anchor: basic-pitch may emit the
# first member slightly before (fragment) or after the physical attack.
ONSET_GATE_TOL_BEFORE_S: float = float(os.getenv("ATS_ONSET_GATE_TOL_BEFORE_S", "0.20"))
ONSET_GATE_TOL_AFTER_S: float = float(os.getenv("ATS_ONSET_GATE_TOL_AFTER_S", "0.15"))

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
    polyphonic_pre_floor: float = field(default_factory=lambda: POLYPHONIC_PRE_FLOOR)
    polyphonic_multi_floor: float = field(default_factory=lambda: POLYPHONIC_MULTI_FLOOR)
    min_note_duration_s: float = field(default_factory=lambda: MIN_NOTE_DURATION_S)
    fragment_max_duration_s: float = field(default_factory=lambda: FRAGMENT_MAX_DURATION_S)
    fragment_max_gap_s: float = field(default_factory=lambda: FRAGMENT_MAX_GAP_S)

    onset_grouping_window_ms: float = field(default_factory=lambda: ONSET_GROUPING_WINDOW_MS)
    polyphonic_relative_floor: float = field(default_factory=lambda: POLYPHONIC_RELATIVE_FLOOR)
    merge_gap_chord_ms: float = field(default_factory=lambda: MERGE_GAP_CHORD_MS)
    ring_suppression_ratio: float = field(default_factory=lambda: RING_SUPPRESSION_RATIO)
    ring_chain_gap_s: float = field(default_factory=lambda: RING_CHAIN_GAP_S)
    onset_gate_tol_before_s: float = field(default_factory=lambda: ONSET_GATE_TOL_BEFORE_S)
    onset_gate_tol_after_s: float = field(default_factory=lambda: ONSET_GATE_TOL_AFTER_S)

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
