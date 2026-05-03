"""
cli.py — Batch transcription CLI (no GUI required).

Transcribes a .wav file and writes PDF and/or MIDI output.

Usage
-----
python -m audio_to_sheet.cli \\
    --input recording.wav \\
    --instrument guitar \\
    --backend pyin \\
    --mode auto \\
    --output-pdf out.pdf \\
    --output-midi out.mid

For a full option list:
    python -m audio_to_sheet.cli --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_wav(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load a WAV file, convert to mono float32, resample if needed."""
    import scipy.io.wavfile as wav  # type: ignore[import]
    import scipy.signal as signal

    sr, data = wav.read(str(path))
    data = data.astype(np.float32)

    # Normalise integer formats
    if data.dtype == np.int16:
        data = data / 32768.0
    elif data.dtype == np.int32:
        data = data / 2147483648.0

    # Mix down to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        n_samples = int(len(data) * target_sr / sr)
        data = signal.resample(data, n_samples).astype(np.float32)
        sr = target_sr

    return data, sr


def run(args: argparse.Namespace) -> int:
    """Main CLI logic. Returns exit code."""
    from audio_to_sheet.analysis.instrument_profiles import get_profile, Instrument
    from audio_to_sheet.analysis.note_tracker import NoteTracker
    from audio_to_sheet.analysis.onset import OnsetDetector
    from audio_to_sheet.analysis.pitch import PitchEstimator
    from audio_to_sheet.config import AppConfig
    from audio_to_sheet.export.midi_exporter import MIDIExporter
    from audio_to_sheet.export.pdf_exporter import PDFExporter
    from audio_to_sheet.notation.score_builder import ScoreBuilder
    from audio_to_sheet.quantization.grid import RhythmQuantizer
    from audio_to_sheet.quantization.tempo import TempoEstimator

    config = AppConfig()
    config.pitch_backend = args.backend

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading {input_path} …")
    audio, sr = _load_wav(input_path, config.sample_rate)
    print(f"  {len(audio) / sr:.2f}s of audio at {sr} Hz")

    instrument = Instrument(args.instrument.lower())
    profile = get_profile(instrument)
    print(f"  Instrument: {profile.name}")

    # 1. Pitch
    print(f"Estimating pitch ({args.backend}) …")
    estimator = PitchEstimator(config, backend=args.backend)
    pitch_result = estimator.estimate(audio, profile)
    n_voiced = len(pitch_result.voiced_frames)
    print(f"  {n_voiced} voiced frames detected")

    # 2. Onsets
    print("Detecting onsets …")
    onset_detector = OnsetDetector(config)
    onsets = onset_detector.detect(audio)
    print(f"  {len(onsets)} onsets detected")

    # 3. NoteEvents
    tracker = NoteTracker(config, profile)
    events = tracker.process(pitch_result, onsets)
    print(f"  {len(events)} note events assembled")

    if not events:
        print("ERROR: No notes detected. Check --instrument and input file.", file=sys.stderr)
        return 1

    # 4. Tempo
    tempo_estimator = TempoEstimator(config)
    tempo_result = tempo_estimator.estimate(events)
    print(f"  Estimated tempo: {tempo_result.bpm:.1f} BPM (confidence={tempo_result.confidence:.2f})")

    # 5. Quantize + score
    score = None
    quantized_notes = None
    if args.mode == "auto":
        print("Quantizing rhythm …")
        quantizer = RhythmQuantizer(tempo_result)
        quantized_notes = quantizer.quantize(events)
        score_builder = ScoreBuilder(profile, tempo_result, title=input_path.stem)
        score = score_builder.build(quantized_notes)
        print(f"  {len(quantized_notes)} quantized elements (notes + rests)")

    # 6. Export
    if args.output_pdf:
        if score is None:
            print("WARNING: PDF export requires --mode auto. Skipping PDF.", file=sys.stderr)
        else:
            print(f"Exporting PDF → {args.output_pdf} …")
            exporter = PDFExporter(config)
            out = exporter.export(score, Path(args.output_pdf))
            print(f"  PDF written: {out}")

    if args.output_midi:
        print(f"Exporting MIDI → {args.output_midi} …")
        midi_exporter = MIDIExporter(config)
        if score is not None and args.mode == "auto":
            out = midi_exporter.quantized_from_score(score, Path(args.output_midi))
        else:
            out = midi_exporter.raw_from_events(events, tempo_result, Path(args.output_midi))
        print(f"  MIDI written: {out}")

    print("Done.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m audio_to_sheet.cli",
        description="Batch audio-to-sheet-music transcription (no GUI).",
    )
    p.add_argument("--input", required=True, help="Path to input .wav file")
    p.add_argument(
        "--instrument", choices=["guitar", "bass"], default="guitar",
        help="Instrument type (default: guitar)"
    )
    p.add_argument(
        "--backend", choices=["pyin", "crepe"], default="pyin",
        help="Pitch detection backend (default: pyin)"
    )
    p.add_argument(
        "--mode", choices=["auto", "raw"], default="auto",
        help="Rhythm mode: auto=grid-snap, raw=float timing (default: auto)"
    )
    p.add_argument("--output-pdf", metavar="PATH", help="Write rendered PDF to PATH")
    p.add_argument("--output-midi", metavar="PATH", help="Write MIDI to PATH")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))
