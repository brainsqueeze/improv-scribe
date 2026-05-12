"""Phase 1 prerequisite probe: verify basic-pitch's actual API on this machine.

Records:
  - import surface (which backends loaded?)
  - return shape of predict()
  - cold-start latency (first call)
  - warm-call latency (second call)
  - whether numpy-array input is accepted
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

print("=" * 60)
print("PROBE: basic-pitch installation introspection")
print("=" * 60)

print(f"Python: {sys.version.split()[0]}")

import basic_pitch  # noqa: E402
print(f"basic-pitch: {getattr(basic_pitch, '__version__', 'unknown')}")
print(f"basic-pitch __file__: {basic_pitch.__file__}")

from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
print(f"ICASSP_2022_MODEL_PATH = {ICASSP_2022_MODEL_PATH}")
print(f"  type = {type(ICASSP_2022_MODEL_PATH).__name__}")
print(f"  exists = {Path(str(ICASSP_2022_MODEL_PATH)).exists()}")

print()
print("=" * 60)
print("PROBE: predict() on a real sample (path input)")
print("=" * 60)

sample = Path(__file__).resolve().parents[0] / ".." / "Users/davehollander/Documents/Personal/Projects/audio_to_sheet/samples/guitar/6_string_electric_line_in.mp3"
# Fallback to absolute path since the relative is fragile
sample_abs = Path(
    "/Users/davehollander/Documents/Personal/Projects/audio_to_sheet/samples/guitar/6_string_electric_line_in.mp3"
)
print(f"Sample: {sample_abs}")
print(f"  exists = {sample_abs.exists()}")

from basic_pitch.inference import predict  # noqa: E402

t0 = time.perf_counter()
result = predict(str(sample_abs))
t1 = time.perf_counter()
print(f"Cold-call latency: {t1 - t0:.2f} s")

print(f"Return type: {type(result).__name__}, len={len(result) if hasattr(result, '__len__') else '?'}")
if isinstance(result, tuple):
    for i, item in enumerate(result):
        print(f"  result[{i}]: type={type(item).__name__}", end="")
        if hasattr(item, "__len__"):
            print(f" len={len(item)}", end="")
        if hasattr(item, "keys"):
            print(f" keys={list(item.keys())[:5]}", end="")
        print()

# Unpack the conventional 3-tuple
model_output, midi_data, note_events = result
print()
print(f"model_output type: {type(model_output).__name__}")
if hasattr(model_output, "keys"):
    print(f"  keys = {list(model_output.keys())}")
print(f"midi_data type: {type(midi_data).__name__}")
print(f"note_events type: {type(note_events).__name__}, len={len(note_events)}")

if note_events:
    first = note_events[0]
    print(f"  note_events[0] type: {type(first).__name__}, len={len(first) if hasattr(first, '__len__') else '?'}")
    print(f"  note_events[0] = {first}")
    if isinstance(first, tuple):
        for i, field in enumerate(first):
            print(f"    field[{i}]: type={type(field).__name__}, value={field!r}")

    print()
    print(f"All {len(note_events)} note events:")
    for i, ev in enumerate(note_events[:20]):
        print(f"  [{i}] start={ev[0]:.3f}s end={ev[1]:.3f}s midi={ev[2]} amp={ev[3]:.3f} pitch_bend={ev[4] if len(ev) > 4 else None}")

print()
print("=" * 60)
print("PROBE: warm-call latency + numpy array input")
print("=" * 60)

import librosa
import numpy as np

y, sr = librosa.load(str(sample_abs), sr=22050, mono=True)
print(f"loaded audio: dtype={y.dtype}, shape={y.shape}, sr={sr}")

t2 = time.perf_counter()
try:
    result2 = predict(y)  # try numpy array directly
    t3 = time.perf_counter()
    print(f"numpy-array input: SUCCEEDED  (warm-call: {t3 - t2:.2f} s)")
    _, _, note_events_np = result2
    print(f"  -> {len(note_events_np)} notes")
except Exception as e:
    t3 = time.perf_counter()
    print(f"numpy-array input: FAILED after {t3 - t2:.2f}s with {type(e).__name__}: {e}")
    print("  -> path input is the only supported route")

print()
print("=" * 60)
print("DONE")
print("=" * 60)
