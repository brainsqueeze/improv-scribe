"""
analysis/pitch.py — Pitch estimation with pluggable backends.

Backends
--------
pyin  : librosa.pyin — probabilistic YIN, CPU-only, no model weights.
        Returns voiced probability per frame. Fast and reliable for clean
        monophonic signals. Default for MVP.

crepe : CREPE CNN (Kim et al. 2018) via the `crepe` package (optional dep).
        More robust on noisy/attack-transient signals. Requires `pip install crepe`.

Both backends return a common FramePitchResult dataclass.

Design note
-----------
pYIN operates on a full audio chunk at once (batch mode). CREPE similarly
processes a chunk. For real-time use, the caller should accumulate a
sufficient analysis window (≥ frame_length samples) before calling estimate().
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_to_sheet.analysis.instrument_profiles import InstrumentProfile
from audio_to_sheet.config import AppConfig

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PitchFrame:
    """Pitch estimate for a single analysis frame."""
    time_s: float         # frame centre time (seconds from chunk start)
    freq_hz: float        # estimated fundamental frequency, or NaN if unvoiced
    confidence: float     # voiced probability [0, 1]
    is_voiced: bool       # confidence >= threshold


@dataclass
class PitchResult:
    """Collection of PitchFrames for one analysis chunk."""
    frames: list[PitchFrame]
    sample_rate: int
    hop_length: int

    @property
    def voiced_frames(self) -> list[PitchFrame]:
        return [f for f in self.frames if f.is_voiced]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _PitchBackend:
    def estimate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        profile: InstrumentProfile,
        config: AppConfig,
    ) -> PitchResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# pYIN backend
# ---------------------------------------------------------------------------

class _PYinBackend(_PitchBackend):
    """
    Wraps librosa.pyin.

    pYIN fits a probabilistic model over the YIN difference function to estimate
    voiced probability alongside f0. Frame time resolution = hop_length / sr.
    """

    def estimate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        profile: InstrumentProfile,
        config: AppConfig,
    ) -> PitchResult:
        import math as _math
        import librosa  # lazy import — keeps startup fast if unused

        # pYIN requires enough samples to fit several full periods of fmin.
        # In practice, pYIN voiced probability is unreliable below ~4 periods.
        # We compute the frame length to fit 6 periods of fmin (conservative),
        # then round up to the next power of two (required by the FFT).
        #
        # Example: E2 (82.41 Hz) @ 44100 Hz → 6 periods = 3210 → next pow2 = 4096
        #          E1 (41.20 Hz) @ 44100 Hz → 6 periods = 6423 → next pow2 = 8192
        min_frames_for_fmin = _math.ceil(6.0 * sample_rate / profile.freq_min_hz)
        adaptive_frame_length = max(
            config.frame_length,
            2 ** _math.ceil(_math.log2(max(min_frames_for_fmin, 2))),
        )

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=profile.freq_min_hz,
            fmax=profile.freq_max_hz,
            sr=sample_rate,
            frame_length=adaptive_frame_length,
            hop_length=config.hop_length,
            fill_na=np.nan,
        )

        times = librosa.frames_to_time(
            np.arange(len(f0)),
            sr=sample_rate,
            hop_length=config.hop_length,
        )

        frames: list[PitchFrame] = []
        for t, freq, vp, vf in zip(times, f0, voiced_prob, voiced_flag, strict=True):
            freq_val = float(freq) if (vf and not np.isnan(freq)) else float("nan")
            frames.append(
                PitchFrame(
                    time_s=float(t),
                    freq_hz=freq_val,
                    confidence=float(vp),
                    is_voiced=bool(vf) and float(vp) >= config.confidence_threshold,
                )
            )

        return PitchResult(frames=frames, sample_rate=sample_rate, hop_length=config.hop_length)


# ---------------------------------------------------------------------------
# CREPE backend
# ---------------------------------------------------------------------------

class _CrepeBackend(_PitchBackend):
    """
    Wraps `torchcrepe` — the actively maintained PyTorch port of CREPE.

    The original TensorFlow `crepe` package (marl/crepe) has been unmaintained
    since 2022. `torchcrepe` uses the same pre-trained weights, identical
    accuracy, and is updated regularly (latest: May 2025). It also benefits
    from PyTorch's MPS backend on Apple Silicon.

    Install: pip install torchcrepe torch

    torchcrepe predicts pitch at hop_length intervals. Frequencies outside the
    instrument's physical range are masked and treated as unvoiced.
    """

    def estimate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        profile: InstrumentProfile,
        config: AppConfig,
    ) -> PitchResult:
        try:
            import torch
            import torchcrepe  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "CREPE backend requires: pip install torchcrepe torch\n"
                "Alternatively, set ATS_PITCH_BACKEND=pyin (no extra install needed)."
            ) from exc

        # torchcrepe expects a float32 tensor of shape (1, n_samples)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        # Use MPS on Apple Silicon if available, otherwise CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # torchcrepe.predict returns (times, pitch_hz, periodicity)
            # 'periodicity' is equivalent to voiced confidence
            pitch_hz, periodicity = torchcrepe.predict(
                audio_tensor,
                sample_rate,
                hop_length=config.hop_length,
                fmin=profile.freq_min_hz,
                fmax=profile.freq_max_hz,
                model="full",
                decoder=torchcrepe.decode.viterbi,
                return_periodicity=True,
                device=device,
            )

        # torchcrepe returns tensors of shape (1, n_frames) — squeeze to 1D
        pitch_arr = pitch_hz.squeeze(0).cpu().numpy()
        conf_arr = periodicity.squeeze(0).cpu().numpy()
        n_frames = len(pitch_arr)
        hop_s = config.hop_length / sample_rate
        times = np.arange(n_frames) * hop_s

        frames: list[PitchFrame] = []
        for t, freq, conf in zip(times, pitch_arr, conf_arr, strict=True):
            freq_f = float(freq)
            conf_f = float(conf)
            in_range = (
                profile.freq_min_hz <= freq_f <= profile.freq_max_hz
                and conf_f >= config.confidence_threshold
            )
            frames.append(
                PitchFrame(
                    time_s=float(t),
                    freq_hz=freq_f if in_range else float("nan"),
                    confidence=conf_f,
                    is_voiced=in_range,
                )
            )

        return PitchResult(frames=frames, sample_rate=sample_rate, hop_length=config.hop_length)


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------

class PitchEstimator:
    """
    Facade over pitch detection backends.

    Parameters
    ----------
    config : AppConfig
    backend : str | None
        'pyin' or 'crepe'. Defaults to config.pitch_backend.
    """

    _BACKENDS: dict[str, type[_PitchBackend]] = {
        "pyin": _PYinBackend,
        "crepe": _CrepeBackend,
    }

    def __init__(self, config: AppConfig, backend: str | None = None) -> None:
        self._config = config
        backend_name = (backend or config.pitch_backend).lower()
        if backend_name not in self._BACKENDS:
            raise ValueError(
                f"Unknown pitch backend {backend_name!r}. "
                f"Choose from: {list(self._BACKENDS)}"
            )
        self._backend: _PitchBackend = self._BACKENDS[backend_name]()
        self._backend_name = backend_name
        self._debug_rows: list[dict[str, float]] = []

    def estimate(self, audio: np.ndarray, profile: InstrumentProfile) -> PitchResult:
        """
        Estimate pitch for *audio* (1-D float32, mono).

        Parameters
        ----------
        audio : np.ndarray
            Raw audio samples. Should be at least config.frame_length samples.
        profile : InstrumentProfile
            Used to constrain frequency search range.

        Returns
        -------
        PitchResult
        """
        result = self._backend.estimate(
            audio=audio,
            sample_rate=self._config.sample_rate,
            profile=profile,
            config=self._config,
        )

        if self._config.debug_pitch:
            self._accumulate_debug(result)

        return result

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _accumulate_debug(self, result: PitchResult) -> None:
        for f in result.frames:
            self._debug_rows.append(
                {"time_s": f.time_s, "freq_hz": f.freq_hz, "confidence": f.confidence}
            )

    def flush_debug_csv(self, path: Path | None = None) -> None:
        """Write accumulated debug frames to CSV. Called on session end."""
        if not self._debug_rows:
            return
        dest = path or self._config.debug_pitch_csv if hasattr(self._config, "debug_pitch_csv") \
            else Path("/tmp/ats_pitch_debug.csv")
        with open(dest, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["time_s", "freq_hz", "confidence"])
            writer.writeheader()
            writer.writerows(self._debug_rows)
        print(f"[PitchEstimator] debug CSV written → {dest}")
        self._debug_rows.clear()

    @property
    def backend_name(self) -> str:
        return self._backend_name
