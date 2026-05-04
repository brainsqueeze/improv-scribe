"""
export/pdf_exporter.py — Renders a music21 Score to PDF via MuseScore CLI.

MuseScore is used as the rendering backend because it produces publication-quality
engraving and handles guitar clef (8vb treble) and bass clef correctly.

The pipeline is:
    music21.Score  →  MusicXML (temp file)  →  MuseScore CLI  →  PDF

MuseScore 4 CLI reference:
    mscore --export-to output.pdf input.mxl

If MuseScore is not found, raises MuseScoreNotFoundError with installation hints.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import music21.stream

from improv_scribe.config import AppConfig

if TYPE_CHECKING:
    from improv_scribe.analysis.instrument_profiles import InstrumentProfile
    from improv_scribe.quantization.grid import QuantizedNote


class MuseScoreNotFoundError(RuntimeError):
    """Raised when the MuseScore CLI binary cannot be located."""


class PDFExporter:
    """
    Exports a music21 Score to a PDF file using MuseScore as the renderer.

    Parameters
    ----------
    config : AppConfig
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._mscore_path = self._resolve_musescore(config.musescore_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        score: music21.stream.Score,
        output_path: Path,
        tab_notes: list[QuantizedNote] | None = None,
        tab_assignments: list[tuple[int, int] | None] | None = None,
        tab_profile: InstrumentProfile | None = None,
    ) -> Path:
        """
        Render *score* to PDF at *output_path*.

        Parameters
        ----------
        score : music21.stream.Score
        output_path : Path
            Destination PDF path. Parent directory will be created if needed.
        tab_notes : list[QuantizedNote] | None
            When provided (along with tab_assignments and tab_profile), injects
            a tablature Part into the intermediate MusicXML before rendering.
        tab_assignments : list[tuple[int, int] | None] | None
            Fret assignments parallel to tab_notes; see tab_xml.inject_tab_part.
        tab_profile : InstrumentProfile | None
            Instrument profile for tab string/tuning info.

        Returns
        -------
        Path
            Resolved output path of the written PDF.
        """
        output_path = Path(output_path).with_suffix(".pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            mxl_path = Path(tmpdir) / "score.musicxml"
            self._write_musicxml(score, mxl_path)

            if (
                tab_notes is not None
                and tab_assignments is not None
                and tab_profile is not None
            ):
                from improv_scribe.export.tab_xml import inject_tab_part

                inject_tab_part(mxl_path, tab_notes, tab_assignments, tab_profile)

            self._run_musescore(mxl_path, output_path)

        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_musicxml(self, score: music21.stream.Score, path: Path) -> None:
        """Serialize score to MusicXML."""
        score.write("musicxml", fp=str(path))

    def _run_musescore(self, input_mxl: Path, output_pdf: Path) -> None:
        """Invoke MuseScore CLI to convert MusicXML → PDF."""
        cmd = [
            self._mscore_path,
            "--export-to", str(output_pdf),
            str(input_mxl),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("MuseScore timed out after 60 seconds.") from exc
        except FileNotFoundError as exc:
            raise MuseScoreNotFoundError(
                f"MuseScore binary not found at {self._mscore_path!r}. "
                "Install via: brew install musescore\n"
                "Or set ATS_MUSESCORE_PATH env var."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"MuseScore exited with code {result.returncode}.\n"
                f"stderr: {result.stderr[:2000]}"
            )

    @staticmethod
    def _resolve_musescore(configured_path: str) -> str:
        """Return the MuseScore binary path, checking PATH first."""
        # Try PATH first (e.g. if installed via brew and linked)
        which = shutil.which("mscore") or shutil.which("musescore")
        if which:
            return which
        # Fall back to configured path
        if Path(configured_path).exists():
            return configured_path
        # Return configured path anyway — will fail with clear error at export time
        return configured_path

    @property
    def musescore_path(self) -> str:
        return self._mscore_path
