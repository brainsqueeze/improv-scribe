"""
notation/score_builder.py — Converts QuantizedNotes into a music21 Score.

The resulting Score object can be:
  - Exported to MusicXML (music21 native)
  - Rendered to PDF via MuseScore CLI (see export/pdf_exporter.py)
  - Written to MIDI (see export/midi_exporter.py)
  - Displayed interactively via score.show()

Guitar transposition
--------------------
Guitar is conventionally written one octave above sounding pitch (treble8vb /
bass8vb clef).  We rely on the *clef* alone to communicate the octave offset
to MuseScore: notes are written at concert (sounding) MIDI, and the clef's
``<clef-octave-change>`` handles visual placement.

We do NOT pre-shift pitch or emit a ``<transpose>`` element.  Combining
``<clef-octave-change>`` with ``<transpose>`` causes MuseScore to mis-compute
TAB fret positions (it overrides explicit ``<technical><fret>`` annotations
when redundant transposition signals are present).

Tablature
---------
Tab output is NOT included in the Score returned by build().  Instead, call
compute_tab_assignments() to get fret assignments, then pass those to
export.tab_xml.inject_tab_part() to post-process the MusicXML directly.
This bypasses music21's incomplete tablature support entirely.
"""

from __future__ import annotations

import music21.chord
import music21.clef
import music21.instrument
import music21.metadata
import music21.meter
import music21.note
import music21.stream
import music21.tempo
from music21.duration import Duration
from music21.meter.base import TimeSignature

from improv_scribe.analysis.instrument_profiles import Instrument, InstrumentProfile
from improv_scribe.quantization.grid import QuantizedNote
from improv_scribe.quantization.tempo import TempoResult

# Map our instrument enum to music21 instrument objects
_INSTRUMENT_MAP: dict[Instrument, type[music21.instrument.Instrument]] = {
    Instrument.GUITAR: music21.instrument.Guitar,
    Instrument.BASS: music21.instrument.ElectricBass,
}


class ScoreBuilder:
    """
    Builds a music21 Score from a list of QuantizedNotes.

    Parameters
    ----------
    profile : InstrumentProfile
    tempo_result : TempoResult
    time_signature : tuple[int, int]
    title : str
    include_tab : bool
        When True (default), tab assignments will be available via
        compute_tab_assignments().  Does NOT affect the Score returned by
        build() — that is always a single-part standard notation Score.
    """

    def __init__(
        self,
        profile: InstrumentProfile,
        tempo_result: TempoResult,
        time_signature: tuple[int, int] = (4, 4),
        title: str = "Transcription",
        include_tab: bool = True,
    ) -> None:
        self._profile = profile
        self._tempo_result = tempo_result
        self._time_sig = time_signature
        self._title = title
        self._include_tab = include_tab

    def build(self, notes: list[QuantizedNote]) -> music21.stream.Score:
        """
        Build and return a music21 Score (single standard notation Part).

        Parameters
        ----------
        notes : list[QuantizedNote]
            Sorted by onset_beat ascending. May include rests (is_rest=True).

        Returns
        -------
        music21.stream.Score
            Always a single-part Score.  Tab injection is done separately via
            export.tab_xml.inject_tab_part().
        """
        score = music21.stream.Score()

        # Metadata
        meta = music21.metadata.Metadata()
        meta.title = self._title
        score.insert(0, meta)

        # Single part
        part = music21.stream.Part()

        # Instrument — disable music21's built-in transposition.  Guitar's
        # default music21 transposition would emit a <transpose> element that
        # conflicts with the treble8vb clef's <clef-octave-change>.  We rely on
        # the clef alone for octave display.
        m21_instrument_cls = _INSTRUMENT_MAP.get(
            self._profile.instrument, music21.instrument.Guitar
        )
        inst = m21_instrument_cls()
        inst.transposition = None
        part.insert(0, inst)

        # Clef
        clef_obj = music21.clef.clefFromString(self._profile.clef)
        part.append(clef_obj)

        # Time signature
        ts = TimeSignature(f"{self._time_sig[0]}/{self._time_sig[1]}")
        part.append(ts)

        # Tempo mark — round to nearest integer; librosa's beat_track returns floats
        mm = music21.tempo.MetronomeMark(number=round(self._tempo_result.bpm))
        part.append(mm)

        # Notes and rests at concert (sounding) pitch.  The treble8vb /
        # bass8vb clef carries the octave-display offset; no manual shift here.
        for qn in notes:
            dur = Duration(quarterLength=qn.quarter_length)
            if qn.is_rest:
                element: music21.note.GeneralNote = music21.note.Rest(duration=dur)
            elif len(qn.midi_notes) == 1:
                element = music21.note.Note(qn.midi_notes[0], duration=dur)
            else:
                element = music21.chord.Chord(list(qn.midi_notes), duration=dur)

            part.append(element)

        # Make measures from the flat stream
        part_with_measures = part.makeMeasures()
        if part_with_measures is None:
            raise AssertionError("`part_with_measures` cannot be None")
        part_with_measures.makeBeams(inPlace=True)

        score.append(part_with_measures)

        return score

    def compute_tab_assignments(
        self, notes: list[QuantizedNote]
    ) -> list[tuple[int, int] | None]:
        """
        Return fret assignments for notes; None entries for rests.

        Only call when include_tab is True.

        Parameters
        ----------
        notes : list[QuantizedNote]
            Same note list used to build the score.

        Returns
        -------
        list[tuple[int, int] | None]
            Parallel to *notes*. Each non-rest entry is (string_idx, fret)
            where string_idx is 0-based from the lowest string.
            None for rests.
        """
        if not self._include_tab:
            raise RuntimeError(
                "compute_tab_assignments() called but include_tab=False"
            )
        from improv_scribe.notation.tab_builder import assign_frets

        return assign_frets(notes, self._profile.instrument)

    def build_raw(self, notes: list[QuantizedNote]) -> music21.stream.Score:
        """
        Build a Score without measure-level quantization.
        Used for raw-timing MIDI export only — no PDF rendering.
        """
        score = music21.stream.Score()
        part = music21.stream.Part()
        part.insert(0, music21.tempo.MetronomeMark(number=self._tempo_result.bpm))

        for qn in notes:
            if qn.is_rest:
                continue
            if len(qn.midi_notes) == 1:
                el: music21.note.GeneralNote = music21.note.Note(qn.midi_notes[0])
            else:
                el = music21.chord.Chord(list(qn.midi_notes))
            el.quarterLength = qn.quarter_length
            el.offset = qn.onset_beat
            part.insert(el.offset, el)

        score.append(part)
        return score
