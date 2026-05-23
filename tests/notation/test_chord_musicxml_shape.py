"""Verify music21's exact MusicXML serialization shape for a Chord.

This is a prerequisite for tab_xml's chord-sibling injection (Task 10):
we need to know the exact element structure music21 produces so the
walker can recognise chord groups deterministically.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import music21.chord
import music21.meter.base
import music21.note
import music21.stream
import music21.tempo


def _make_chord_score(midi_notes: list[int]) -> music21.stream.Score:
    score = music21.stream.Score()
    part = music21.stream.Part()
    part.append(music21.meter.base.TimeSignature("4/4"))
    part.append(music21.tempo.MetronomeMark(number=120))
    if len(midi_notes) == 1:
        part.append(music21.note.Note(midi_notes[0]))
    else:
        part.append(music21.chord.Chord(midi_notes))
    score.append(part.makeMeasures())
    return score


def test_chord_serializes_with_chord_marker_on_siblings(tmp_path: Path):
    """A 3-note Chord produces 3 <note> elements (plus a rest); first has no <chord/>;
    siblings 2 and 3 each have a <chord/> child."""
    score = _make_chord_score([60, 64, 67])
    xml_path = tmp_path / "chord.musicxml"
    score.write("musicxml", fp=str(xml_path))

    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Get only the notes with pitch (filter out rests)
    notes = [n for n in root.iter("note") if n.find("pitch") is not None]
    # Chord produces 3 pitched notes
    assert len(notes) == 3, f"Expected 3 pitched <note> elements, got {len(notes)}"
    # First note: no <chord/> child
    assert notes[0].find("chord") is None
    # Subsequent notes: have <chord/> child
    for n in notes[1:]:
        assert n.find("chord") is not None, "expected <chord/> on sibling"


def test_singleton_note_has_no_chord_marker(tmp_path: Path):
    score = _make_chord_score([60])
    xml_path = tmp_path / "note.musicxml"
    score.write("musicxml", fp=str(xml_path))
    tree = ET.parse(xml_path)
    # Filter to only pitched notes (exclude rest)
    notes = [n for n in tree.getroot().iter("note") if n.find("pitch") is not None]
    assert len(notes) == 1
    assert notes[0].find("chord") is None


def test_chord_pitches_are_in_chord_pitch_order(tmp_path: Path, capsys):
    """Record music21's actual chord-pitch ordering (ascending or input).

    This test prints the ordering so Task 10 can rely on it.
    """
    score = _make_chord_score([67, 60, 64])   # constructed out of order
    xml_path = tmp_path / "chord_order.musicxml"
    score.write("musicxml", fp=str(xml_path))
    tree = ET.parse(xml_path)
    # Filter to only pitched notes (exclude rest)
    notes = [n for n in tree.getroot().iter("note") if n.find("pitch") is not None]
    pitches = []
    for n in notes:
        pitch_el = n.find("pitch")
        if pitch_el is not None:
            step_el = pitch_el.find("step")
            octave_el = pitch_el.find("octave")
            if step_el is not None and octave_el is not None:
                pitches.append(f"{step_el.text}{octave_el.text}")
    # Capture whatever music21 does so the next task can rely on it
    print(f"music21 chord pitch order (input [67, 60, 64]): {pitches}")
    # We expect THREE pitches captured
    assert len(pitches) == 3
