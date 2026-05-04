"""
tests/export/test_tab_xml.py

Tests for export.tab_xml.inject_tab_part().

Uses a minimal hardcoded MusicXML fixture so tests do not depend on music21
or MuseScore being installed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from audio_to_sheet.analysis.instrument_profiles import Instrument, get_profile
from audio_to_sheet.export.tab_xml import inject_tab_part
from audio_to_sheet.quantization.grid import NoteDuration, QuantizedNote

# ---------------------------------------------------------------------------
# Minimal MusicXML fixture
# One measure, two notes (E4 and A3), no namespace.
# ---------------------------------------------------------------------------

_MINIMAL_MXL = """\
<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Guitar</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>1</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>1</duration>
        <type>quarter</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""

# MusicXML with a rest note included (for matching logic tests)
_MINIMAL_MXL_WITH_REST = """\
<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Guitar</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>1</duration>
        <type>quarter</type>
      </note>
      <note>
        <rest/>
        <duration>1</duration>
        <type>quarter</type>
      </note>
      <note>
        <pitch><step>A</step><octave>3</octave></pitch>
        <duration>1</duration>
        <type>quarter</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_note(midi: int, ql: float = 1.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=midi,
        frequency_hz=0.0,
        confidence=1.0,
        cents_deviation=0.0,
        onset_beat=0.0,
        duration_beats=ql,
        duration_type=NoteDuration.QUARTER,
        quarter_length=ql,
        is_rest=False,
    )


def _make_rest(ql: float = 1.0) -> QuantizedNote:
    return QuantizedNote(
        midi_note=0,
        frequency_hz=0.0,
        confidence=1.0,
        cents_deviation=0.0,
        onset_beat=0.0,
        duration_beats=ql,
        duration_type=NoteDuration.QUARTER,
        quarter_length=ql,
        is_rest=True,
    )


@pytest.fixture
def guitar_profile():
    return get_profile(Instrument.GUITAR)


@pytest.fixture
def mxl_file(tmp_path: Path) -> Path:
    """Write minimal MusicXML to a temp file and return its path."""
    path = tmp_path / "score.musicxml"
    path.write_text(_MINIMAL_MXL, encoding="utf-8")
    return path


@pytest.fixture
def mxl_file_with_rest(tmp_path: Path) -> Path:
    """Write minimal MusicXML with a rest to a temp file and return its path."""
    path = tmp_path / "score_rest.musicxml"
    path.write_text(_MINIMAL_MXL_WITH_REST, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInjectTabPart:
    def test_inject_creates_p2_tab_part(self, mxl_file, guitar_profile):
        """After injection, root must contain a <part id='P2-Tab'> element."""
        # E4 (midi 64) → string 1 fret 0; A3 (midi 57) → string 2 fret 7
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]  # (string_idx, fret) — string_idx 0=lowest

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        parts = root.findall(".//part")
        ids = [p.get("id") for p in parts]
        assert "P2-Tab" in ids, f"Expected 'P2-Tab' in part ids, got: {ids}"

    def test_inject_adds_tab_clef(self, mxl_file, guitar_profile):
        """P2-Tab's first measure must contain <clef><sign>TAB</sign></clef>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        p2 = root.find(".//part[@id='P2-Tab']")
        assert p2 is not None

        first_measure = p2.find("measure")
        assert first_measure is not None

        clef_signs = [
            sign.text
            for sign in first_measure.findall(".//clef/sign")
        ]
        assert "TAB" in clef_signs, f"No TAB sign found in clef elements: {clef_signs}"

    def test_inject_adds_staff_details(self, mxl_file, guitar_profile):
        """P2-Tab's first measure must have <staff-details><staff-type>tab</staff-type>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        p2 = root.find(".//part[@id='P2-Tab']")
        first_measure = p2.find("measure")

        staff_type_els = first_measure.findall(".//staff-details/staff-type")
        assert len(staff_type_els) == 1
        assert staff_type_els[0].text == "tab"

    def test_inject_adds_technical_fret_string(self, mxl_file, guitar_profile):
        """
        Non-rest notes in P2-Tab must have <notations><technical><string> and <fret>.
        """
        notes = [_make_note(64), _make_note(57)]
        # E4: string_idx=5 (highest guitar string), fret=0
        # A3: string_idx=4, fret=7
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        p2 = root.find(".//part[@id='P2-Tab']")
        p2_notes = p2.findall(".//note")

        # Collect all (string, fret) technical annotations found
        found_technical = []
        for note_el in p2_notes:
            for tech in note_el.findall(".//notations/technical"):
                s = tech.find("string")
                f = tech.find("fret")
                if s is not None and f is not None:
                    found_technical.append((int(s.text), int(f.text)))

        assert len(found_technical) == 2, (
            f"Expected 2 technical annotations, got {len(found_technical)}"
        )

        # Guitar: n_strings=6; string_num = n_strings - string_idx
        # E4: string_idx=5 → string_num=6-5=1, fret=0
        # A3: string_idx=4 → string_num=6-4=2, fret=7
        assert (1, 0) in found_technical, f"E4 annotation not found in {found_technical}"
        assert (2, 7) in found_technical, f"A3 annotation not found in {found_technical}"

    def test_inject_preserves_p1(self, mxl_file, guitar_profile):
        """After injection, P1 must remain unchanged (same note count and pitch content)."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        # Capture P1 state before injection
        tree_before = ET.parse(mxl_file)
        p1_before = tree_before.find(".//part[@id='P1']")
        notes_before = p1_before.findall(".//note")

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree_after = ET.parse(mxl_file)
        p1_after = tree_after.find(".//part[@id='P1']")
        assert p1_after is not None, "P1 was removed after injection"

        notes_after = p1_after.findall(".//note")
        assert len(notes_after) == len(notes_before), (
            f"P1 note count changed: {len(notes_before)} → {len(notes_after)}"
        )

        # Verify pitch content of first note (E4) is unchanged
        first_note = notes_after[0]
        assert first_note.find("pitch/step").text == "E"
        assert first_note.find("pitch/octave").text == "4"

    def test_inject_rest_notes_have_no_technical(self, mxl_file_with_rest, guitar_profile):
        """Rest notes in P2-Tab must not receive technical annotations."""
        # 3 elements: note, rest, note
        notes = [_make_note(64), _make_rest(), _make_note(57)]
        assignments = [(5, 0), None, (4, 7)]

        inject_tab_part(mxl_file_with_rest, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file_with_rest)
        root = tree.getroot()
        p2 = root.find(".//part[@id='P2-Tab']")

        # Find the rest note in P2 and confirm it has no <notations><technical>
        for note_el in p2.findall(".//note"):
            if note_el.find("rest") is not None:
                technical_els = note_el.findall(".//notations/technical")
                assert len(technical_els) == 0, (
                    "Rest note should not have technical annotations"
                )

    def test_inject_adds_score_part_to_part_list(self, mxl_file, guitar_profile):
        """After injection, <part-list> must contain a <score-part id='P2-Tab'>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        part_list = root.find("part-list")
        assert part_list is not None

        score_parts = part_list.findall("score-part")
        ids = [sp.get("id") for sp in score_parts]
        assert "P2-Tab" in ids, f"P2-Tab not in part-list score-parts: {ids}"
