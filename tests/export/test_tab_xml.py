"""
tests/export/test_tab_xml.py

Tests for export.tab_xml.inject_tab_part().

MuseScore requires a single part with two linked staves (not two separate parts).
Staff 1 = standard notation, staff 2 = TAB.

Uses a minimal hardcoded MusicXML fixture so tests do not depend on music21
or MuseScore being installed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from improv_scribe.analysis.instrument_profiles import Instrument, get_profile
from improv_scribe.export.tab_xml import inject_tab_part
from improv_scribe.quantization.grid import NoteDuration, QuantizedNote

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
    path = tmp_path / "score.musicxml"
    path.write_text(_MINIMAL_MXL, encoding="utf-8")
    return path


@pytest.fixture
def mxl_file_with_rest(tmp_path: Path) -> Path:
    path = tmp_path / "score_rest.musicxml"
    path.write_text(_MINIMAL_MXL_WITH_REST, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests — single-part two-staves structure
# ---------------------------------------------------------------------------

class TestInjectTabPart:
    def test_still_one_part_after_injection(self, mxl_file, guitar_profile):
        """inject_tab_part must NOT add a second <part>; P1 is modified in-place."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        parts = root.findall(".//part")
        assert len(parts) == 1, f"Expected 1 part, got {len(parts)}"

    def test_staves_element_equals_two(self, mxl_file, guitar_profile):
        """First measure attributes must contain <staves>2</staves>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        staves_els = root.findall(".//measure/attributes/staves")
        assert len(staves_els) == 1
        assert staves_els[0].text == "2"

    def test_tab_clef_on_staff_two(self, mxl_file, guitar_profile):
        """First measure must contain <clef number='2'><sign>TAB</sign></clef>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        clefs = root.findall(".//measure/attributes/clef")
        tab_clefs = [c for c in clefs if c.get("number") == "2"]
        assert len(tab_clefs) == 1
        assert tab_clefs[0].find("sign").text == "TAB"

    def test_staff_details_on_staff_two(self, mxl_file, guitar_profile):
        """First measure must have <staff-details number='2'><staff-type>tab</staff-type>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        sd_els = root.findall(".//measure/attributes/staff-details[@number='2']")
        assert len(sd_els) == 1
        assert sd_els[0].find("staff-type").text == "tab"

    def test_notes_doubled_with_staff_1_and_2(self, mxl_file, guitar_profile):
        """Each original note must produce a staff-1 note and a staff-2 copy."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        all_notes = root.findall(".//part/measure/note")
        staff_values = [n.find("staff").text for n in all_notes if n.find("staff") is not None]

        assert staff_values.count("1") == 2, f"Expected 2 staff-1 notes, got {staff_values}"
        assert staff_values.count("2") == 2, f"Expected 2 staff-2 notes, got {staff_values}"

    def test_technical_annotation_on_staff_1_notes(self, mxl_file, guitar_profile):
        """Staff-1 non-rest notes must carry <technical><string>/<fret> annotations."""
        notes = [_make_note(64), _make_note(57)]
        # E4: string_idx=5 (highest guitar string), fret=0
        # A3: string_idx=4, fret=7
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        all_notes = root.findall(".//part/measure/note")
        staff1_notes = [n for n in all_notes if n.find("staff") is not None and n.find("staff").text == "1"]

        found_technical = []
        for note_el in staff1_notes:
            for tech in note_el.findall(".//notations/technical"):
                s = tech.find("string")
                f = tech.find("fret")
                if s is not None and f is not None:
                    found_technical.append((int(s.text), int(f.text)))

        # Guitar n_strings=6; E4 string_idx=5 → string_num=1, fret=0
        # A3 string_idx=4 → string_num=2, fret=7
        assert (1, 0) in found_technical, f"E4 annotation missing from {found_technical}"
        assert (2, 7) in found_technical, f"A3 annotation missing from {found_technical}"

    def test_rest_notes_have_no_technical(self, mxl_file_with_rest, guitar_profile):
        """Rest notes must not receive technical annotations on either staff."""
        notes = [_make_note(64), _make_rest(), _make_note(57)]
        assignments = [(5, 0), None, (4, 7)]

        inject_tab_part(mxl_file_with_rest, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file_with_rest)
        root = tree.getroot()
        for note_el in root.findall(".//part/measure/note"):
            if note_el.find("rest") is not None:
                technical_els = note_el.findall(".//notations/technical")
                assert len(technical_els) == 0, "Rest note should not have technical annotations"

    def test_original_clef_gets_number_one(self, mxl_file, guitar_profile):
        """The existing treble clef must be numbered 1 after injection."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        clefs = root.findall(".//measure/attributes/clef")
        numbered_clefs = {c.get("number"): c for c in clefs}
        assert "1" in numbered_clefs
        assert numbered_clefs["1"].find("sign").text == "G"

    def test_part_list_unchanged(self, mxl_file, guitar_profile):
        """<part-list> must still contain exactly one <score-part> after injection."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [(5, 0), (4, 7)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        score_parts = root.findall(".//part-list/score-part")
        assert len(score_parts) == 1, f"Expected 1 score-part, got {len(score_parts)}"
