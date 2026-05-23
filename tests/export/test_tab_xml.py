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

# MusicXML with a 3-note chord (E2/A2/D3) followed by a singleton (E4).
# Chord members are: first note has no <chord/>, subsequent have <chord/> first.
# Pitches ordered MIDI-ascending (music21 emission order).
_MINIMAL_MXL_WITH_CHORD = """\
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
        <pitch><step>E</step><octave>2</octave></pitch>
        <duration>2</duration>
        <type>half</type>
      </note>
      <note>
        <chord/>
        <pitch><step>A</step><octave>2</octave></pitch>
        <duration>2</duration>
        <type>half</type>
      </note>
      <note>
        <chord/>
        <pitch><step>D</step><octave>3</octave></pitch>
        <duration>2</duration>
        <type>half</type>
      </note>
      <note>
        <pitch><step>E</step><octave>4</octave></pitch>
        <duration>2</duration>
        <type>half</type>
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

def _make_note(midi: int | tuple[int, ...], ql: float = 1.0) -> QuantizedNote:
    midi_tuple = (midi,) if isinstance(midi, int) else midi
    n = len(midi_tuple)
    return QuantizedNote(
        midi_notes=midi_tuple,
        frequencies_hz=(0.0,) * n,
        confidences=(1.0,) * n,
        cents_deviations=(0.0,) * n,
        onset_beat=0.0,
        duration_beats=ql,
        duration_type=NoteDuration.QUARTER,
        quarter_length=ql,
        is_rest=False,
    )


def _make_rest(ql: float = 1.0) -> QuantizedNote:
    return QuantizedNote(
        midi_notes=(),
        frequencies_hz=(),
        confidences=(),
        cents_deviations=(),
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


@pytest.fixture
def mxl_file_with_chord(tmp_path: Path) -> Path:
    path = tmp_path / "score_chord.musicxml"
    path.write_text(_MINIMAL_MXL_WITH_CHORD, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests — single-part two-staves structure
# ---------------------------------------------------------------------------

class TestInjectTabPart:
    def test_still_one_part_after_injection(self, mxl_file, guitar_profile):
        """inject_tab_part must NOT add a second <part>; P1 is modified in-place."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [((5, 0),), ((4, 7),)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        parts = root.findall(".//part")
        assert len(parts) == 1, f"Expected 1 part, got {len(parts)}"

    def test_staves_element_equals_two(self, mxl_file, guitar_profile):
        """First measure attributes must contain <staves>2</staves>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [((5, 0),), ((4, 7),)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        staves_els = root.findall(".//measure/attributes/staves")
        assert len(staves_els) == 1
        assert staves_els[0].text == "2"

    def test_tab_clef_on_staff_two(self, mxl_file, guitar_profile):
        """First measure must contain <clef number='2'><sign>TAB</sign></clef>."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [((5, 0),), ((4, 7),)]

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
        assignments = [((5, 0),), ((4, 7),)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        sd_els = root.findall(".//measure/attributes/staff-details[@number='2']")
        assert len(sd_els) == 1
        assert sd_els[0].find("staff-type").text == "tab"

    def test_notes_doubled_with_staff_1_and_2(self, mxl_file, guitar_profile):
        """Each original note must produce a staff-1 note and a staff-2 copy."""
        notes = [_make_note(64), _make_note(57)]
        assignments = [((5, 0),), ((4, 7),)]

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
        assignments = [((5, 0),), ((4, 7),)]

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
        assignments = [((5, 0),), None, ((4, 7),)]

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
        assignments = [((5, 0),), ((4, 7),)]

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
        assignments = [((5, 0),), ((4, 7),)]

        inject_tab_part(mxl_file, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file)
        root = tree.getroot()
        score_parts = root.findall(".//part-list/score-part")
        assert len(score_parts) == 1, f"Expected 1 score-part, got {len(score_parts)}"

    def test_chord_emits_chord_siblings_on_staff_2(
        self, mxl_file_with_chord, guitar_profile
    ):
        """A 3-note chord followed by a singleton produces the correct staff-2 structure.

        Staff 1: 3 chord-sibling <note>s + 1 standalone <note>
        Staff 2: 3 chord-sibling <note>s + 1 standalone <note>,
                 all with <technical><string><fret> annotations.

        Guitar TUNING[0..2] = [40, 45, 50] (E2, A2, D3).
        Chord assignment (string_ascending): ((0, 0), (1, 0), (2, 0))
          → implied MIDIs: 40 (E2), 45 (A2), 50 (D3) — same as XML pitch order.
        Singleton: ((5, 0),) → E4 on string 0 (highest string, string_num=1).
        """
        # Chord: E2/A2/D3 → open strings 0/1/2
        chord_assignment: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (2, 0))
        # Singleton: E4 on highest guitar string (idx=5)
        singleton_assignment: tuple[tuple[int, int], ...] = ((5, 0),)
        notes = [
            _make_note((40, 45, 50), ql=2.0),
            _make_note(64, ql=2.0),
        ]
        assignments = [chord_assignment, singleton_assignment]

        inject_tab_part(mxl_file_with_chord, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file_with_chord)
        root = tree.getroot()

        # Collect all staff-2 pitched notes (non-rest).
        staff2_pitched = [
            n for n in root.iter("note")
            if n.find("staff") is not None
            and n.find("staff").text == "2"
            and n.find("pitch") is not None
        ]

        # 3 chord members + 1 singleton = 4 pitched staff-2 notes.
        assert len(staff2_pitched) == 4, (
            f"Expected 4 staff-2 pitched notes, got {len(staff2_pitched)}"
        )
        # First chord member: no <chord/>.
        assert staff2_pitched[0].find("chord") is None, (
            "First staff-2 chord member must NOT have <chord/>"
        )
        # Second and third chord members: must have <chord/>.
        assert staff2_pitched[1].find("chord") is not None, (
            "Second staff-2 chord member must have <chord/>"
        )
        assert staff2_pitched[2].find("chord") is not None, (
            "Third staff-2 chord member must have <chord/>"
        )
        # Singleton: no <chord/>.
        assert staff2_pitched[3].find("chord") is None, (
            "Singleton staff-2 note must NOT have <chord/>"
        )

        # All 4 staff-2 notes must carry <technical><string>/<fret>.
        for i, note_el in enumerate(staff2_pitched):
            tech = note_el.find(".//notations/technical")
            assert tech is not None, f"staff-2 note {i} has no <technical>"
            assert tech.find("string") is not None, f"staff-2 note {i} has no <string>"
            assert tech.find("fret") is not None, f"staff-2 note {i} has no <fret>"

        # Staff-1 chord notes must also carry technical annotations.
        staff1_pitched = [
            n for n in root.iter("note")
            if n.find("staff") is not None
            and n.find("staff").text == "1"
            and n.find("pitch") is not None
        ]
        assert len(staff1_pitched) == 4, (
            f"Expected 4 staff-1 pitched notes, got {len(staff1_pitched)}"
        )
        for i, note_el in enumerate(staff1_pitched):
            tech = note_el.find(".//notations/technical")
            assert tech is not None, f"staff-1 note {i} has no <technical>"

    def test_chord_midi_ordering_maps_correctly(
        self, mxl_file_with_chord, guitar_profile
    ):
        """Assignment pairs map to XML notes in MIDI-ascending order (not string order).

        For the natural voicing used in _MINIMAL_MXL_WITH_CHORD:
          chord_assignment = ((0, 0), (1, 0), (2, 0))  ← string ascending
          Implied MIDIs     =   40,     45,     50     ← also MIDI ascending
          XML notes (MIDI-asc) = E2(40), A2(45), D3(50)
        String-to-string-num conversion: n_strings(6) - string_idx.
          string_idx=0 → string_num=6 (low E string in TAB notation)
          string_idx=1 → string_num=5
          string_idx=2 → string_num=4
        All three chord staff-2 notes should have string_num ∈ {6, 5, 4}.
        """
        chord_assignment: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (2, 0))
        singleton_assignment: tuple[tuple[int, int], ...] = ((5, 0),)
        notes = [_make_note((40, 45, 50), ql=2.0), _make_note(64, ql=2.0)]
        assignments = [chord_assignment, singleton_assignment]

        inject_tab_part(mxl_file_with_chord, notes, assignments, guitar_profile)

        tree = ET.parse(mxl_file_with_chord)
        root = tree.getroot()

        staff2_pitched = [
            n for n in root.iter("note")
            if n.find("staff") is not None
            and n.find("staff").text == "2"
            and n.find("pitch") is not None
        ]
        # First 3 are the chord group.
        chord_string_nums = set()
        for note_el in staff2_pitched[:3]:
            tech = note_el.find(".//notations/technical")
            assert tech is not None
            chord_string_nums.add(int(tech.find("string").text))

        assert chord_string_nums == {4, 5, 6}, (
            f"Expected string nums {{4,5,6}} for open E2/A2/D3, got {chord_string_nums}"
        )
