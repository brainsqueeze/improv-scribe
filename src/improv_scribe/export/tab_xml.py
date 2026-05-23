"""
export/tab_xml.py — Injects linked tablature staves into an existing MusicXML file.

MuseScore requires a single part with two staves for proper notation+TAB rendering:
  - Staff 1: standard notation (existing)
  - Staff 2: tablature (injected)

MusicXML structure produced:
  <part id="P1">
    <measure number="1">
      <attributes>
        <staves>2</staves>
        <clef number="1">...</clef>        ← existing, gets number="1"
        <clef number="2"><sign>TAB</sign></clef>   ← injected
        <staff-details number="2">         ← injected
          <staff-type>tab</staff-type>
          <staff-lines>6</staff-lines>
          <staff-tuning line="1">...</staff-tuning>
          ...
        </staff-details>
      </attributes>
      <note>                        ← existing note, gains <staff>1</staff> + <technical>
        <staff>1</staff>
        <notations><technical><string>N</string><fret>M</fret></technical></notations>
      </note>
      <note>                        ← new staff-2 copy
        <staff>2</staff>
        <notations><technical><string>N</string><fret>M</fret></technical></notations>
      </note>
    </measure>
  </part>

Chord support (Phase 2)
-----------------------
A MusicXML chord is represented as a group of consecutive ``<note>`` elements where
the first has no ``<chord/>`` child and subsequent members have ``<chord/>`` as their
first child.  inject_tab_part groups these into *slots* so that one assignment tuple
of length N is consumed per N-note chord (or 1 for a monophonic note/rest).

MIDI-ordering invariant
-----------------------
music21 emits chord pitches in MIDI-ascending order (Task 9 finding).  assign_frets
returns shapes sorted by string ascending.  For natural voicings these orderings
match, but they can diverge for non-standard voicings.  _order_assignment_by_midi
sorts the assignment tuple by implied MIDI (tuning[string_idx] + fret) before it is
zipped onto the XML notes, guaranteeing correct mapping regardless of voicing.
"""

from __future__ import annotations

import contextlib
import copy
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from improv_scribe.analysis.instrument_profiles import Instrument, InstrumentProfile
from improv_scribe.quantization.grid import QuantizedNote

# ---------------------------------------------------------------------------
# Tuning data for <staff-tuning> elements.
#
# MusicXML spec: <staff-tuning line="N"> numbers staff lines from the BOTTOM
# up (line 1 = bottom).  In standard guitar/bass tab the bottom line is the
# lowest-pitched string, so tunings are ordered low → high.  Values are
# sounding pitch.
# ---------------------------------------------------------------------------

# Guitar: line 1 (bottom, low E2) → line 6 (top, high E4)
GUITAR_STAFF_TUNING: list[tuple[str, int]] = [
    ("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4),
]

# Bass: line 1 (bottom, low E1) → line 4 (top, high G2)
BASS_STAFF_TUNING: list[tuple[str, int]] = [
    ("E", 1), ("A", 1), ("D", 2), ("G", 2),
]

_STAFF_TUNING: dict[Instrument, list[tuple[str, int]]] = {
    Instrument.GUITAR: GUITAR_STAFF_TUNING,
    Instrument.BASS: BASS_STAFF_TUNING,
}

# MIDI pitch of each open string, ordered low-to-high (index == string_idx).
# These must mirror the values in notation/tab_builder.py so that the implied
# MIDI computation (tuning[string_idx] + fret) round-trips correctly.
_MIDI_TUNING: dict[Instrument, list[int]] = {
    Instrument.GUITAR: [40, 45, 50, 55, 59, 64],  # E2 A2 D3 G3 B3 E4
    Instrument.BASS:   [28, 33, 38, 43],           # E1 A1 D2 G2
}

_DEFAULT_MIDI_TUNING: list[int] = [40, 45, 50, 55, 59, 64]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_tab_part(
    mxl_path: Path,
    notes: list[QuantizedNote],
    assignments: list[tuple[tuple[int, int], ...] | None],
    profile: InstrumentProfile,
) -> None:
    """Modify *mxl_path* in-place to add a linked TAB staff to the existing notation staff.

    The existing P1 part is modified to contain two staves:
    staff 1 (notation) and staff 2 (TAB).  Each non-rest note gains a
    ``<staff>1</staff>`` element plus ``<technical><string>/<fret>`` annotations,
    and a staff-2 copy is inserted immediately after.

    Chord support: a QuantizedNote whose ``midi_notes`` has length N produces N
    consecutive ``<note>`` elements in the MusicXML file (first without ``<chord/>``,
    siblings 2–N with ``<chord/>``).  All N elements share one assignment tuple of
    length N emitted by ``assign_frets()``.

    Parameters
    ----------
    mxl_path : Path
        Path to an existing MusicXML file (will be overwritten).
    notes : list[QuantizedNote]
        The same note list used to build the score (in order).
    assignments : list[tuple[tuple[int, int], ...] | None]
        Fret assignments from tab_builder.assign_frets(). Parallel to notes.
        Each non-rest entry is a tuple of (string_idx, fret) pairs — one per
        chord member, sorted by string ascending.  None for rests.
    profile : InstrumentProfile
        Used for string count and tuning info.
    """
    tree = ET.parse(mxl_path)
    root = tree.getroot()

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def tag(name: str) -> str:
        return f"{ns}{name}"

    p1 = root.find(f".//{tag('part')}")
    if p1 is None:
        raise ValueError("MusicXML file has no <part> element.")

    tuning_data = _STAFF_TUNING.get(profile.instrument, GUITAR_STAFF_TUNING)
    midi_tuning = _MIDI_TUNING.get(profile.instrument, _DEFAULT_MIDI_TUNING)
    n_strings = len(tuning_data)

    # -----------------------------------------------------------------------
    # Build slot_assignment_map: id(first_note_in_slot) → assignment tuple.
    #
    # Walk measures grouping <note> elements into chord-slots, then consume
    # one assignment entry per slot.  Rests produce None-assignment slots;
    # tie continuations re-use the previous non-rest assignment so the TAB
    # staff shows the same fret for the full tied duration.
    # -----------------------------------------------------------------------
    non_rest_assignments: list[tuple[tuple[int, int], ...]] = [
        a for a in assignments if a is not None
    ]
    assignment_iter = iter(non_rest_assignments)

    # Map from id(first_note_el_in_slot) → MIDI-ordered assignment tuple.
    slot_assignment_map: dict[int, tuple[tuple[int, int], ...]] = {}
    # Map from id(any_note_el) → assignment tuple (used for tie continuations).
    note_assignment_map: dict[int, tuple[tuple[int, int], ...]] = {}
    current_assignment: tuple[tuple[int, int], ...] | None = None

    for measure_el in p1.findall(tag("measure")):
        slots = _group_notes_into_slots(measure_el, tag)
        for slot in slots:
            first_note = slot[0]
            is_rest = first_note.find(tag("rest")) is not None
            if is_rest:
                # Rests never appear mid-tie; don't reset current_assignment.
                continue
            is_tie_continuation = any(
                t.get("type") == "stop"
                for t in first_note.findall(tag("tie"))
            )
            if is_tie_continuation:
                if current_assignment is not None:
                    midi_ordered = _order_assignment_by_midi(
                        current_assignment, midi_tuning
                    )
                    slot_assignment_map[id(first_note)] = midi_ordered
                    for note_el in slot:
                        note_assignment_map[id(note_el)] = midi_ordered
            else:
                raw = next(assignment_iter, None)
                if raw is not None:
                    current_assignment = raw
                    midi_ordered = _order_assignment_by_midi(raw, midi_tuning)
                    slot_assignment_map[id(first_note)] = midi_ordered
                    for note_el in slot:
                        note_assignment_map[id(note_el)] = midi_ordered

    # -----------------------------------------------------------------------
    # Process the first measure: inject <staves>, tab clef, staff-details.
    # -----------------------------------------------------------------------
    first_measure = p1.find(tag("measure"))
    if first_measure is None:
        raise ValueError("MusicXML part has no measures.")

    attrs_el = first_measure.find(tag("attributes"))
    if attrs_el is None:
        attrs_el = ET.Element(tag("attributes"))
        first_measure.insert(0, attrs_el)

    _inject_two_staves_attributes(attrs_el, tuning_data, tag)

    # -----------------------------------------------------------------------
    # For every measure: annotate staff-1 notes and insert staff-2 copies.
    # -----------------------------------------------------------------------
    for measure_el in p1.findall(tag("measure")):
        _annotate_measure(
            measure_el, note_assignment_map, n_strings, tag, midi_tuning
        )

    tree.write(str(mxl_path), encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _group_notes_into_slots(
    measure_el: ET.Element,
    tag: Callable[[str], str],
) -> list[list[ET.Element]]:
    """Group consecutive ``<note>`` children of *measure_el* into chord-slots.

    A slot begins at a ``<note>`` without a ``<chord/>`` child and includes any
    immediately following ``<note>`` elements whose first child is ``<chord/>``.
    Non-``<note>`` children (``<backup>``, ``<attributes>``, etc.) are ignored.

    Parameters
    ----------
    measure_el : ET.Element
        A MusicXML ``<measure>`` element.
    tag : Callable[[str], str]
        Namespace-aware tag builder.

    Returns
    -------
    list[list[ET.Element]]
        Each inner list is one slot (1 element for a monophonic note/rest,
        N elements for an N-member chord).
    """
    slots: list[list[ET.Element]] = []
    current: list[ET.Element] = []
    for child in list(measure_el):
        if child.tag != tag("note"):
            continue
        # A chord-sibling has <chord/> as its first *element* child.
        first_child = next((c for c in child if c.tag == tag("chord")), None)
        is_chord_sibling = first_child is not None
        if is_chord_sibling and current:
            current.append(child)
        else:
            if current:
                slots.append(current)
            current = [child]
    if current:
        slots.append(current)
    return slots


def _order_assignment_by_midi(
    assignment: tuple[tuple[int, int], ...],
    tuning: list[int],
) -> tuple[tuple[int, int], ...]:
    """Reorder a (string, fret) assignment tuple to MIDI-ascending order.

    music21 emits chord pitches in input order (MIDI-ascending, since
    ``qn.midi_notes`` is sorted ascending).  assign_frets returns shapes sorted
    by string ascending.  For natural voicings these orderings align; for unusual
    voicings (e.g. low MIDI pitch on a high string number) they can diverge.
    This function sorts by implied MIDI (``tuning[string_idx] + fret``) so the
    resulting tuple zips correctly onto the XML ``<note>`` sequence.

    Parameters
    ----------
    assignment : tuple[tuple[int, int], ...]
        (string_idx, fret) pairs, arbitrarily ordered.
    tuning : list[int]
        MIDI pitch of each open string (index == string_idx), low to high.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Same pairs sorted by implied MIDI ascending.
    """
    return tuple(sorted(assignment, key=lambda sf: tuning[sf[0]] + sf[1]))


def _inject_two_staves_attributes(
    attrs_el: ET.Element,
    tuning_data: list[tuple[str, int]],
    tag: Callable[[str], str],
) -> None:
    """Add ``<staves>2</staves>``, a numbered tab ``<clef>``, and ``<staff-details>``.

    The score is written at concert pitch with a ``treble8vb`` / ``bass8vb`` clef
    that carries the octave-display offset via ``<clef-octave-change>``.  No
    ``<transpose>`` element is emitted: combining ``<transpose>`` with
    ``<clef-octave-change>`` causes MuseScore to mis-compute TAB fret positions.
    """
    n_strings = len(tuning_data)

    # <staves>2</staves> — must appear before <clef> per MusicXML schema
    staves_el = ET.Element(tag("staves"))
    staves_el.text = "2"
    attrs_el.insert(0, staves_el)

    # Number the existing clef as staff 1 (if any)
    existing_clef = attrs_el.find(tag("clef"))
    if existing_clef is not None and not existing_clef.get("number"):
        existing_clef.set("number", "1")

    # <clef number="2"><sign>TAB</sign></clef>
    tab_clef = ET.SubElement(attrs_el, tag("clef"))
    tab_clef.set("number", "2")
    sign_el = ET.SubElement(tab_clef, tag("sign"))
    sign_el.text = "TAB"

    # <staff-details number="2">
    staff_details = ET.SubElement(attrs_el, tag("staff-details"))
    staff_details.set("number", "2")

    staff_type = ET.SubElement(staff_details, tag("staff-type"))
    staff_type.text = "tab"

    staff_lines_el = ET.SubElement(staff_details, tag("staff-lines"))
    staff_lines_el.text = str(n_strings)

    for line_num, (step, octave) in enumerate(tuning_data, start=1):
        st = ET.SubElement(staff_details, tag("staff-tuning"))
        st.set("line", str(line_num))
        step_el = ET.SubElement(st, tag("tuning-step"))
        step_el.text = step
        octave_el = ET.SubElement(st, tag("tuning-octave"))
        octave_el.text = str(octave)


def _annotate_measure(
    measure_el: ET.Element,
    note_assignment_map: dict[int, tuple[tuple[int, int], ...]],
    n_strings: int,
    tag: Callable[[str], str],
    midi_tuning: list[int],
) -> None:
    """Annotate staff-1 notes and append staff-2 mirror copies for one measure.

    For each note in *measure_el*:

    - Add ``<staff>1</staff>`` and fret/string technical annotation
      (non-rest notes that have an assignment only).
    - Deep-copy each slot's notes with ``<staff>2</staff>`` and the same
      technical annotation, then append them after a ``<backup>`` element.

    Chord slots: the N staff-2 copies for an N-note chord are emitted with
    ``<chord/>`` on members 2..N so the MusicXML cursor is not advanced more
    than once per slot.  The assignment entries (one pair per chord member) are
    zipped onto the XML notes in the order they appear in
    *note_assignment_map* (already MIDI-ascending from the pre-processing step).

    MusicXML cursor semantics: each ``<note>`` without ``<chord/>`` advances the
    cursor by its duration.  Staff-2 notes must be preceded by a ``<backup>`` that
    resets the cursor to the beginning of the measure; otherwise they appear at
    beat positions past the barline and are not rendered.

    Parameters
    ----------
    measure_el : ET.Element
        A MusicXML ``<measure>`` element (modified in-place).
    note_assignment_map : dict[int, tuple[tuple[int, int], ...]]
        Maps ``id(note_el)`` to the MIDI-ordered assignment tuple for that note.
        Chord siblings all map to the same tuple — each sibling's own pair is
        looked up by its position within the slot (XML note index matches
        assignment index because both are MIDI-ascending).
    n_strings : int
        Total number of strings for this instrument (used to convert string_idx
        to MusicXML 1-based string number from the top).
    tag : Callable[[str], str]
        Namespace-aware tag builder.
    midi_tuning : list[int]
        MIDI pitch of each open string (index == string_idx).
    """
    slots = _group_notes_into_slots(measure_el, tag)
    staff2_notes: list[ET.Element] = []
    total_duration = 0

    for slot in slots:
        is_rest = slot[0].find(tag("rest")) is not None
        # Get the full assignment tuple for this slot (same for all siblings).
        slot_assignment = note_assignment_map.get(id(slot[0]))

        for note_el in slot:
            # Add <staff>1</staff> to every original note.
            staff1_el = ET.SubElement(note_el, tag("staff"))
            staff1_el.text = "1"

        # Annotate staff-1 notes with technical if there's an assignment.
        if not is_rest and slot_assignment is not None:
            for idx, note_el in enumerate(slot):
                if idx < len(slot_assignment):
                    string_idx, fret = slot_assignment[idx]
                    string_num = n_strings - string_idx
                    _add_technical_annotation(note_el, string_num, fret, tag)

        # Accumulate measure duration from the first note of this slot only
        # (chord siblings share the same beat position and do not advance the
        # cursor in MusicXML).
        first_note = slot[0]
        if first_note.find(tag("chord")) is None:
            dur_el = first_note.find(tag("duration"))
            if dur_el is not None:
                with contextlib.suppress(ValueError):
                    total_duration += int(dur_el.text or "0")

        # Build staff-2 copies for this slot.
        for note_el in slot:
            staff2_note = copy.deepcopy(note_el)
            # Update (or add) <staff>2</staff>.
            staff_el = staff2_note.find(tag("staff"))
            if staff_el is not None:
                staff_el.text = "2"
            else:
                s = ET.SubElement(staff2_note, tag("staff"))
                s.text = "2"
            # Chord siblings 2..N must retain <chord/>; the first must not have
            # one so it anchors the beat position.  Deep-copy preserves whatever
            # <chord/> state the original had, which is already correct.
            staff2_notes.append(staff2_note)

    if not staff2_notes or total_duration == 0:
        return

    # <backup> resets the MusicXML cursor to the start of the measure so
    # the staff-2 notes are placed at the correct beat positions.
    backup_el = ET.Element(tag("backup"))
    backup_dur = ET.SubElement(backup_el, tag("duration"))
    backup_dur.text = str(total_duration)
    measure_el.append(backup_el)

    for note in staff2_notes:
        measure_el.append(note)


def _add_technical_annotation(
    note_el: ET.Element,
    string_num: int,
    fret: int,
    tag: Callable[[str], str],
) -> None:
    """Inject ``<notations><technical><string>/<fret>`` into *note_el*.

    Parameters
    ----------
    note_el : ET.Element
        A MusicXML ``<note>`` element (modified in-place).
    string_num : int
        1-based string number counting from the highest (thinnest) string,
        as required by MusicXML (e.g. string 1 = high E on guitar).
    fret : int
        Fret number (0 = open).
    tag : Callable[[str], str]
        Namespace-aware tag builder.
    """
    notations = note_el.find(tag("notations"))
    if notations is None:
        notations = ET.SubElement(note_el, tag("notations"))
    technical = ET.SubElement(notations, tag("technical"))
    string_el = ET.SubElement(technical, tag("string"))
    string_el.text = str(string_num)
    fret_el = ET.SubElement(technical, tag("fret"))
    fret_el.text = str(fret)
