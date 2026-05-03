"""
export/tab_xml.py — Injects a tablature Part into an existing MusicXML file.

MusicXML tab notation uses:
  - <staff-details><staff-type>tab</staff-type>...</staff-details>
  - <clef><sign>TAB</sign></clef>
  - <technical><string>N</string><fret>M</fret></technical> on each note

This approach works reliably with MuseScore because we control the XML directly,
bypassing music21's incomplete tablature support.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from audio_to_sheet.analysis.instrument_profiles import Instrument, InstrumentProfile
from audio_to_sheet.quantization.grid import QuantizedNote


# ---------------------------------------------------------------------------
# Tuning data for <staff-tuning> elements
# Strings ordered high → low (line 1 = highest string in MusicXML convention)
# Each entry is (step, octave) where step is the note letter name.
# ---------------------------------------------------------------------------

# Guitar: string 1 (high E4) → string 6 (low E2)
GUITAR_STAFF_TUNING: list[tuple[str, int]] = [
    ("E", 4), ("B", 3), ("G", 3), ("D", 3), ("A", 2), ("E", 2),
]

# Bass: string 1 (high G2) → string 4 (low E1)
BASS_STAFF_TUNING: list[tuple[str, int]] = [
    ("G", 2), ("D", 2), ("A", 1), ("E", 1),
]

_STAFF_TUNING: dict[Instrument, list[tuple[str, int]]] = {
    Instrument.GUITAR: GUITAR_STAFF_TUNING,
    Instrument.BASS: BASS_STAFF_TUNING,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_tab_part(
    mxl_path: Path,
    notes: list[QuantizedNote],
    assignments: list[tuple[int, int] | None],
    profile: InstrumentProfile,
) -> None:
    """
    Modify *mxl_path* in-place to add a TAB Part alongside the existing notation Part.

    Parameters
    ----------
    mxl_path : Path
        Path to an existing MusicXML file (will be overwritten).
    notes : list[QuantizedNote]
        The same note list used to build the score (in order).
    assignments : list[tuple[int, int] | None]
        Fret assignments from tab_builder.assign_frets(). Parallel to notes.
        Each entry is (string_idx, fret) or None for rests.
    profile : InstrumentProfile
        Used for string count and tuning info.
    """
    tree = ET.parse(mxl_path)
    root = tree.getroot()

    # Detect namespace (some music21 versions include it)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    def tag(name: str) -> str:
        return f"{ns}{name}"

    # -----------------------------------------------------------------------
    # 1. Find the first existing <part> (music21 uses non-sequential IDs)
    # -----------------------------------------------------------------------
    p1 = root.find(f".//{tag('part')}")
    if p1 is None:
        raise ValueError("MusicXML file has no <part> element.")

    # -----------------------------------------------------------------------
    # 2. Add <score-part id="P2-Tab"> to <part-list>
    # -----------------------------------------------------------------------
    part_list = root.find(tag("part-list"))
    if part_list is None:
        raise ValueError("MusicXML file has no <part-list> element.")

    tab_part_id = "P2-Tab"
    score_part = ET.SubElement(part_list, tag("score-part"))
    score_part.set("id", tab_part_id)
    part_name_el = ET.SubElement(score_part, tag("part-name"))
    part_name_el.text = "Tab"

    # -----------------------------------------------------------------------
    # 3. Collect all P1 notes in document order so we can match assignments
    # -----------------------------------------------------------------------
    # We iterate in the same order the note elements appear across all measures.
    p1_notes_ordered: list[ET.Element] = []
    for measure_el in p1.findall(tag("measure")):
        for note_el in measure_el.findall(tag("note")):
            p1_notes_ordered.append(note_el)

    # Build an assignment iterator that mirrors the note-matching logic:
    # non-rest notes consume non-None assignments in order.
    # Rests consume None assignments.
    assignment_iter = iter(assignments)

    def _is_rest_element(note_el: ET.Element) -> bool:
        return note_el.find(tag("rest")) is not None

    # Pre-build a map: note_el → (string_num, fret) or None
    note_assignment_map: dict[int, tuple[int, int] | None] = {}  # id(el) → assignment
    for note_el in p1_notes_ordered:
        if _is_rest_element(note_el):
            # Rest — consume None from assignments
            try:
                assignment = next(assignment_iter)
            except StopIteration:
                assignment = None
        else:
            # Non-rest note — consume non-None assignment
            try:
                assignment = next(assignment_iter)
            except StopIteration:
                assignment = None
        note_assignment_map[id(note_el)] = assignment

    # Tuning and string count
    tuning_data = _STAFF_TUNING.get(profile.instrument, GUITAR_STAFF_TUNING)
    n_strings = len(tuning_data)

    # -----------------------------------------------------------------------
    # 4. Build the new P2 <part> element
    # -----------------------------------------------------------------------
    p2 = ET.Element(tag("part"))
    p2.set("id", tab_part_id)

    first_measure = True
    for p1_measure in p1.findall(tag("measure")):
        measure_num = p1_measure.get("number", "1")
        p2_measure = ET.SubElement(p2, tag("measure"))
        p2_measure.set("number", measure_num)

        # In the first measure, inject tab staff attributes
        if first_measure:
            attrs_el = ET.SubElement(p2_measure, tag("attributes"))
            _build_tab_attributes(attrs_el, tuning_data, n_strings, tag)
            first_measure = False

        # Copy note elements, adding technical annotations where applicable
        for note_el in p1_measure.findall(tag("note")):
            new_note = _copy_element_deep(note_el, tag)
            assignment = note_assignment_map.get(id(note_el))

            if assignment is not None and not _is_rest_element(note_el):
                string_idx, fret = assignment
                # MusicXML string numbering: 1 = highest string
                string_num = n_strings - string_idx
                _add_technical_annotation(new_note, string_num, fret, tag)

            p2_measure.append(new_note)

    # -----------------------------------------------------------------------
    # 5. Append P2 to root and write back
    # -----------------------------------------------------------------------
    root.append(p2)
    tree.write(str(mxl_path), encoding="unicode", xml_declaration=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_tab_attributes(
    attrs_el: ET.Element,
    tuning_data: list[tuple[str, int]],
    n_strings: int,
    tag: object,  # callable str → str
) -> None:
    """Append <staff-details> and <clef> for tab notation to *attrs_el*."""
    staff_details = ET.SubElement(attrs_el, tag("staff-details"))

    staff_type = ET.SubElement(staff_details, tag("staff-type"))
    staff_type.text = "tab"

    staff_lines = ET.SubElement(staff_details, tag("staff-lines"))
    staff_lines.text = str(n_strings)

    # One <staff-tuning> per string; line 1 = highest string
    for line_num, (step, octave) in enumerate(tuning_data, start=1):
        st = ET.SubElement(staff_details, tag("staff-tuning"))
        st.set("line", str(line_num))
        step_el = ET.SubElement(st, tag("tuning-step"))
        step_el.text = step
        octave_el = ET.SubElement(st, tag("tuning-octave"))
        octave_el.text = str(octave)

    clef_el = ET.SubElement(attrs_el, tag("clef"))
    sign_el = ET.SubElement(clef_el, tag("sign"))
    sign_el.text = "TAB"


def _add_technical_annotation(
    note_el: ET.Element,
    string_num: int,
    fret: int,
    tag: object,  # callable str → str
) -> None:
    """Inject <notations><technical><string> and <fret> into *note_el*."""
    notations = ET.SubElement(note_el, tag("notations"))
    technical = ET.SubElement(notations, tag("technical"))
    string_el = ET.SubElement(technical, tag("string"))
    string_el.text = str(string_num)
    fret_el = ET.SubElement(technical, tag("fret"))
    fret_el.text = str(fret)


def _copy_element_deep(el: ET.Element, tag: object) -> ET.Element:
    """Return a deep copy of *el* (tag, attribs, text, tail, children)."""
    new_el = ET.Element(el.tag, attrib=dict(el.attrib))
    new_el.text = el.text
    new_el.tail = el.tail
    for child in el:
        new_el.append(_copy_element_deep(child, tag))
    return new_el
