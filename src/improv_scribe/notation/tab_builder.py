"""
notation/tab_builder.py — Guitar/bass fret assignment via dynamic programming.

Given a sequence of QuantizedNotes and an instrument, assigns each non-rest
note a tuple of (string_idx, fret) pairs that minimises total left-hand
position shift across the phrase.

Phase 2 (chord-aware): each non-rest note receives a
``tuple[tuple[int, int], ...]`` — one ``(string_idx, fret)`` pair per chord
member, with the no-string-conflict constraint enforced.  Mono notes get
length-1 outer tuples, so the Phase 0 / Phase 1 DP behaviour is preserved
bit-equivalently on monophonic input.
"""

from __future__ import annotations

import math

from improv_scribe.analysis.instrument_profiles import Instrument
from improv_scribe.quantization.grid import QuantizedNote

# ---------------------------------------------------------------------------
# String tunings — MIDI note of each open string, ordered low-to-high
# ---------------------------------------------------------------------------

GUITAR_TUNING: list[int] = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
BASS_TUNING:   list[int] = [28, 33, 38, 43]           # E1 A1 D2 G2

MAX_FRET: int = 22

_TUNINGS: dict[Instrument, list[int]] = {
    Instrument.GUITAR: GUITAR_TUNING,
    Instrument.BASS:   BASS_TUNING,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_candidates(midi_note: int, tuning: list[int]) -> list[tuple[int, int]]:
    """Return all (string_idx, fret) pairs on which *midi_note* is playable.

    string_idx is 0-based, where 0 is the lowest string.
    Only frets in the range [0, MAX_FRET] are returned.

    Parameters
    ----------
    midi_note : int
        MIDI note number to find candidates for.
    tuning : list[int]
        MIDI numbers of open strings, low to high.

    Returns
    -------
    list[tuple[int, int]]
        (string_idx, fret) pairs, ordered by string ascending.
    """
    candidates: list[tuple[int, int]] = []
    for string_idx, open_midi in enumerate(tuning):
        fret = midi_note - open_midi
        if 0 <= fret <= MAX_FRET:
            candidates.append((string_idx, fret))
    return candidates


def get_chord_shapes(
    midi_notes: tuple[int, ...],
    tuning: list[int],
) -> list[tuple[tuple[int, int], ...]]:
    """Return all (string, fret) assignments that put each chord member on a
    distinct string.

    For mono (length-1 midi_notes), returns one shape per candidate string.
    For chord midi_notes, enumerates the Cartesian product of per-member
    candidates and keeps only combinations where strings are pairwise distinct.

    Parameters
    ----------
    midi_notes : tuple[int, ...]
        MIDI note numbers, typically sorted ascending (canonical form).
    tuning : list[int]
        MIDI numbers of open strings, low to high.

    Returns
    -------
    list[tuple[tuple[int, int], ...]]
        Each inner tuple is one valid shape, sorted by string ascending.
        Empty list if no conflict-free shape exists or any member is
        out of range.
    """
    import itertools  # noqa: PLC0415

    per_note_candidates = [get_candidates(m, tuning) for m in midi_notes]
    if any(not c for c in per_note_candidates):
        # At least one member has no candidates; no shape possible.
        return []

    shapes: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.product(*per_note_candidates):
        strings = [s for s, _f in combo]
        if len(set(strings)) == len(strings):
            # Sort by string ascending so the shape is canonical
            shapes.append(tuple(sorted(combo, key=lambda sf: sf[0])))
    return shapes


def assign_frets(
    notes: list[QuantizedNote],
    instrument: Instrument,
) -> list[tuple[tuple[int, int], ...] | None]:
    """Assign chord-aware (string, fret) tuples to each note using DP.

    Each non-rest note is assigned a tuple of (string, fret) pairs — one
    pair per chord member, with the no-string-conflict constraint
    (members on distinct strings).

    Cost model
    ----------
    Within-shape cost: hand stretch = max(fret) - min(fret) over fretted
                       members; open strings (fret 0) excluded; 0 for
                       all-open shapes.
    Transition cost:   |centroid_curr - centroid_prev| where centroid is
                       the mean fret of fretted members in the shape;
                       defaults to 0 if all open.
    Tie-break:         lex (cumulative_cost, max_fret_in_shape,
                       min_fret_in_shape). On singletons this reduces
                       bit-equivalently to the Phase 0 single-note DP.

    Rests receive None.

    Fallbacks
    ---------
    - Shape enumeration empty (member out of range): drop offending members
      from the highest fret down until a non-empty enumeration succeeds.
      If even the playable subset has no conflict-free shape, return
      ``((0, 0),)``.
    - All members unplayable: ``((0, 0),)`` so the score still renders.

    Parameters
    ----------
    notes : list[QuantizedNote]
        Mixed mono and chord notes. Rests receive None.
    instrument : Instrument

    Returns
    -------
    list[tuple[tuple[int, int], ...] | None]
        Parallel to *notes*. Each non-rest entry is a tuple of
        (string, fret) pairs sorted by string ascending. Rests are None.
    """
    tuning = _TUNINGS[instrument]
    result: list[tuple[tuple[int, int], ...] | None] = [None] * len(notes)

    # Identify non-rest indices and their shape lists
    non_rest_indices: list[int] = []
    shape_lists: list[list[tuple[tuple[int, int], ...]]] = []
    for i, note in enumerate(notes):
        if note.is_rest:
            continue
        non_rest_indices.append(i)
        shapes = get_chord_shapes(note.midi_notes, tuning)
        if not shapes:
            shapes = _fallback_shapes(note.midi_notes, tuning)
        shape_lists.append(shapes)

    if not non_rest_indices:
        return result

    def _centroid(shape: tuple[tuple[int, int], ...]) -> float:
        fretted = [f for _s, f in shape if f > 0]
        return sum(fretted) / len(fretted) if fretted else 0.0

    def _stretch(shape: tuple[tuple[int, int], ...]) -> int:
        fretted = [f for _s, f in shape if f > 0]
        if not fretted:
            return 0
        return max(fretted) - min(fretted)

    INF = math.inf
    n_pos = len(non_rest_indices)
    # dp[j][k] = (cumulative_cost, max_fret_in_shape, min_fret_in_shape)
    dp: list[list[tuple[float, int, int]]] = [
        [(INF, INF, INF)] * len(shape_lists[j]) for j in range(n_pos)
    ]
    prev: list[list[int]] = [[-1] * len(shape_lists[j]) for j in range(n_pos)]

    # Initialise first position: no transition cost; tie-break by (max_fret, min_fret)
    for k, shape in enumerate(shape_lists[0]):
        all_frets = [f for _s, f in shape]
        dp[0][k] = (0.0, max(all_frets), min(all_frets))

    for j in range(1, n_pos):
        for k, shape in enumerate(shape_lists[j]):
            stretch = _stretch(shape)
            curr_cent = _centroid(shape)
            all_frets = [f for _s, f in shape]
            best: tuple[float, int, int] = (INF, INF, INF)
            best_p = -1
            for p, prev_shape in enumerate(shape_lists[j - 1]):
                if dp[j - 1][p][0] == INF:
                    continue
                trans = abs(curr_cent - _centroid(prev_shape))
                cumulative = dp[j - 1][p][0] + stretch + trans
                cand = (cumulative, max(all_frets), min(all_frets))
                if cand < best:
                    best = cand
                    best_p = p
            dp[j][k] = best
            prev[j][k] = best_p

    # Backtrack
    final_k = min(range(len(shape_lists[-1])), key=lambda k: dp[-1][k])
    assignments: list[int] = [-1] * n_pos
    assignments[-1] = final_k
    for j in range(n_pos - 1, 0, -1):
        assignments[j - 1] = prev[j][assignments[j]]

    for pos, j in enumerate(non_rest_indices):
        k = assignments[pos]
        if k == -1:
            result[j] = ((0, 0),)
        else:
            result[j] = shape_lists[pos][k]

    return result


def _fallback_shapes(
    midi_notes: tuple[int, ...],
    tuning: list[int],
) -> list[tuple[tuple[int, int], ...]]:
    """When a chord has no conflict-free shape (e.g. duplicate-pitch chord),
    drop members until a valid shape exists.

    Returns at least one shape (the final fallback is ``((0, 0),)``).

    Parameters
    ----------
    midi_notes : tuple[int, ...]
        MIDI note numbers of the chord.
    tuning : list[int]
        MIDI numbers of open strings, low to high.

    Returns
    -------
    list[tuple[tuple[int, int], ...]]
        Non-empty list of shapes after dropping offending members.
    """
    for drop_count in range(1, len(midi_notes)):
        for combo in _combinations_of_size(midi_notes, len(midi_notes) - drop_count):
            shapes = get_chord_shapes(combo, tuning)
            if shapes:
                return shapes
    return [((0, 0),)]


def _combinations_of_size(
    midi_notes: tuple[int, ...],
    size: int,
) -> list[tuple[int, ...]]:
    """All subsets of midi_notes with the given size.

    Parameters
    ----------
    midi_notes : tuple[int, ...]
        Source MIDI note numbers.
    size : int
        Number of elements to select.

    Returns
    -------
    list[tuple[int, ...]]
        All combinations of the given size.
    """
    import itertools  # noqa: PLC0415
    return [tuple(c) for c in itertools.combinations(midi_notes, size)]
