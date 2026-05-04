"""
notation/tab_builder.py — Guitar/bass fret assignment via dynamic programming.

Given a sequence of QuantizedNotes and an instrument, assigns each non-rest
note a (string_idx, fret) pair that minimises total left-hand position shift
across the phrase.  The DP transition cost is the absolute difference in fret
number between consecutive assigned positions, summed over all transitions.
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
    """
    Return all (string_idx, fret) pairs on which *midi_note* is playable.

    string_idx is 0-based, where 0 is the lowest string.
    Only frets in the range [0, MAX_FRET] are returned.
    """
    candidates: list[tuple[int, int]] = []
    for string_idx, open_midi in enumerate(tuning):
        fret = midi_note - open_midi
        if 0 <= fret <= MAX_FRET:
            candidates.append((string_idx, fret))
    return candidates


def assign_frets(
    notes: list[QuantizedNote],
    instrument: Instrument,
) -> list[tuple[int, int] | None]:
    """
    Assign a (string_idx, fret) pair to each note in *notes*.

    Rests (``note.is_rest == True``) receive ``None``.
    Notes with no playable candidates (outside instrument range) receive the
    fallback ``(0, 0)``.

    The assignment minimises ``sum(|fret[i] - fret[i-1]|)`` across consecutive
    non-rest notes, breaking ties by preferring the lower fret.
    """
    tuning = _TUNINGS[instrument]
    result: list[tuple[int, int] | None] = [None] * len(notes)

    non_rest: list[tuple[int, QuantizedNote]] = [
        (i, note) for i, note in enumerate(notes) if not note.is_rest
    ]

    if not non_rest:
        return result

    cands: list[list[tuple[int, int]]] = []
    for _, note in non_rest:
        c = get_candidates(note.midi_note, tuning)
        cands.append(c)

    n_notes = len(non_rest)

    # dp[j][c] = (cost, fret) — minimum cumulative transition cost to reach
    # candidate c at note j, with the fret at j as a tie-breaker so that when
    # two paths have equal cost the one with the lower current fret is preferred.
    # prev[j][c] = candidate index at the previous note on the optimal path to (j,c).
    INF = math.inf
    dp:   list[list[tuple[float, float]]] = [[(INF, INF)] * len(cands[j]) for j in range(n_notes)]
    prev: list[list[int]]                 = [[-1]          * len(cands[j]) for j in range(n_notes)]

    # Initialise first note: transition cost = 0; tie-break by own fret.
    if cands[0]:
        for c_idx, (_s, fret) in enumerate(cands[0]):
            dp[0][c_idx] = (0, fret)

    # Fill DP table.
    # When a note has no candidates we skip it and chain the *next* valid note
    # back to the last note that did have candidates (tracked via last_valid).
    last_valid = 0 if cands[0] else -1
    for j in range(1, n_notes):
        if not cands[j]:
            # No candidates — will use fallback; do not update last_valid
            continue
        p = last_valid  # index of last note that had candidates
        if p == -1:
            # No prior note had candidates; all dp[j] stay INF
            last_valid = j
            continue
        for c_idx, (_s, fret) in enumerate(cands[j]):
            for p_idx, (_ps, p_fret) in enumerate(cands[p]):
                if dp[p][p_idx][0] == INF:
                    continue
                cost = dp[p][p_idx][0] + abs(fret - p_fret)
                candidate = (cost, fret)
                if candidate < dp[j][c_idx]:
                    dp[j][c_idx] = candidate
                    prev[j][c_idx] = p_idx
        last_valid = j

    # Backtrack: find the best final candidate at the last note with candidates.
    assignments: list[int] = [-1] * n_notes

    last_with_cands = -1
    for j in range(n_notes - 1, -1, -1):
        if cands[j]:
            last_with_cands = j
            break

    if last_with_cands >= 0:
        best_c = min(
            range(len(cands[last_with_cands])),
            key=lambda c_idx: dp[last_with_cands][c_idx],
        )
        assignments[last_with_cands] = best_c

        # Walk backwards, following prev links and skipping no-candidate notes.
        j = last_with_cands - 1
        while j >= 0:
            if not cands[j]:
                # No candidates — fallback; skip
                j -= 1
                continue
            # Find the next note after j that has an assignment
            nxt = j + 1
            while nxt <= last_with_cands and not cands[nxt]:
                nxt += 1
            if nxt > last_with_cands or assignments[nxt] == -1:
                assignments[j] = -1
            else:
                assignments[j] = prev[nxt][assignments[nxt]]
            j -= 1

    for j, (orig_idx, _note) in enumerate(non_rest):
        c_idx = assignments[j]
        if c_idx == -1 or not cands[j]:
            result[orig_idx] = (0, 0)
        else:
            result[orig_idx] = cands[j][c_idx]

    return result
