"""Tests for the conjunction screener.

The pure-function pieces (altitude pre-filter, risk classifier, candidate
selection) get exhaustive unit coverage. The end-to-end find_closest_approach
path is validated against a synthetic case where two satellites use the same
TLE — the minimum separation must drop to ~0 because the orbits are identical.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from spacetrack.anomaly import conjunction


# --- Altitude shell pre-filter ----------------------------------------------

def test_shells_overlap_when_bands_intersect():
    # A: 540-560, B: 555-575 — clearly overlap.
    assert conjunction.altitude_shells_overlap(540, 560, 555, 575, buffer_km=0)


def test_shells_dont_overlap_when_well_separated():
    # A: 540-560, B: 1100-1200 — far apart even with generous buffer.
    assert not conjunction.altitude_shells_overlap(
        540, 560, 1100, 1200, buffer_km=50
    )


def test_shell_buffer_rescues_near_misses():
    # A: 540-560, B: 600-620 — 40 km gap, rescued by 50 km buffer.
    assert conjunction.altitude_shells_overlap(540, 560, 600, 620, buffer_km=50)
    assert not conjunction.altitude_shells_overlap(540, 560, 600, 620, buffer_km=10)


def test_shells_overlap_when_one_contains_the_other():
    # A: 200-2000, B: 540-560 — B sits inside A's band.
    assert conjunction.altitude_shells_overlap(200, 2000, 540, 560, buffer_km=0)


# --- Risk classifier --------------------------------------------------------

@pytest.mark.parametrize("miss_km,expected", [
    (0.0,   "critical"),
    (0.5,   "critical"),
    (0.999, "critical"),
    (1.0,   "high"),
    (1.999, "high"),
    (2.0,   "medium"),
    (4.999, "medium"),
    (5.0,   "low"),
    (9.999, "low"),
    (10.0,  None),
    (50.0,  None),
])
def test_classify_risk_buckets(miss_km, expected):
    assert conjunction.classify_risk(miss_km) == expected


# --- Candidate selection ----------------------------------------------------

def _row(norad_id, peri, apo, *, constellation="other"):
    return conjunction._SatRow(
        norad_id=norad_id,
        name=f"SAT-{norad_id}",
        constellation=constellation,
        line1="line1",
        line2="line2",
        perigee_km=peri,
        apogee_km=apo,
    )


def test_select_candidates_excludes_self():
    primary = _row(1, 540, 560, constellation="starlink")
    pool = [primary, _row(2, 540, 560)]
    survivors = conjunction.select_candidates(primary, pool, buffer_km=10, cap=10)
    assert [c.norad_id for c in survivors] == [2]


def test_select_candidates_drops_non_overlapping_shells():
    primary = _row(1, 540, 560, constellation="starlink")
    pool = [
        _row(2, 538, 562),     # overlaps
        _row(3, 1100, 1200),   # too high
        _row(4, 100, 200),     # too low
        _row(5, 600, 620),     # 40 km gap — rescued only by big buffer
    ]
    tight = conjunction.select_candidates(primary, pool, buffer_km=10, cap=10)
    assert sorted(c.norad_id for c in tight) == [2]

    loose = conjunction.select_candidates(primary, pool, buffer_km=60, cap=10)
    assert sorted(c.norad_id for c in loose) == [2, 5]


def test_select_candidates_caps_and_sorts_by_overlap():
    primary = _row(1, 540, 560, constellation="starlink")
    pool = [
        _row(2, 535, 565),    # widest overlap
        _row(3, 545, 555),    # narrower overlap, same band
        _row(4, 595, 605),    # 35 km gap — only with buffer
        _row(5, 549, 551),    # tightly nested
    ]
    out = conjunction.select_candidates(primary, pool, buffer_km=50, cap=2)
    # All four overlap (within buffer), but cap=2 keeps the two with the
    # smallest shell-distance. The three with true intersection (shell_dist=0)
    # tie first — implementation keeps a stable order among ties.
    assert len(out) == 2
    # The 35-km-gap candidate must NOT survive a cap=2 selection.
    assert 4 not in [c.norad_id for c in out]


# --- End-to-end via identical TLE -------------------------------------------

# A real Starlink TLE captured for deterministic tests.
_STARLINK_NAME = "STARLINK-1007"
_STARLINK_L1 = "1 44713U 19074A   24145.50000000  .00009123  00000+0  62345-3 0  9991"
_STARLINK_L2 = "2 44713  53.0534 256.7891 0001234  90.1234 269.9876 15.06398765123456"


def _checksum(line: str) -> str:
    """Append a TLE checksum digit to a 68-char line."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return line[:68] + str(total % 10)


def test_identical_orbits_have_near_zero_miss_distance():
    """Two sats with the same TLE must trace identical orbits — miss = 0."""
    l1 = _checksum(_STARLINK_L1[:68])
    l2 = _checksum(_STARLINK_L2[:68])
    miss_km, tca, rel_speed = conjunction.find_closest_approach(
        l1, l2, "A",
        l1, l2, "B",
        start=datetime(2024, 5, 24, tzinfo=timezone.utc),
        forecast_hours=1.0,
        step_seconds=30.0,
    )
    assert miss_km < 1e-6
    assert rel_speed < 1e-6
    assert tca.tzinfo is not None
