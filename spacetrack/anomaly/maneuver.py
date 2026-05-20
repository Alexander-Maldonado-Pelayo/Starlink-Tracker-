"""Maneuver detection from TLE mean-motion jumps.

A satellite's semi-major axis (and therefore altitude) changes for only two
reasons in the TLE record:

* **Drag** — a slow, monotonic, negative drift that's roughly linear over a
  few days. For Starlink at 540-570 km, the magnitude is on the order of
  -10 to -100 m/day depending on solar activity.
* **Maneuver** — a discrete burn that shifts the semi-major axis by several
  hundred metres to tens of kilometres between consecutive TLE epochs.

Distinguishing the two is therefore a matter of looking at the *rate* and
*magnitude* of the change across consecutive snapshots:

* A short gap with a multi-km Δa cannot be drag (drag is too slow).
* A long gap with a sub-km Δa might be drag (and is ignored).

Output is a :class:`ManeuverEvent` per detected jump, classified by
magnitude (small / medium / large) and direction (boost / drop). The same
satellite can have multiple events across its history.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Literal

from spacetrack.anomaly.decay import EARTH_RADIUS_KM, semi_major_axis_km

# Defaults tuned for Starlink at ~550 km. ``min_delta_km`` excludes ephemeris
# noise (TLE mean-motion is reported to 8 decimals → submetre precision in a,
# but real epoch-to-epoch drift is at the metre scale). ``min_rate_km_per_day``
# excludes drag, which is generously bounded by ~0.1 km/day at this altitude.
DEFAULT_MIN_DELTA_KM: float = 0.5
DEFAULT_MIN_RATE_KM_PER_DAY: float = 0.1
DEFAULT_MAX_GAP_DAYS: float = 10.0

Magnitude = Literal["small", "medium", "large"]
_MAGNITUDE_ORDER: dict[Magnitude, int] = {"small": 0, "medium": 1, "large": 2}

Direction = Literal["boost", "drop"]


@dataclass(frozen=True)
class ManeuverEvent:
    norad_id: int
    name: str
    epoch_before: float          # JD of earlier TLE
    epoch_after: float           # JD of later TLE
    gap_days: float
    a_before_km: float           # semi-major axis (km)
    a_after_km: float
    altitude_before_km: float    # midpoint altitude (km) — a - R_earth
    altitude_after_km: float
    delta_a_km: float            # signed: positive = boost, negative = drop
    delta_a_km_per_day: float    # signed rate
    direction: Direction
    magnitude: Magnitude


def classify_magnitude(abs_delta_km: float) -> Magnitude:
    if abs_delta_km < 5.0:
        return "small"
    if abs_delta_km < 15.0:
        return "medium"
    return "large"


def detect_pair(
    *,
    norad_id: int,
    name: str,
    epoch_before: float,
    epoch_after: float,
    mean_motion_before: float,
    mean_motion_after: float,
    min_delta_km: float = DEFAULT_MIN_DELTA_KM,
    min_rate_km_per_day: float = DEFAULT_MIN_RATE_KM_PER_DAY,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
) -> ManeuverEvent | None:
    """Compare two consecutive snapshots; return a ManeuverEvent if it fires.

    Pure function — no DB access. Returns ``None`` when the change is
    consistent with drag or the epoch gap is too large to attribute to a
    single burn (``max_gap_days``).
    """
    gap_days = epoch_after - epoch_before
    if gap_days <= 0:
        return None
    if gap_days > max_gap_days:
        # Long gaps blur a maneuver with accumulated drag; skip rather than
        # falsely attribute the combined effect.
        return None

    a_before = semi_major_axis_km(mean_motion_before)
    a_after = semi_major_axis_km(mean_motion_after)
    delta = a_after - a_before
    abs_delta = abs(delta)
    rate = delta / gap_days

    if abs_delta < min_delta_km:
        return None
    if abs(rate) < min_rate_km_per_day:
        return None

    return ManeuverEvent(
        norad_id=norad_id,
        name=name,
        epoch_before=epoch_before,
        epoch_after=epoch_after,
        gap_days=gap_days,
        a_before_km=a_before,
        a_after_km=a_after,
        altitude_before_km=a_before - EARTH_RADIUS_KM,
        altitude_after_km=a_after - EARTH_RADIUS_KM,
        delta_a_km=delta,
        delta_a_km_per_day=rate,
        direction="boost" if delta > 0 else "drop",
        magnitude=classify_magnitude(abs_delta),
    )


def scan_satellite(
    conn: sqlite3.Connection,
    norad_id: int,
    *,
    min_delta_km: float = DEFAULT_MIN_DELTA_KM,
    min_rate_km_per_day: float = DEFAULT_MIN_RATE_KM_PER_DAY,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
) -> list[ManeuverEvent]:
    """Walk one satellite's snapshot history; return all detected events."""
    rows = conn.execute(
        """
        SELECT s.name, t.epoch, t.mean_motion
        FROM tle_snapshots t
        JOIN satellites s ON s.norad_id = t.norad_id
        WHERE t.norad_id = ?
        ORDER BY t.epoch ASC
        """,
        (norad_id,),
    ).fetchall()
    if len(rows) < 2:
        return []

    name = rows[0]["name"]
    events: list[ManeuverEvent] = []
    for prev, curr in zip(rows, rows[1:]):
        event = detect_pair(
            norad_id=norad_id,
            name=name,
            epoch_before=prev["epoch"],
            epoch_after=curr["epoch"],
            mean_motion_before=prev["mean_motion"],
            mean_motion_after=curr["mean_motion"],
            min_delta_km=min_delta_km,
            min_rate_km_per_day=min_rate_km_per_day,
            max_gap_days=max_gap_days,
        )
        if event is not None:
            events.append(event)
    return events


def scan(
    conn: sqlite3.Connection,
    *,
    min_magnitude: Magnitude = "small",
    min_delta_km: float = DEFAULT_MIN_DELTA_KM,
    min_rate_km_per_day: float = DEFAULT_MIN_RATE_KM_PER_DAY,
    max_gap_days: float = DEFAULT_MAX_GAP_DAYS,
    constellation: str | None = "starlink",
) -> list[ManeuverEvent]:
    """Detect maneuvers across every tracked satellite.

    Returns a flat list of events at or above ``min_magnitude``, sorted with
    the most recent and largest first.
    """
    if constellation:
        norad_rows = conn.execute(
            "SELECT norad_id FROM satellites WHERE constellation = ?",
            (constellation,),
        ).fetchall()
    else:
        norad_rows = conn.execute("SELECT norad_id FROM satellites").fetchall()

    threshold = _MAGNITUDE_ORDER[min_magnitude]
    flagged: list[ManeuverEvent] = []
    for row in norad_rows:
        events = scan_satellite(
            conn, row["norad_id"],
            min_delta_km=min_delta_km,
            min_rate_km_per_day=min_rate_km_per_day,
            max_gap_days=max_gap_days,
        )
        for ev in events:
            if _MAGNITUDE_ORDER[ev.magnitude] >= threshold:
                flagged.append(ev)

    flagged.sort(
        key=lambda e: (-e.epoch_after, -_MAGNITUDE_ORDER[e.magnitude], -abs(e.delta_a_km))
    )
    return flagged
