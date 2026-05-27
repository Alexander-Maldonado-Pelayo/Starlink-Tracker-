"""Conjunction screening: close-approach forecast between Starlink and the
external catalog.

A *conjunction* is a pair of satellites whose 3-D ECI separation drops below
a chosen threshold within a forecast window. The screening problem is N×M:
the Starlink set (≈ 7 000 sats) crossed with everything else in CelesTrak's
``active`` catalog (≈ 10 000 sats) is too large to brute-force at fine
time-resolution, so we apply two cheap geometric pre-filters before any
SGP4 propagation:

1. **Altitude-shell overlap** — perigee/apogee bands must overlap within a
   buffer; otherwise the two orbits never touch the same altitude and
   collision is geometrically impossible.
2. **Per-primary candidate cap** — keep the N nearest-altitude candidates
   per Starlink. Bounds total compute regardless of catalog size; documented
   as a v1 limitation rather than a complete screen.

Survivors are batch-propagated with Skyfield over the forecast window; the
minimum 3-D distance, time-of-closest-approach (TCA), and relative speed at
TCA are recorded per pair. Risk is tiered by miss distance:

* ``critical`` — < 1 km   (NASA red-box style)
* ``high``     — < 2 km
* ``medium``   — < 5 km
* ``low``      — < 10 km  (default scan floor)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
from skyfield.api import EarthSatellite, load

from spacetrack.anomaly.decay import perigee_apogee_km

log = logging.getLogger(__name__)

Risk = Literal["low", "medium", "high", "critical"]
_RISK_ORDER: dict[Risk, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

DEFAULT_FORECAST_HOURS: float = 24.0
DEFAULT_STEP_SECONDS: float = 60.0
DEFAULT_MAX_DISTANCE_KM: float = 10.0
DEFAULT_ALTITUDE_BUFFER_KM: float = 50.0
DEFAULT_MAX_CANDIDATES_PER_PRIMARY: int = 25

# Julian date of the Unix epoch.
_JD_UNIX_EPOCH = 2440587.5


@dataclass(frozen=True)
class ConjunctionEvent:
    primary_norad_id: int
    primary_name: str
    secondary_norad_id: int
    secondary_name: str
    secondary_constellation: str
    tca_jd: float                  # Julian date of TCA
    tca_unix: float                # convenience: unix ts at TCA
    miss_distance_km: float
    relative_velocity_km_s: float  # |v_primary - v_secondary| at TCA
    risk: Risk


@dataclass(frozen=True)
class _SatRow:
    """Internal record carrying everything needed to screen and propagate."""
    norad_id: int
    name: str
    constellation: str
    line1: str
    line2: str
    perigee_km: float
    apogee_km: float


def classify_risk(miss_distance_km: float) -> Risk | None:
    """Return the strictest tier the miss distance qualifies for, or None
    when the pair is too far apart to be worth reporting (>= 10 km)."""
    if miss_distance_km < 1.0:
        return "critical"
    if miss_distance_km < 2.0:
        return "high"
    if miss_distance_km < 5.0:
        return "medium"
    if miss_distance_km < 10.0:
        return "low"
    return None


def altitude_shells_overlap(
    a_peri_km: float, a_apo_km: float,
    b_peri_km: float, b_apo_km: float,
    *, buffer_km: float = DEFAULT_ALTITUDE_BUFFER_KM,
) -> bool:
    """True iff the two altitude shells overlap within ``buffer_km``."""
    low = max(a_peri_km, b_peri_km)
    high = min(a_apo_km, b_apo_km)
    return low <= high + buffer_km


def _shell_distance_km(primary: _SatRow, secondary: _SatRow) -> float:
    """Vertical separation between the closest faces of two altitude shells.

    Zero when the shells overlap; positive otherwise. Used as a tie-breaker
    to pick the most-overlapping candidates per primary.
    """
    low = max(primary.perigee_km, secondary.perigee_km)
    high = min(primary.apogee_km, secondary.apogee_km)
    if low <= high:
        return 0.0
    return low - high


def select_candidates(
    primary: _SatRow,
    pool: list[_SatRow],
    *,
    buffer_km: float = DEFAULT_ALTITUDE_BUFFER_KM,
    cap: int = DEFAULT_MAX_CANDIDATES_PER_PRIMARY,
) -> list[_SatRow]:
    """Pre-screen the candidate pool against one primary.

    Returns a list of at most ``cap`` candidates whose altitude shell
    overlaps the primary's (within ``buffer_km``), sorted with the
    most-overlapping shells first.
    """
    survivors: list[tuple[float, _SatRow]] = []
    for cand in pool:
        if cand.norad_id == primary.norad_id:
            continue
        if not altitude_shells_overlap(
            primary.perigee_km, primary.apogee_km,
            cand.perigee_km, cand.apogee_km,
            buffer_km=buffer_km,
        ):
            continue
        survivors.append((_shell_distance_km(primary, cand), cand))
    survivors.sort(key=lambda x: x[0])
    return [c for _, c in survivors[:cap]]


def _jd_from_datetime(dt: datetime) -> float:
    return _JD_UNIX_EPOCH + dt.astimezone(timezone.utc).timestamp() / 86400.0


def _propagate_eci(
    sat: EarthSatellite, times,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one sat to a batch of times.

    Returns ``(position_km, velocity_km_s)`` each shaped ``(3, N)``.
    """
    geo = sat.at(times)
    return geo.position.km, geo.velocity.km_per_s


def find_closest_approach(
    primary_l1: str, primary_l2: str, primary_name: str,
    secondary_l1: str, secondary_l2: str, secondary_name: str,
    *,
    start: datetime,
    forecast_hours: float = DEFAULT_FORECAST_HOURS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    timescale=None,
) -> tuple[float, datetime, float]:
    """Compute min distance, TCA, and relative speed at TCA for one pair.

    Pure-ish — only depends on Skyfield's Timescale (cached at module scope
    when ``timescale`` is None). Returns ``(miss_km, tca_utc, rel_speed_km_s)``.
    """
    if timescale is None:
        timescale = _get_timescale()
    if start.tzinfo is None:
        raise ValueError("`start` must be timezone-aware (UTC)")

    samples = max(2, int(forecast_hours * 3600.0 / step_seconds) + 1)
    start_utc = start.astimezone(timezone.utc)
    moments = [start_utc + timedelta(seconds=i * step_seconds) for i in range(samples)]
    t_array = timescale.from_datetimes(moments)

    sat_a = EarthSatellite(primary_l1, primary_l2, primary_name, timescale)
    sat_b = EarthSatellite(secondary_l1, secondary_l2, secondary_name, timescale)

    pos_a, vel_a = _propagate_eci(sat_a, t_array)
    pos_b, vel_b = _propagate_eci(sat_b, t_array)

    diff = pos_a - pos_b                                  # (3, N)
    distances = np.linalg.norm(diff, axis=0)              # (N,)

    finite = np.isfinite(distances)
    if not finite.any():
        return math.inf, start_utc, 0.0
    distances[~finite] = math.inf
    idx = int(distances.argmin())

    rel_v = vel_a[:, idx] - vel_b[:, idx]
    rel_speed = float(np.linalg.norm(rel_v))
    miss_km = float(distances[idx])
    tca = moments[idx]
    return miss_km, tca, rel_speed


_TIMESCALE = None


def _get_timescale():
    global _TIMESCALE
    if _TIMESCALE is None:
        _TIMESCALE = load.timescale()
    return _TIMESCALE


def _load_pool(
    conn: sqlite3.Connection, constellation: str | None,
) -> list[_SatRow]:
    """Read the latest snapshot per satellite for a constellation."""
    if constellation is None:
        rows = conn.execute(
            """
            SELECT s.norad_id, s.name, s.constellation,
                   t.line1, t.line2, t.eccentricity, t.mean_motion
            FROM satellites s
            JOIN tle_snapshots t ON t.norad_id = s.norad_id
            WHERE t.epoch = (
                SELECT MAX(epoch) FROM tle_snapshots WHERE norad_id = s.norad_id
            )
            """
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT s.norad_id, s.name, s.constellation,
                   t.line1, t.line2, t.eccentricity, t.mean_motion
            FROM satellites s
            JOIN tle_snapshots t ON t.norad_id = s.norad_id
            WHERE s.constellation = ?
              AND t.epoch = (
                  SELECT MAX(epoch) FROM tle_snapshots WHERE norad_id = s.norad_id
              )
            """,
            (constellation,),
        ).fetchall()

    out: list[_SatRow] = []
    for r in rows:
        try:
            peri, apo = perigee_apogee_km(r["mean_motion"], r["eccentricity"])
        except (ValueError, ZeroDivisionError):
            continue
        out.append(_SatRow(
            norad_id=r["norad_id"],
            name=r["name"] or f"NORAD-{r['norad_id']}",
            constellation=r["constellation"] or "unknown",
            line1=r["line1"],
            line2=r["line2"],
            perigee_km=peri,
            apogee_km=apo,
        ))
    return out


def scan_satellite(
    conn: sqlite3.Connection,
    norad_id: int,
    *,
    start: datetime | None = None,
    forecast_hours: float = DEFAULT_FORECAST_HOURS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    buffer_km: float = DEFAULT_ALTITUDE_BUFFER_KM,
    cap: int = DEFAULT_MAX_CANDIDATES_PER_PRIMARY,
    secondary_constellation: str | None = "other",
) -> list[ConjunctionEvent]:
    """Find all close approaches between ``norad_id`` and the secondary pool.

    Loads the latest snapshot for the primary, applies the altitude-shell
    pre-filter to the pool, batch-propagates each survivor, and returns
    every pair whose miss distance is below ``max_distance_km``.
    """
    if start is None:
        start = datetime.now(timezone.utc)

    # Primary lookup. Pull from the full catalog (no constellation filter)
    # so per-sat mode works even for non-Starlink IDs.
    pool_all = _load_pool(conn, constellation=None)
    primary = next((r for r in pool_all if r.norad_id == norad_id), None)
    if primary is None:
        return []

    if secondary_constellation is None:
        pool = [r for r in pool_all if r.norad_id != norad_id]
    else:
        pool = [
            r for r in pool_all
            if r.constellation == secondary_constellation
            and r.norad_id != norad_id
        ]

    candidates = select_candidates(primary, pool, buffer_km=buffer_km, cap=cap)
    log.info(
        "Conjunction scan for NORAD %d: %d candidates after pre-filter "
        "(pool=%d, cap=%d)",
        norad_id, len(candidates), len(pool), cap,
    )

    ts = _get_timescale()
    events: list[ConjunctionEvent] = []
    for cand in candidates:
        miss_km, tca, rel_speed = find_closest_approach(
            primary.line1, primary.line2, primary.name,
            cand.line1, cand.line2, cand.name,
            start=start,
            forecast_hours=forecast_hours,
            step_seconds=step_seconds,
            timescale=ts,
        )
        if not math.isfinite(miss_km) or miss_km >= max_distance_km:
            continue
        risk = classify_risk(miss_km)
        if risk is None:
            continue
        events.append(ConjunctionEvent(
            primary_norad_id=primary.norad_id,
            primary_name=primary.name,
            secondary_norad_id=cand.norad_id,
            secondary_name=cand.name,
            secondary_constellation=cand.constellation,
            tca_jd=_jd_from_datetime(tca),
            tca_unix=tca.timestamp(),
            miss_distance_km=miss_km,
            relative_velocity_km_s=rel_speed,
            risk=risk,
        ))

    events.sort(key=lambda e: (-_RISK_ORDER[e.risk], e.miss_distance_km))
    return events


def scan(
    conn: sqlite3.Connection,
    *,
    start: datetime | None = None,
    forecast_hours: float = DEFAULT_FORECAST_HOURS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    buffer_km: float = DEFAULT_ALTITUDE_BUFFER_KM,
    cap: int = DEFAULT_MAX_CANDIDATES_PER_PRIMARY,
    min_risk: Risk = "low",
    primary_constellation: str = "starlink",
    secondary_constellation: str | None = "other",
    primary_limit: int | None = None,
) -> list[ConjunctionEvent]:
    """Full-catalog conjunction screen.

    Pre-filters every primary × secondary pair by altitude-shell overlap,
    keeps the ``cap`` nearest-shell candidates per primary, and propagates
    those over the forecast window. Set ``primary_limit`` to bound compute
    on very large primary sets (handy for dashboard previews).

    Results are sorted by risk tier, then by miss distance ascending.
    """
    if start is None:
        start = datetime.now(timezone.utc)

    pool_all = _load_pool(conn, constellation=None)
    primaries = [r for r in pool_all if r.constellation == primary_constellation]
    if secondary_constellation is None:
        secondary_pool = pool_all
    else:
        secondary_pool = [r for r in pool_all if r.constellation == secondary_constellation]

    if primary_limit is not None:
        primaries = primaries[:primary_limit]

    log.info(
        "Conjunction scan: %d primaries × %d secondaries, "
        "forecast=%.1fh, step=%.0fs, max_distance=%.1fkm, cap=%d",
        len(primaries), len(secondary_pool),
        forecast_hours, step_seconds, max_distance_km, cap,
    )

    ts = _get_timescale()
    threshold = _RISK_ORDER[min_risk]
    flagged: list[ConjunctionEvent] = []
    for primary in primaries:
        candidates = select_candidates(
            primary, secondary_pool, buffer_km=buffer_km, cap=cap,
        )
        for cand in candidates:
            miss_km, tca, rel_speed = find_closest_approach(
                primary.line1, primary.line2, primary.name,
                cand.line1, cand.line2, cand.name,
                start=start,
                forecast_hours=forecast_hours,
                step_seconds=step_seconds,
                timescale=ts,
            )
            if not math.isfinite(miss_km) or miss_km >= max_distance_km:
                continue
            risk = classify_risk(miss_km)
            if risk is None or _RISK_ORDER[risk] < threshold:
                continue
            flagged.append(ConjunctionEvent(
                primary_norad_id=primary.norad_id,
                primary_name=primary.name,
                secondary_norad_id=cand.norad_id,
                secondary_name=cand.name,
                secondary_constellation=cand.constellation,
                tca_jd=_jd_from_datetime(tca),
                tca_unix=tca.timestamp(),
                miss_distance_km=miss_km,
                relative_velocity_km_s=rel_speed,
                risk=risk,
            ))

    flagged.sort(key=lambda e: (-_RISK_ORDER[e.risk], e.miss_distance_km))
    return flagged
