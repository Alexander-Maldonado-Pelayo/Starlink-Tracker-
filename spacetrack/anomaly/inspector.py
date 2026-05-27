"""Per-satellite orbital-element residual scan.

For each tracked satellite, fits a linear trend to each of the five shape
elements (mean motion, eccentricity, inclination, RAAN, argument of perigee)
across the last N days of snapshots, then checks the residual of the *latest*
snapshot against the standard deviation of all residuals in the window. The
satellite is flagged when any element's z-score exceeds the notable threshold.

This catches anomalies the maneuver detector misses:

* Maneuvers below the |Δa| ≥ 0.5 km threshold (e.g. fine station-keeping
  burns or small inclination tweaks).
* Plane changes — the maneuver detector only watches semi-major axis, but
  inclination/RAAN deviations are unmistakable inspection signatures.
* Drag anomalies (solar storm response, attitude changes) that don't move
  the orbit far enough to trip maneuver but still depart the trend.

Trend baseline: ordinary least-squares linear fit on (epoch, element). For
angular elements (RAAN, arg perigee, inclination) the series is unwrapped
modulo 360° before fitting so the trend isn't poisoned by wrap-arounds.

Severity tiers by max |z|:

* ``quiet``       — < 3  (not flagged)
* ``notable``     — 3-5
* ``significant`` — 5-8
* ``extreme``     — ≥ 8
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Literal

import numpy as np

# Element names match column names in tle_snapshots so they double as SQL fields.
# arg_perigee is omitted: it's ill-defined for near-circular orbits and adds
# only noise to the scan (p99 of latest-vs-trend deltas across Starlink is
# >100°, i.e. essentially random).
ELEMENTS: tuple[str, ...] = (
    "mean_motion",
    "eccentricity",
    "inclination",
    "raan",
)
_ANGULAR_ELEMENTS: frozenset[str] = frozenset({"inclination", "raan", "arg_perigee"})

# Per-element noise floor — sets the *minimum* standard deviation used in
# z-score normalization. Without these, a quiet 7d window's residual std
# collapses below real day-to-day TLE variability and every ordinary drag-
# induced drift in the latest snapshot scores at thousands of sigma.
#
# Values are calibrated against the p99 of |latest residual| measured across
# a 2 000-sat sample of the live constellation, divided by 3 — so a "z = 3"
# crossing corresponds roughly to a 99th-percentile-of-routine event. Sized
# for Starlink at ~550 km; constellations at radically different altitudes
# may want their own floors.
NOISE_FLOOR: dict[str, float] = {
    "mean_motion":  1.5e-2,   # rev/day  (≈ 15 km of altitude wobble)
    "eccentricity": 1.5e-4,
    "inclination":  1.0e-2,   # degrees
    "raan":         5.0e-2,   # degrees  (J2 precession is ~1°/day at Starlink alt)
}

Severity = Literal["quiet", "notable", "significant", "extreme"]
_SEVERITY_ORDER: dict[Severity, int] = {
    "quiet": 0, "notable": 1, "significant": 2, "extreme": 3,
}

DEFAULT_WINDOW_DAYS: float = 7.0
DEFAULT_MIN_SNAPSHOTS: int = 5
DEFAULT_NOTABLE_THRESHOLD: float = 3.0


@dataclass(frozen=True)
class ResidualFinding:
    norad_id: int
    name: str
    epoch: float                # JD of the latest snapshot
    snapshots_used: int         # how many snapshots in the fit window
    window_days: float          # span of the fit window (actual, may be < requested)
    element: str                # which element flagged worst
    z_score: float              # signed z-score of latest residual
    observed: float             # latest observed value
    expected: float             # baseline trend's prediction at the latest epoch
    sigma: float                # standard deviation of residuals in the window
    severity: Severity


def classify_severity(abs_z: float) -> Severity:
    if abs_z < 3.0:
        return "quiet"
    if abs_z < 5.0:
        return "notable"
    if abs_z < 8.0:
        return "significant"
    return "extreme"


def _unwrap_degrees(values: np.ndarray) -> np.ndarray:
    """Unwrap a 0-360° series so a linear trend isn't poisoned by wrap-arounds."""
    return np.degrees(np.unwrap(np.radians(values)))


def _z_for_element(
    epochs: np.ndarray, values: np.ndarray, *, is_angular: bool,
    noise_floor: float = 0.0,
) -> tuple[float, float, float, float] | None:
    """Fit a baseline trend on the prior points, return the latest point's
    z-score against that trend: ``(z_latest, observed, expected, sigma)``.

    Excluding the latest point from the fit is what makes this an outlier
    test rather than a goodness-of-fit test — otherwise a large kick at the
    final epoch flexes the slope to accommodate itself and slips past the
    residual threshold.

    Returns ``None`` when the residual standard deviation collapses to the
    floating-point noise floor (no signal to anomaly-detect against) or the
    inputs aren't usable.
    """
    if len(epochs) < 4 or len(values) != len(epochs):
        # Need ≥ 4 so the prior window (which excludes the latest) has ≥ 3
        # points — the minimum for a meaningful linear fit + residual std.
        return None

    series = _unwrap_degrees(values) if is_angular else values
    if not np.all(np.isfinite(series)):
        return None

    prior_epochs = epochs[:-1]
    prior_series = series[:-1]
    slope, intercept = np.polyfit(prior_epochs, prior_series, 1)
    prior_predicted = slope * prior_epochs + intercept
    prior_residuals = prior_series - prior_predicted
    sigma = float(np.std(prior_residuals))
    if not math.isfinite(sigma) or sigma <= 0.0:
        return None

    # Numerical floor: if sigma is at the FP-noise level relative to the
    # value magnitude, the trend fit absorbed everything and a "z-score"
    # is meaningless. ``1e-9 * |median|`` is well below TLE precision for
    # every element we scan (mean_motion ~ 1e-7, inclination ~ 1e-4, etc.).
    median_abs = float(np.median(np.abs(series))) or 1.0
    if sigma < 1e-9 * median_abs:
        return None

    # Empirical noise floor: a 7-day linear fit on Starlink TLEs leaves
    # residuals much smaller than real day-to-day variability (drag,
    # short-period J2, etc.). Without this floor every routine drift fires
    # at thousands of sigma. ``max(empirical, floor)`` keeps the detector
    # sensitive to true outliers in noisy windows while reining in the
    # quiet-window false positives.
    effective_sigma = max(sigma, noise_floor)

    expected_unwrapped = float(slope * epochs[-1] + intercept)
    latest_residual = float(series[-1]) - expected_unwrapped
    z_latest = latest_residual / effective_sigma
    observed = float(values[-1])
    expected = expected_unwrapped % 360.0 if is_angular else expected_unwrapped
    return z_latest, observed, expected, effective_sigma


def assess(
    *,
    norad_id: int,
    name: str,
    snapshots: list[dict],
    notable_threshold: float = DEFAULT_NOTABLE_THRESHOLD,
) -> ResidualFinding | None:
    """Run the residual scan on a pre-loaded snapshot list.

    ``snapshots`` must be a chronological list of dicts/Rows containing
    ``epoch`` and each of the keys in :data:`ELEMENTS`. Pure-ish (only
    depends on numpy) so it tests cleanly without a DB.
    """
    if len(snapshots) < 2:
        return None

    epochs = np.array([s["epoch"] for s in snapshots], dtype=float)
    window_days = float(epochs[-1] - epochs[0])

    best: tuple[str, float, float, float, float] | None = None
    for elem in ELEMENTS:
        values = np.array([s[elem] for s in snapshots], dtype=float)
        result = _z_for_element(
            epochs, values,
            is_angular=elem in _ANGULAR_ELEMENTS,
            noise_floor=NOISE_FLOOR.get(elem, 0.0),
        )
        if result is None:
            continue
        z, observed, expected, sigma = result
        if best is None or abs(z) > abs(best[1]):
            best = (elem, z, observed, expected, sigma)

    if best is None:
        return None

    elem, z, observed, expected, sigma = best
    severity = classify_severity(abs(z))
    if abs(z) < notable_threshold:
        return ResidualFinding(
            norad_id=norad_id, name=name,
            epoch=float(epochs[-1]),
            snapshots_used=len(snapshots),
            window_days=window_days,
            element=elem, z_score=z,
            observed=observed, expected=expected, sigma=sigma,
            severity="quiet",
        )

    return ResidualFinding(
        norad_id=norad_id, name=name,
        epoch=float(epochs[-1]),
        snapshots_used=len(snapshots),
        window_days=window_days,
        element=elem, z_score=z,
        observed=observed, expected=expected, sigma=sigma,
        severity=severity,
    )


def assess_satellite(
    conn: sqlite3.Connection,
    norad_id: int,
    *,
    window_days: float = DEFAULT_WINDOW_DAYS,
    min_snapshots: int = DEFAULT_MIN_SNAPSHOTS,
    notable_threshold: float = DEFAULT_NOTABLE_THRESHOLD,
) -> ResidualFinding | None:
    """Assess one satellite's recent history."""
    cutoff_jd_row = conn.execute(
        "SELECT MAX(epoch) AS latest FROM tle_snapshots WHERE norad_id = ?",
        (norad_id,),
    ).fetchone()
    if cutoff_jd_row is None or cutoff_jd_row["latest"] is None:
        return None
    latest_jd = float(cutoff_jd_row["latest"])
    window_start = latest_jd - window_days

    rows = conn.execute(
        """
        SELECT s.name, t.epoch, t.mean_motion, t.eccentricity,
               t.inclination, t.raan, t.arg_perigee
        FROM tle_snapshots t
        JOIN satellites s ON s.norad_id = t.norad_id
        WHERE t.norad_id = ?
          AND t.epoch >= ?
        ORDER BY t.epoch ASC
        """,
        (norad_id, window_start),
    ).fetchall()
    if len(rows) < min_snapshots:
        return None

    name = rows[0]["name"] or f"NORAD-{norad_id}"
    snapshots = [dict(r) for r in rows]
    return assess(
        norad_id=norad_id, name=name,
        snapshots=snapshots,
        notable_threshold=notable_threshold,
    )


def scan(
    conn: sqlite3.Connection,
    *,
    window_days: float = DEFAULT_WINDOW_DAYS,
    min_snapshots: int = DEFAULT_MIN_SNAPSHOTS,
    notable_threshold: float = DEFAULT_NOTABLE_THRESHOLD,
    min_severity: Severity = "notable",
    constellation: str | None = "starlink",
) -> list[ResidualFinding]:
    """Run the residual scan across every tracked satellite.

    Returns flagged sats at or above ``min_severity``, sorted with the most
    extreme deviations first.
    """
    if constellation:
        rows = conn.execute(
            "SELECT norad_id FROM satellites WHERE constellation = ?",
            (constellation,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT norad_id FROM satellites").fetchall()

    threshold = _SEVERITY_ORDER[min_severity]
    flagged: list[ResidualFinding] = []
    for r in rows:
        finding = assess_satellite(
            conn, r["norad_id"],
            window_days=window_days,
            min_snapshots=min_snapshots,
            notable_threshold=notable_threshold,
        )
        if finding is None:
            continue
        if _SEVERITY_ORDER[finding.severity] >= threshold:
            flagged.append(finding)

    flagged.sort(key=lambda f: (-_SEVERITY_ORDER[f.severity], -abs(f.z_score)))
    return flagged
