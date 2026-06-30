"""Run every detector and persist findings into the ``anomalies`` table.

The four detectors (conjunction, maneuver, inspector, decay) are *ephemeral*
on their own — they answer "what looks wrong right now?" and print. This module
turns those point-in-time answers into an accumulating, queryable record so the
dashboard (and a Space-Domain-Awareness analyst) can ask the bigger question:
"what has this constellation been doing over the past weeks?"

Idempotency
-----------
``scan`` is meant to run on a schedule (every few hours, from the refresh
GitHub Action). The same predicted conjunction or already-flagged maneuver will
surface on consecutive runs, so each finding carries a deterministic
``fingerprint`` and rows are inserted with ``INSERT OR IGNORE`` against a unique
index. Re-running a scan with no new TLE data is a no-op; the timeline grows
only when genuinely new events appear.

Fingerprint design (one row per real-world event, not per scan):
  * conjunction — primary, secondary, and TCA bucketed to the hour
  * maneuver    — satellite and the epoch of the *later* TLE in the pair
  * inspector   — satellite, latest-snapshot epoch, and worst element
  * decay       — satellite, UTC day, and risk tier (decay is a slow state,
                  so we record at most one row per sat per day per tier)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from spacetrack.anomaly import conjunction as conjunction_mod
from spacetrack.anomaly import decay as decay_mod
from spacetrack.anomaly import inspector as inspector_mod
from spacetrack.anomaly import maneuver as maneuver_mod

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AnomalyRow:
    type: str
    severity: str
    primary_id: int | None
    secondary_id: int | None
    details: dict[str, Any]
    fingerprint: str


# ---------------------------------------------------------------------------
# Detector-event -> anomaly-row mappers
# ---------------------------------------------------------------------------


def _conjunction_row(ev: conjunction_mod.ConjunctionEvent) -> _AnomalyRow:
    hour_bucket = int(ev.tca_unix // 3600)
    return _AnomalyRow(
        type="conjunction",
        severity=ev.risk,
        primary_id=ev.primary_norad_id,
        secondary_id=ev.secondary_norad_id,
        details={
            "primary_name": ev.primary_name,
            "secondary_name": ev.secondary_name,
            "secondary_constellation": ev.secondary_constellation,
            "miss_distance_km": round(ev.miss_distance_km, 4),
            "relative_velocity_km_s": round(ev.relative_velocity_km_s, 4),
            "tca_unix": ev.tca_unix,
            "tca_utc": datetime.fromtimestamp(ev.tca_unix, tz=timezone.utc).isoformat(),
        },
        fingerprint=f"conj:{ev.primary_norad_id}:{ev.secondary_norad_id}:{hour_bucket}",
    )


def _maneuver_row(ev: maneuver_mod.ManeuverEvent) -> _AnomalyRow:
    return _AnomalyRow(
        type="maneuver",
        severity=ev.magnitude,
        primary_id=ev.norad_id,
        secondary_id=None,
        details={
            "name": ev.name,
            "direction": ev.direction,
            "delta_a_km": round(ev.delta_a_km, 4),
            "delta_a_km_per_day": round(ev.delta_a_km_per_day, 4),
            "altitude_before_km": round(ev.altitude_before_km, 3),
            "altitude_after_km": round(ev.altitude_after_km, 3),
            "gap_days": round(ev.gap_days, 4),
            "epoch_before": ev.epoch_before,
            "epoch_after": ev.epoch_after,
        },
        fingerprint=f"manv:{ev.norad_id}:{ev.epoch_after:.5f}",
    )


def _inspector_row(f: inspector_mod.ResidualFinding) -> _AnomalyRow:
    return _AnomalyRow(
        type="inspector",
        severity=f.severity,
        primary_id=f.norad_id,
        secondary_id=None,
        details={
            "name": f.name,
            "element": f.element,
            "z_score": round(f.z_score, 4),
            "observed": f.observed,
            "expected": f.expected,
            "sigma": f.sigma,
            "snapshots_used": f.snapshots_used,
            "window_days": round(f.window_days, 4),
            "epoch": f.epoch,
        },
        fingerprint=f"insp:{f.norad_id}:{f.epoch:.5f}:{f.element}",
    )


def _decay_row(a: decay_mod.DecayAssessment, day: str) -> _AnomalyRow:
    return _AnomalyRow(
        type="decay",
        severity=a.risk,
        primary_id=a.norad_id,
        secondary_id=None,
        details={
            "name": a.name,
            "perigee_km": round(a.perigee_km, 3),
            "apogee_km": round(a.apogee_km, 3),
            "altitude_km": round(a.altitude_km, 3),
            "decay_rate_km_per_day": (
                round(a.decay_rate_km_per_day, 4)
                if a.decay_rate_km_per_day is not None
                else None
            ),
            "days_to_reentry": (
                round(a.days_to_reentry, 2) if a.days_to_reentry is not None else None
            ),
            "epoch": a.epoch,
        },
        fingerprint=f"decay:{a.norad_id}:{day}:{a.risk}",
    )


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------


def _insert(conn: Any, detected_at: int, rows: list[_AnomalyRow]) -> int:
    """Insert anomaly rows, skipping ones whose fingerprint already exists.

    Returns the number of genuinely new rows written.
    """
    if not rows:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO anomalies
            (detected_at, type, severity, primary_id, secondary_id, details, fingerprint)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                detected_at,
                r.type,
                r.severity,
                r.primary_id,
                r.secondary_id,
                json.dumps(r.details, separators=(",", ":")),
                r.fingerprint,
            )
            for r in rows
        ],
    )
    after = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
    return after - before


@dataclass(frozen=True)
class ScanSummary:
    detected_at: int
    new_by_type: dict[str, int]
    found_by_type: dict[str, int]

    @property
    def total_new(self) -> int:
        return sum(self.new_by_type.values())


def run_scan(
    conn: Any,
    *,
    detected_at: int,
    run_conjunction: bool = True,
    run_maneuver: bool = True,
    run_inspector: bool = True,
    run_decay: bool = True,
    conjunction_min_risk: str = "medium",
    conjunction_primary_limit: int | None = None,
    conjunction_primary_offset: int = 0,
    maneuver_min_magnitude: str = "small",
    inspector_min_severity: str = "notable",
    decay_min_risk: str = "elevated",
) -> ScanSummary:
    """Run the enabled detectors and persist their findings.

    ``detected_at`` is the unix timestamp stamped on every row produced by this
    run (the moment of detection — what the timeline is ordered by). Detector
    thresholds default to the "report it" tiers so the feed isn't flooded with
    nominal/quiet states, but every threshold is overridable.
    """
    new_by_type: dict[str, int] = {}
    found_by_type: dict[str, int] = {}

    if run_maneuver:
        events = maneuver_mod.scan(conn, min_magnitude=maneuver_min_magnitude)  # type: ignore[arg-type]
        found_by_type["maneuver"] = len(events)
        new_by_type["maneuver"] = _insert(
            conn, detected_at, [_maneuver_row(e) for e in events]
        )

    if run_inspector:
        findings = inspector_mod.scan(conn, min_severity=inspector_min_severity)  # type: ignore[arg-type]
        found_by_type["inspector"] = len(findings)
        new_by_type["inspector"] = _insert(
            conn, detected_at, [_inspector_row(f) for f in findings]
        )

    if run_decay:
        assessments = decay_mod.scan(conn, min_risk=decay_min_risk)  # type: ignore[arg-type]
        found_by_type["decay"] = len(assessments)
        day = datetime.fromtimestamp(detected_at, tz=timezone.utc).strftime("%Y-%m-%d")
        new_by_type["decay"] = _insert(
            conn, detected_at, [_decay_row(a, day) for a in assessments]
        )

    if run_conjunction:
        # Conjunction is the heaviest detector (propagation over every
        # primary x candidate). It runs last so a slow/aborted scan still
        # persists the cheap detectors' results.
        other_count = conn.execute(
            "SELECT COUNT(*) FROM satellites WHERE constellation = 'other'"
        ).fetchone()[0]
        if other_count == 0:
            log.warning(
                "conjunction skipped: no external catalog loaded "
                "(run `spacetrack update-catalog`)."
            )
            found_by_type["conjunction"] = 0
            new_by_type["conjunction"] = 0
        else:
            events = conjunction_mod.scan(
                conn,
                min_risk=conjunction_min_risk,  # type: ignore[arg-type]
                primary_limit=conjunction_primary_limit,
                primary_offset=conjunction_primary_offset,
            )
            found_by_type["conjunction"] = len(events)
            new_by_type["conjunction"] = _insert(
                conn, detected_at, [_conjunction_row(e) for e in events]
            )

    return ScanSummary(
        detected_at=detected_at,
        new_by_type=new_by_type,
        found_by_type=found_by_type,
    )
