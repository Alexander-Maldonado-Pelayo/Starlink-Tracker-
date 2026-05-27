"""Persist parsed TLEs into the snapshot history table.

Idempotent: the (norad_id, epoch) primary key means re-running an update
without a fresh CelesTrak refresh just no-ops on the conflicting rows.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable

from spacetrack.tle.parser import ParsedTLE

log = logging.getLogger(__name__)


def upsert_satellite(conn: sqlite3.Connection, tle: ParsedTLE, constellation: str) -> None:
    conn.execute(
        """
        INSERT INTO satellites (norad_id, name, constellation)
        VALUES (?, ?, ?)
        ON CONFLICT(norad_id) DO UPDATE SET
            name = excluded.name,
            constellation = excluded.constellation
        """,
        (tle.norad_id, tle.name, constellation),
    )


def insert_snapshot(
    conn: sqlite3.Connection, tle: ParsedTLE, fetched_at: int
) -> bool:
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO tle_snapshots (
            norad_id, epoch, fetched_at, line1, line2,
            inclination, raan, eccentricity,
            arg_perigee, mean_anomaly, mean_motion
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tle.norad_id,
            tle.epoch_jd,
            fetched_at,
            tle.line1,
            tle.line2,
            tle.inclination,
            tle.raan,
            tle.eccentricity,
            tle.arg_perigee,
            tle.mean_anomaly,
            tle.mean_motion,
        ),
    )
    return cur.rowcount > 0


def write_snapshots(
    conn: sqlite3.Connection,
    tles: Iterable[ParsedTLE],
    *,
    fetched_at: int,
    constellation: str,
) -> tuple[int, int]:
    new_count = 0
    total = 0
    for tle in tles:
        upsert_satellite(conn, tle, constellation)
        if insert_snapshot(conn, tle, fetched_at):
            new_count += 1
        total += 1

    log.info("Persisted %d new TLEs (out of %d fetched)", new_count, total)
    return new_count, total


def write_external_catalog(
    conn: sqlite3.Connection,
    tles: Iterable[ParsedTLE],
    *,
    fetched_at: int,
    constellation: str = "other",
    skip_constellation: str = "starlink",
) -> tuple[int, int, int]:
    """Ingest a non-Starlink catalog for conjunction screening.

    Skips any TLE whose ``norad_id`` is already tagged with
    ``skip_constellation`` so the upsert doesn't clobber the Starlink label
    on overlapping rows (CelesTrak's ``active`` group includes Starlink).

    Returns ``(new_snapshots, total_seen, skipped)``.
    """
    starlink_ids = {
        row["norad_id"]
        for row in conn.execute(
            "SELECT norad_id FROM satellites WHERE constellation = ?",
            (skip_constellation,),
        )
    }

    new_count = 0
    total = 0
    skipped = 0
    for tle in tles:
        total += 1
        if tle.norad_id in starlink_ids:
            skipped += 1
            continue
        upsert_satellite(conn, tle, constellation)
        if insert_snapshot(conn, tle, fetched_at):
            new_count += 1

    log.info(
        "External catalog: %d new TLEs out of %d (skipped %d already-tracked sats)",
        new_count, total, skipped,
    )
    return new_count, total, skipped
