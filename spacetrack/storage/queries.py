"""Read-only queries for satellites and TLE snapshots."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LatestTLE:
    norad_id: int
    name: str
    line1: str
    line2: str
    epoch: float       # Julian date
    fetched_at: int    # unix ts


def find_satellite(conn: sqlite3.Connection, query: str) -> int | None:
    """Resolve a user-supplied identifier to a NORAD ID.

    Accepts either a numeric NORAD ID or a (case-insensitive) name.
    Returns None if no match.
    """
    q = query.strip()
    if q.isdigit():
        row = conn.execute(
            "SELECT norad_id FROM satellites WHERE norad_id = ?", (int(q),)
        ).fetchone()
        return row["norad_id"] if row else None

    row = conn.execute(
        "SELECT norad_id FROM satellites WHERE UPPER(name) = UPPER(?)", (q,)
    ).fetchone()
    if row:
        return row["norad_id"]

    # Fall back to a LIKE match if there's exactly one hit.
    rows = conn.execute(
        "SELECT norad_id, name FROM satellites WHERE UPPER(name) LIKE UPPER(?) LIMIT 2",
        (f"%{q}%",),
    ).fetchall()
    if len(rows) == 1:
        return rows[0]["norad_id"]
    return None


def get_latest_tle(conn: sqlite3.Connection, norad_id: int) -> LatestTLE | None:
    row = conn.execute(
        """
        SELECT s.norad_id, s.name, t.line1, t.line2, t.epoch, t.fetched_at
        FROM tle_snapshots t
        JOIN satellites s ON s.norad_id = t.norad_id
        WHERE t.norad_id = ?
        ORDER BY t.epoch DESC
        LIMIT 1
        """,
        (norad_id,),
    ).fetchone()
    if row is None:
        return None
    return LatestTLE(
        norad_id=row["norad_id"],
        name=row["name"],
        line1=row["line1"],
        line2=row["line2"],
        epoch=row["epoch"],
        fetched_at=row["fetched_at"],
    )


@dataclass(frozen=True)
class AnomalyRecord:
    id: int
    detected_at: int       # unix ts when the scan recorded it
    type: str              # conjunction | maneuver | inspector | decay
    severity: str
    primary_id: int | None
    secondary_id: int | None
    details: dict[str, Any]


def recent_anomalies(
    conn: Any,
    *,
    types: list[str] | None = None,
    since_unix: int | None = None,
    limit: int | None = 500,
) -> list[AnomalyRecord]:
    """Read persisted anomalies for the dashboard feed, newest first.

    Optionally filter by detector ``types`` and a ``since_unix`` lower bound on
    ``detected_at``. ``details`` is decoded from its stored JSON.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if types:
        placeholders = ",".join("?" for _ in types)
        clauses.append(f"type IN ({placeholders})")
        params.extend(types)
    if since_unix is not None:
        clauses.append("detected_at >= ?")
        params.append(since_unix)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT id, detected_at, type, severity, primary_id, secondary_id, details "
        f"FROM anomalies {where} ORDER BY detected_at DESC, id DESC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    out: list[AnomalyRecord] = []
    for row in conn.execute(sql, tuple(params)).fetchall():
        raw = row["details"]
        try:
            details = json.loads(raw) if raw else {}
        except (ValueError, TypeError):
            details = {}
        out.append(
            AnomalyRecord(
                id=row["id"],
                detected_at=row["detected_at"],
                type=row["type"],
                severity=row["severity"],
                primary_id=row["primary_id"],
                secondary_id=row["secondary_id"],
                details=details,
            )
        )
    return out


def anomaly_counts_by_type(
    conn: Any, *, since_unix: int | None = None
) -> dict[str, int]:
    """Return {type: count} of persisted anomalies, optionally since a time."""
    if since_unix is not None:
        rows = conn.execute(
            "SELECT type, COUNT(*) AS n FROM anomalies "
            "WHERE detected_at >= ? GROUP BY type",
            (since_unix,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT type, COUNT(*) AS n FROM anomalies GROUP BY type"
        ).fetchall()
    return {row["type"]: row["n"] for row in rows}
