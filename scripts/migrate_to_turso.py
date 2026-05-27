"""One-shot migration: copy local SQLite DB to Turso.

Reads ``data/spacetrack.db`` (or whatever you pass with ``--source``) and
writes its contents into a Turso database. Idempotent — re-running it just
skips rows that are already present (snapshots dedupe on (norad_id, epoch);
satellites upsert by primary key).

Usage:
    set TURSO_URL=libsql://your-db.turso.io
    set TURSO_AUTH_TOKEN=eyJh...
    python scripts/migrate_to_turso.py

Both env vars are required. Optional flags:
    --source data/spacetrack.db   Local DB path (default).
    --batch-size 5000             Rows per executemany batch.
    --skip-snapshots              Migrate satellites only (faster smoke test).

Run from the repo root with the project venv active.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# Make sure we can import the project even when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacetrack.storage import db as db_mod


_SATELLITES_COLUMNS = (
    "norad_id", "name", "country", "launch_date", "object_type", "constellation",
)
_SNAPSHOT_COLUMNS = (
    "norad_id", "epoch", "fetched_at", "line1", "line2",
    "inclination", "raan", "eccentricity",
    "arg_perigee", "mean_anomaly", "mean_motion",
)


def _require_turso_env() -> None:
    missing = [
        v for v in ("TURSO_URL", "TURSO_AUTH_TOKEN")
        if not os.environ.get(v)
    ]
    if missing:
        sys.exit(
            f"error: missing env vars: {', '.join(missing)}. "
            "Set them to your Turso database URL and auth token before running."
        )


def _open_source(source: Path) -> sqlite3.Connection:
    if not source.exists():
        sys.exit(f"error: source DB not found at {source}")
    conn = sqlite3.connect(source)
    conn.row_factory = sqlite3.Row
    return conn


def _write_batch(sql: str, batch: list[tuple]) -> None:
    """Open a fresh Turso connection, run one batched write, close.

    Turso (Hrana) evicts idle HTTP streams between calls, so a single
    long-lived connection survives one batch and then 404s on the next
    (``stream not found``). Reconnecting per batch sidesteps the lifecycle
    issue entirely — the per-batch latency cost is dwarfed by the network
    write itself.
    """
    conn = db_mod.connect()
    try:
        cur = conn.cursor()
        cur.executemany(sql, batch)
        conn.commit()
    finally:
        conn.close()


def _migrate_satellites(src: sqlite3.Connection, batch_size: int) -> int:
    rows = src.execute(
        f"SELECT {', '.join(_SATELLITES_COLUMNS)} FROM satellites"
    ).fetchall()
    total = len(rows)
    if total == 0:
        print("satellites: source is empty, nothing to do")
        return 0

    placeholders = ", ".join("?" for _ in _SATELLITES_COLUMNS)
    update_set = ", ".join(
        f"{c} = excluded.{c}" for c in _SATELLITES_COLUMNS if c != "norad_id"
    )
    sql = (
        f"INSERT INTO satellites ({', '.join(_SATELLITES_COLUMNS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(norad_id) DO UPDATE SET {update_set}"
    )

    t0 = time.time()
    for i in range(0, total, batch_size):
        batch = [tuple(r) for r in rows[i:i + batch_size]]
        _write_batch(sql, batch)
        print(f"  satellites: {min(i + batch_size, total):>6}/{total} "
              f"({(min(i + batch_size, total) / total * 100):5.1f}%)")
    print(f"satellites: {total} rows in {time.time() - t0:.1f}s")
    return total


def _migrate_snapshots(src: sqlite3.Connection, batch_size: int) -> int:
    total = src.execute("SELECT COUNT(*) FROM tle_snapshots").fetchone()[0]
    if total == 0:
        print("snapshots: source is empty, nothing to do")
        return 0

    placeholders = ", ".join("?" for _ in _SNAPSHOT_COLUMNS)
    sql = (
        f"INSERT OR IGNORE INTO tle_snapshots ({', '.join(_SNAPSHOT_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )

    t0 = time.time()
    # Stream from source so we don't load all 144k rows into memory at once.
    src_cur = src.execute(
        f"SELECT {', '.join(_SNAPSHOT_COLUMNS)} FROM tle_snapshots "
        f"ORDER BY norad_id, epoch"
    )
    pushed = 0
    while True:
        batch_rows = src_cur.fetchmany(batch_size)
        if not batch_rows:
            break
        _write_batch(sql, [tuple(r) for r in batch_rows])
        pushed += len(batch_rows)
        pct = pushed / total * 100
        elapsed = time.time() - t0
        rate = pushed / elapsed if elapsed > 0 else 0.0
        eta = (total - pushed) / rate if rate > 0 else 0.0
        print(
            f"  snapshots: {pushed:>7}/{total} ({pct:5.1f}%) "
            f"@ {rate:6.0f} rows/s  ETA {eta:5.0f}s"
        )
    print(f"snapshots: {pushed} rows in {time.time() - t0:.1f}s")
    return pushed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/spacetrack.db"),
        help="Local SQLite source database",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000,
        help="Rows per executemany batch",
    )
    parser.add_argument(
        "--skip-snapshots", action="store_true",
        help="Migrate satellites only (smoke test)",
    )
    args = parser.parse_args()

    _require_turso_env()

    print(f"Source: {args.source.resolve()}")
    print(f"Target: {os.environ['TURSO_URL']}")
    src = _open_source(args.source)

    # Bootstrap the schema on the target before touching data. Each write
    # batch opens its own connection to dodge Turso's stream-eviction; this
    # init call is the same pattern at one batch.
    print("Initializing schema on Turso...")
    db_mod.init_db()  # uses Turso because env vars are set

    try:
        sats = _migrate_satellites(src, args.batch_size)
        snaps = 0
        if not args.skip_snapshots:
            snaps = _migrate_snapshots(src, args.batch_size)
        print()
        print(f"Done. Migrated {sats:,} satellites and {snaps:,} snapshots.")
    finally:
        src.close()


if __name__ == "__main__":
    main()
