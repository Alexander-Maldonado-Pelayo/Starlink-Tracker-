"""SQLite schema and connection helpers.

The full schema is defined here from day one because the Phase 3 anomaly
detectors (maneuver, conjunction, inspector, decay) all read from the
tle_snapshots history. Designing the schema upfront avoids painful migrations.

Dual-mode connection:

* **Local SQLite** (default) — stdlib ``sqlite3`` against a file path. Used
  for local development, tests, and the CLI when no Turso credentials are
  present. Zero external dependencies.
* **Turso** (production) — remote libsql when ``TURSO_URL`` and
  ``TURSO_AUTH_TOKEN`` are set. Streamlit Cloud reads from this; a scheduled
  GitHub Action writes fresh TLEs into it every two hours. The connection
  returns a thin wrapper that exposes the same ``execute``/``fetchall``
  surface as ``sqlite3.Connection``, including dict-style row access by
  column name, so callers don't branch on backend.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path("data/spacetrack.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tle_snapshots (
    norad_id      INTEGER NOT NULL,
    epoch         REAL    NOT NULL,
    fetched_at    INTEGER NOT NULL,
    line1         TEXT    NOT NULL,
    line2         TEXT    NOT NULL,
    inclination   REAL,
    raan          REAL,
    eccentricity  REAL,
    arg_perigee   REAL,
    mean_anomaly  REAL,
    mean_motion   REAL,
    PRIMARY KEY (norad_id, epoch)
);

CREATE INDEX IF NOT EXISTS idx_tle_norad_epoch
    ON tle_snapshots(norad_id, epoch);

CREATE INDEX IF NOT EXISTS idx_tle_fetched_at
    ON tle_snapshots(fetched_at);

CREATE TABLE IF NOT EXISTS satellites (
    norad_id      INTEGER PRIMARY KEY,
    name          TEXT,
    country       TEXT,
    launch_date   TEXT,
    object_type   TEXT,
    constellation TEXT
);

CREATE INDEX IF NOT EXISTS idx_satellites_constellation
    ON satellites(constellation);

CREATE TABLE IF NOT EXISTS anomalies (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at  INTEGER NOT NULL,
    type         TEXT    NOT NULL,
    severity     TEXT    NOT NULL,
    primary_id   INTEGER,
    secondary_id INTEGER,
    details      TEXT,
    -- Stable per-event key so re-running `scan` on every refresh doesn't
    -- duplicate the same conjunction/maneuver in the timeline. See
    -- spacetrack.anomaly.persist for how each detector's fingerprint is built.
    fingerprint  TEXT
);

CREATE INDEX IF NOT EXISTS idx_anomalies_detected_at
    ON anomalies(detected_at);

CREATE INDEX IF NOT EXISTS idx_anomalies_type
    ON anomalies(type);
"""
# Note: the UNIQUE index on anomalies(fingerprint) is created by
# _migrate_anomalies_fingerprint, not here — a database created before the
# fingerprint column existed would fail this statement inside executescript()
# before the ALTER TABLE has a chance to run.


def _migrate_anomalies_fingerprint(conn: Any) -> None:
    """Add the ``fingerprint`` column to a pre-existing ``anomalies`` table.

    Idempotent: a no-op once the column exists. Needed because the production
    Turso database was created before the fingerprint column existed; a plain
    ``CREATE TABLE IF NOT EXISTS`` won't alter the live table. SQLite/libsql
    both support ``ALTER TABLE ... ADD COLUMN``.
    """
    cols = [row[1] for row in conn.execute("PRAGMA table_info(anomalies)").fetchall()]
    if not cols:
        return  # table doesn't exist yet; SCHEMA will create it with the column
    if "fingerprint" not in cols:
        conn.execute("ALTER TABLE anomalies ADD COLUMN fingerprint TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_anomalies_fingerprint "
        "ON anomalies(fingerprint)"
    )


def _use_turso() -> bool:
    return bool(os.environ.get("TURSO_URL") and os.environ.get("TURSO_AUTH_TOKEN"))


# ---------------------------------------------------------------------------
# Turso adapter: thin wrapper around libsql.Connection that adds
# sqlite3.Row-compatible row access by column name. libsql's native rows are
# plain tuples; this wrapper turns them into objects supporting both
# ``row[0]`` and ``row['column']``.
# ---------------------------------------------------------------------------


class _Row:
    """Dict-like row matching sqlite3.Row's surface."""

    __slots__ = ("_values", "_keys")

    def __init__(self, values: tuple, keys: tuple[str, ...]) -> None:
        self._values = values
        self._keys = keys

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        try:
            idx = self._keys.index(key)
        except ValueError as exc:
            raise KeyError(key) from exc
        return self._values[idx]

    def keys(self) -> tuple[str, ...]:
        return self._keys

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={v!r}" for k, v in zip(self._keys, self._values))
        return f"_Row({pairs})"


class _Cursor:
    """Cursor wrapper that promotes tuple rows to _Row."""

    def __init__(self, real) -> None:
        self._real = real

    @property
    def description(self):
        return self._real.description

    @property
    def rowcount(self) -> int:
        return self._real.rowcount

    @property
    def lastrowid(self):
        return self._real.lastrowid

    def _keys(self) -> tuple[str, ...]:
        desc = self._real.description
        return tuple(d[0] for d in desc) if desc else ()

    def fetchone(self):
        row = self._real.fetchone()
        if row is None:
            return None
        return _Row(tuple(row), self._keys())

    def fetchall(self):
        rows = self._real.fetchall()
        keys = self._keys()
        return [_Row(tuple(r), keys) for r in rows]

    def fetchmany(self, size: int | None = None):
        rows = self._real.fetchmany(size) if size else self._real.fetchmany()
        keys = self._keys()
        return [_Row(tuple(r), keys) for r in rows]

    def execute(self, *args, **kwargs):
        self._real = self._real.execute(*args, **kwargs)
        return self

    def executemany(self, *args, **kwargs):
        self._real.executemany(*args, **kwargs)
        return self

    def close(self) -> None:
        self._real.close()

    def __iter__(self):
        return iter(self.fetchall())


class _Connection:
    """libsql.Connection wrapper that produces _Row results."""

    def __init__(self, real) -> None:
        self._real = real

    def execute(self, *args, **kwargs) -> _Cursor:
        return _Cursor(self._real.execute(*args, **kwargs))

    def executemany(self, *args, **kwargs):
        self._real.executemany(*args, **kwargs)

    def executescript(self, script: str):
        self._real.executescript(script)

    def cursor(self) -> _Cursor:
        return _Cursor(self._real.cursor())

    def commit(self) -> None:
        self._real.commit()

    def rollback(self) -> None:
        self._real.rollback()

    def close(self) -> None:
        self._real.close()

    @property
    def in_transaction(self) -> bool:
        return self._real.in_transaction


def _connect_turso():
    """Open a libsql connection to Turso. Imported lazily — local dev and
    tests never need libsql installed."""
    import libsql  # type: ignore[import-not-found]

    url = os.environ["TURSO_URL"]
    token = os.environ["TURSO_AUTH_TOKEN"]
    raw = libsql.connect(database=url, auth_token=token)
    return _Connection(raw)


def connect(db_path: Path = DEFAULT_DB_PATH) -> Any:
    """Open a database connection. Uses Turso when configured, else SQLite.

    The returned object exposes the same surface either way — callers can
    treat it as a ``sqlite3.Connection`` (``execute`` returns a cursor whose
    rows support both index and column-name access).
    """
    if _use_turso():
        return _connect_turso()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate_anomalies_fingerprint(conn)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def session(db_path: Path = DEFAULT_DB_PATH) -> Iterator[Any]:
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
