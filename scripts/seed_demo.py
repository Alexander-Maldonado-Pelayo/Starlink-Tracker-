"""Seed the local SQLite DB from the bundled Starlink snapshot — no network.

The dashboard bootstraps itself from CelesTrak on first run, but that needs
outbound access to celestrak.org. For a self-contained demo (conference wifi,
locked-down network, offline laptop), this script loads the ~10k-satellite
snapshot committed to the repo straight into ``data/spacetrack.db`` so the
dashboard renders instantly without waiting on — or requiring — a live fetch.

Idempotent: snapshots dedupe on (norad_id, epoch), so re-running is a no-op.

Usage:
    python scripts/seed_demo.py
"""

from __future__ import annotations

from pathlib import Path

from spacetrack.storage import db
from spacetrack.storage.snapshot import write_snapshots
from spacetrack.tle.fetcher import load_bundled_seed, now_unix

DB_PATH = Path("data/spacetrack.db")


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.init_db(DB_PATH)

    tles = load_bundled_seed()
    with db.session(DB_PATH) as conn:
        new_count, total = write_snapshots(
            conn, tles, fetched_at=now_unix(), constellation="starlink"
        )

    print(f"Seeded {new_count}/{total} Starlink TLEs into {DB_PATH}")
    if new_count == 0 and total > 0:
        print("(database was already populated — nothing new to add)")
    print("Ready. Launch the dashboard with:  spacetrack dashboard")


if __name__ == "__main__":
    main()
