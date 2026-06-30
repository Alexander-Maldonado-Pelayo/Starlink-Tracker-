"""Tests for anomaly persistence (spacetrack.anomaly.persist) and the
anomaly query helpers."""

from __future__ import annotations

from pathlib import Path

from spacetrack.anomaly import conjunction, decay, inspector, maneuver, persist
from spacetrack.storage import db
from spacetrack.storage.queries import (
    anomaly_counts_by_type,
    recent_anomalies,
)


# --- fingerprint / mapper unit tests ---------------------------------------

def test_conjunction_fingerprint_buckets_tca_to_the_hour():
    base = conjunction.ConjunctionEvent(
        primary_norad_id=44713, primary_name="STARLINK-1",
        secondary_norad_id=99999, secondary_name="DEBRIS",
        secondary_constellation="other",
        tca_jd=2461170.5, tca_unix=1_800_000_000.0,
        miss_distance_km=1.234, relative_velocity_km_s=12.5, risk="high",
    )
    r1 = persist._conjunction_row(base)
    # Same pair, +30 min (still same hour bucket) -> identical fingerprint.
    base_30m = conjunction.ConjunctionEvent(**{**base.__dict__, "tca_unix": 1_800_000_000.0 + 1800})
    r2 = persist._conjunction_row(base_30m)
    assert r1.fingerprint == r2.fingerprint
    # +2 h -> different bucket.
    base_2h = conjunction.ConjunctionEvent(**{**base.__dict__, "tca_unix": 1_800_000_000.0 + 7200})
    assert persist._conjunction_row(base_2h).fingerprint != r1.fingerprint
    assert r1.type == "conjunction" and r1.severity == "high"
    assert r1.primary_id == 44713 and r1.secondary_id == 99999


def test_maneuver_fingerprint_keys_on_later_epoch():
    ev = maneuver.ManeuverEvent(
        norad_id=44713, name="STARLINK-1",
        epoch_before=2461170.0, epoch_after=2461172.5, gap_days=2.5,
        a_before_km=6900.0, a_after_km=6930.0,
        altitude_before_km=529.0, altitude_after_km=559.0,
        delta_a_km=30.0, delta_a_km_per_day=12.0,
        direction="boost", magnitude="medium",
    )
    row = persist._maneuver_row(ev)
    assert row.fingerprint == "manv:44713:2461172.50000"
    assert row.type == "maneuver" and row.severity == "medium"
    assert row.secondary_id is None
    assert row.details["direction"] == "boost"


def test_inspector_fingerprint_includes_element():
    f = inspector.ResidualFinding(
        norad_id=44713, name="STARLINK-1", epoch=2461172.5,
        snapshots_used=9, window_days=7.0, element="mean_motion",
        z_score=6.1, observed=15.1, expected=15.0, sigma=0.01,
        severity="significant",
    )
    row = persist._inspector_row(f)
    assert row.fingerprint == "insp:44713:2461172.50000:mean_motion"
    assert row.severity == "significant"


def test_decay_fingerprint_buckets_by_day_and_tier():
    a = decay.DecayAssessment(
        norad_id=44713, name="STARLINK-1", epoch=2461172.5,
        mean_motion=16.2, eccentricity=0.0005, semi_major_axis_km=6600.0,
        perigee_km=210.0, apogee_km=240.0, altitude_km=225.0,
        decay_rate_km_per_day=-6.0, days_to_reentry=12.0, risk="high",
    )
    r1 = persist._decay_row(a, "2026-06-30")
    r2 = persist._decay_row(a, "2026-06-30")
    assert r1.fingerprint == r2.fingerprint == "decay:44713:2026-06-30:high"
    # Different day -> different row.
    assert persist._decay_row(a, "2026-07-01").fingerprint != r1.fingerprint


# --- _insert dedup ----------------------------------------------------------

def test_insert_is_idempotent_on_fingerprint(tmp_path: Path):
    p = tmp_path / "t.db"
    db.init_db(p)
    rows = [
        persist._AnomalyRow("maneuver", "large", 1, None, {"x": 1}, "fp-a"),
        persist._AnomalyRow("maneuver", "large", 2, None, {"x": 2}, "fp-b"),
    ]
    with db.session(p) as conn:
        n1 = persist._insert(conn, detected_at=1000, rows=rows)
    assert n1 == 2
    # Re-insert the same fingerprints plus one new -> only the new one lands.
    rows2 = rows + [persist._AnomalyRow("maneuver", "small", 3, None, {}, "fp-c")]
    with db.session(p) as conn:
        n2 = persist._insert(conn, detected_at=2000, rows=rows2)
    assert n2 == 1
    with db.session(p) as conn:
        total = conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]
    assert total == 3


# --- migration --------------------------------------------------------------

def test_migration_adds_fingerprint_to_legacy_table(tmp_path: Path):
    p = tmp_path / "legacy.db"
    # Simulate a pre-fingerprint anomalies table.
    import sqlite3
    conn = sqlite3.connect(p)
    conn.executescript(
        """
        CREATE TABLE anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at INTEGER NOT NULL, type TEXT NOT NULL,
            severity TEXT NOT NULL, primary_id INTEGER,
            secondary_id INTEGER, details TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO anomalies (detected_at, type, severity) VALUES (1, 'decay', 'high')"
    )
    conn.commit()
    conn.close()

    # init_db should migrate it in place without losing the existing row.
    db.init_db(p)
    conn = sqlite3.connect(p)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(anomalies)").fetchall()]
    assert "fingerprint" in cols
    assert conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0] == 1
    # Migration is idempotent.
    conn.close()
    db.init_db(p)


# --- query helpers ----------------------------------------------------------

def _seed_anomalies(p: Path) -> None:
    rows = [
        persist._AnomalyRow("conjunction", "high", 1, 99, {"miss_distance_km": 0.4}, "c1"),
        persist._AnomalyRow("maneuver", "large", 2, None, {"direction": "boost"}, "m1"),
        persist._AnomalyRow("decay", "imminent", 3, None, {"perigee_km": 180.0}, "d1"),
    ]
    with db.session(p) as conn:
        persist._insert(conn, detected_at=1000, rows=rows[:2])
        persist._insert(conn, detected_at=2000, rows=rows[2:])


def test_recent_anomalies_orders_and_decodes(tmp_path: Path):
    p = tmp_path / "q.db"
    db.init_db(p)
    _seed_anomalies(p)
    with db.session(p) as conn:
        recs = recent_anomalies(conn)
    assert [r.type for r in recs] == ["decay", "conjunction", "maneuver"] or \
           [r.type for r in recs][0] == "decay"  # newest detected_at first
    decay_rec = next(r for r in recs if r.type == "decay")
    assert decay_rec.details["perigee_km"] == 180.0


def test_recent_anomalies_filters_by_type_and_since(tmp_path: Path):
    p = tmp_path / "q2.db"
    db.init_db(p)
    _seed_anomalies(p)
    with db.session(p) as conn:
        only_conj = recent_anomalies(conn, types=["conjunction"])
        recent = recent_anomalies(conn, since_unix=1500)
    assert {r.type for r in only_conj} == {"conjunction"}
    assert {r.type for r in recent} == {"decay"}  # only the detected_at=2000 row


def test_anomaly_counts_by_type(tmp_path: Path):
    p = tmp_path / "q3.db"
    db.init_db(p)
    _seed_anomalies(p)
    with db.session(p) as conn:
        counts = anomaly_counts_by_type(conn)
    assert counts == {"conjunction": 1, "maneuver": 1, "decay": 1}


# --- end-to-end scan via run_scan ------------------------------------------

def _seed_maneuvering_sat(p: Path) -> None:
    """One Starlink sat with two snapshots whose mean motion jumps — a clear
    orbit-raise maneuver."""
    with db.session(p) as conn:
        conn.execute(
            "INSERT INTO satellites (norad_id, name, constellation) "
            "VALUES (44713, 'STARLINK-TEST', 'starlink')"
        )
        for epoch, mm in [(2461170.0, 15.50), (2461172.0, 15.05)]:
            conn.execute(
                """
                INSERT INTO tle_snapshots
                    (norad_id, epoch, fetched_at, line1, line2, mean_motion)
                VALUES (44713, ?, ?, 'L1', 'L2', ?)
                """,
                (epoch, int(epoch), mm),
            )


def test_run_scan_persists_and_is_idempotent(tmp_path: Path):
    p = tmp_path / "scan.db"
    db.init_db(p)
    _seed_maneuvering_sat(p)

    with db.session(p) as conn:
        s1 = persist.run_scan(
            conn, detected_at=1000,
            run_conjunction=False, run_inspector=False, run_decay=False,
        )
    assert s1.new_by_type["maneuver"] >= 1
    assert s1.total_new == s1.new_by_type["maneuver"]

    # Re-running on identical data records nothing new.
    with db.session(p) as conn:
        s2 = persist.run_scan(
            conn, detected_at=2000,
            run_conjunction=False, run_inspector=False, run_decay=False,
        )
    assert s2.new_by_type["maneuver"] == 0

    with db.session(p) as conn:
        recs = recent_anomalies(conn, types=["maneuver"])
    assert len(recs) == s1.new_by_type["maneuver"]
    assert recs[0].primary_id == 44713
