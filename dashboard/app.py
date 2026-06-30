"""Streamlit dashboard — live view of the Starlink constellation.

Run with:
    streamlit run dashboard/app.py
or via the CLI:
    spacetrack dashboard
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import spacetrack from the source tree, not from an installed wheel.
# Streamlit Cloud's uv-based `pip install -e .` caches old builds by version
# and serves stale code on subsequent deploys; loading from the repo root
# guarantees we run the code that was just pulled.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from spacetrack.observer.visibility import (
    COLORADO_SPRINGS,
    ObserverLocation,
    find_passes,
    look_angles,
)
from spacetrack.propagate.sgp4_engine import propagate_many, propagate_track
from spacetrack.storage import db
from spacetrack.storage.queries import (
    anomaly_counts_by_type,
    find_satellite,
    get_latest_tle,
    recent_anomalies,
)
from spacetrack.storage.snapshot import write_snapshots
from spacetrack.tle.fetcher import (
    FetchError,
    fetch_starlink,
    load_bundled_seed,
    now_unix,
)
from spacetrack.tle.spacetrack_fetcher import (
    SpaceTrackAuthError,
    SpaceTrackError,
    fetch_starlink_for_range,
)
from spacetrack.anomaly import conjunction as conjunction_mod
from spacetrack.anomaly import decay as decay_mod
from spacetrack.anomaly import inspector as inspector_mod
from spacetrack.anomaly import maneuver as maneuver_mod
from spacetrack.viz.globe3d import render_globe
from spacetrack.viz.groundtrack import render_ground_track
from spacetrack.viz.skyplot import HORIZON_DEG, PRACTICAL_DEG, render_sky

DB_PATH = Path("data/spacetrack.db")


def _db_ready() -> bool:
    """True when there's a database to read from.

    Turso (production) is reachable purely from env vars — there's no local
    file — so a bare ``DB_PATH.exists()`` check would wrongly report "no data"
    on the Cloud deploy. Treat a configured Turso connection as ready.
    """
    return bool(os.environ.get("TURSO_URL")) or DB_PATH.exists()


# How often the page auto-reloads via meta-refresh. Long enough that user
# interaction (3D rotation, scroll position, dropdown state) isn't constantly
# wiped, short enough that an unattended dashboard stays roughly current.
PAGE_REFRESH_SECONDS = 3600  # 1 hour

# How long cached data (TLE rows, DB stats) is reused across reruns. Kept
# short so pressing R for a manual rerun gives you fresh propagation.
DATA_CACHE_SECONDS = 60

st.set_page_config(
    page_title="Starlink Watch",
    layout="wide",
    initial_sidebar_state="expanded",
)

log = logging.getLogger(__name__)

# Forward Streamlit Cloud secrets into env vars so the Space-Track fetcher
# and the Turso DB connector (both of which read from os.environ) work in
# Cloud the same way they do locally. Has no effect when no secrets are
# configured — local dev keeps using its file-based SQLite DB.
try:
    for _key in (
        "SPACETRACK_IDENTITY", "SPACETRACK_PASSWORD",
        "TURSO_URL", "TURSO_AUTH_TOKEN",
    ):
        _val = st.secrets.get(_key) if hasattr(st, "secrets") else None
        if _val and not os.environ.get(_key):
            os.environ[_key] = _val
except Exception:  # noqa: BLE001 — secrets backend not configured locally is fine
    pass

# ---------------------------------------------------------------------------
# Data loaders (cached so propagation isn't re-run every Streamlit interaction)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def load_all_tles() -> list[tuple[str, str, str]]:
    if not DB_PATH.exists():
        return []
    with db.session(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT s.name, t.line1, t.line2
            FROM satellites s
            JOIN tle_snapshots t ON t.norad_id = s.norad_id
            WHERE s.constellation = 'starlink'
              AND t.epoch = (
                  SELECT MAX(epoch) FROM tle_snapshots WHERE norad_id = s.norad_id
              )
            """
        ).fetchall()
    return [(r["name"], r["line1"], r["line2"]) for r in rows]


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def db_stats() -> dict[str, int | str | None]:
    if not DB_PATH.exists():
        return {"sats": 0, "snapshots": 0, "last_update": None}
    with db.session(DB_PATH) as conn:
        sats = conn.execute(
            "SELECT COUNT(*) FROM satellites WHERE constellation = 'starlink'"
        ).fetchone()[0]
        snaps = conn.execute("SELECT COUNT(*) FROM tle_snapshots").fetchone()[0]
        last = conn.execute("SELECT MAX(fetched_at) FROM tle_snapshots").fetchone()[0]
    return {"sats": sats, "snapshots": snaps, "last_update": last}


SPACETRACK_BOOTSTRAP_DAYS = 7


def bootstrap_catalog_if_empty() -> None:
    """Populate the DB on first run, trying the best source available.

    When Turso is configured (production), this short-circuits: the
    scheduled GitHub Action owns catalog refresh, the dashboard just reads.
    An empty Turso DB means the action hasn't run yet — we surface that as
    a clear message instead of trying to backfill live, which is the user-
    facing failure mode this entire deploy was designed to eliminate.

    For local SQLite, the original fallback chain still applies:
    1. **CelesTrak** — single current snapshot. Fast, no auth.
    2. **Space-Track gp_history** — last ``SPACETRACK_BOOTSTRAP_DAYS`` days
       of TLEs. Requires SPACETRACK_IDENTITY + SPACETRACK_PASSWORD.
    3. **Bundled seed** — one snapshot committed to the repo. Last-resort
       fallback so the dashboard still renders if both networks are down.
    """
    db.init_db(DB_PATH)
    with db.session(DB_PATH) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM satellites WHERE constellation = 'starlink'"
        ).fetchone()[0]
    if count > 0:
        return

    if os.environ.get("TURSO_URL"):
        st.error(
            "Turso database is empty. The scheduled `refresh-tles` GitHub "
            "Action populates the catalog every two hours — wait for the "
            "next run, or trigger it manually via Actions ▸ Refresh TLE "
            "catalog ▸ Run workflow."
        )
        st.stop()

    tles = None
    source: str | None = None
    celestrak_err: Exception | None = None

    # 1. CelesTrak (cheap, current only).
    with st.spinner("First run: fetching the current Starlink catalog from CelesTrak..."):
        try:
            tles = fetch_starlink()
            source = "celestrak (current snapshot)"
        except FetchError as exc:
            celestrak_err = exc
            log.info("CelesTrak unreachable, falling through: %s", exc)

    # 2. Space-Track multi-day history (preferred for the maneuver detector).
    if tles is None and os.environ.get("SPACETRACK_IDENTITY"):
        try:
            with st.spinner(
                f"Backfilling {SPACETRACK_BOOTSTRAP_DAYS} days of Starlink history "
                "from Space-Track..."
            ):
                end = datetime.now(timezone.utc)
                start = end - timedelta(days=SPACETRACK_BOOTSTRAP_DAYS)
                tles = fetch_starlink_for_range(start, end)
                source = f"space-track ({SPACETRACK_BOOTSTRAP_DAYS}-day history)"
        except (SpaceTrackAuthError, SpaceTrackError) as exc:
            log.warning("Space-Track bootstrap failed: %s", exc)
            st.warning(f"Space-Track fetch failed: {exc}")

    # 3. Bundled seed (last resort).
    if tles is None:
        try:
            tles = load_bundled_seed()
            source = "bundled seed"
            st.warning(
                "Live data sources unreachable — showing the bundled seed snapshot. "
                "Positions are propagated from the most recent TLEs committed to "
                "the repo. The Maneuvers tab needs multi-epoch history and will "
                "be empty in this mode."
            )
        except Exception as seed_exc:  # noqa: BLE001
            st.error(
                "Couldn't fetch live data and the bundled seed failed to load.\n\n"
                f"CelesTrak: `{celestrak_err}`\n\n"
                f"Seed: `{seed_exc}`"
            )
            st.stop()

    with db.session(DB_PATH) as conn:
        new_count, total = write_snapshots(
            conn, tles, fetched_at=now_unix(), constellation="starlink"
        )
    log.info("Bootstrap source=%s persisted=%d/%d", source, new_count, total)

    db_stats.clear()
    load_all_tles.clear()


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_globe_positions(limit: int, bucket_minute: int):
    """Propagate the constellation, cached by (limit, minute bucket).

    Streamlit hashes the arguments; ``bucket_minute`` is the current UTC
    minute (truncated to int) so positions stay sub-minute fresh but
    flipping tabs within the same minute is instant.
    """
    del bucket_minute  # used only as part of Streamlit's cache key
    tles = load_all_tles()[:limit]
    when = datetime.now(timezone.utc)
    return propagate_many(tles, when=when)


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_risk_map(bucket_minute: int) -> dict[int, str]:
    """Run a full-constellation decay scan, cached per minute bucket."""
    del bucket_minute
    if not DB_PATH.exists():
        return {}
    with db.session(DB_PATH) as conn:
        flagged = decay_mod.scan(conn, min_risk="elevated")
    return {a.norad_id: a.risk for a in flagged}


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_maneuver_events(min_magnitude: str, bucket_minute: int) -> list[dict]:
    """Maneuver scan results as plain dicts (cache-safe, table-ready)."""
    del bucket_minute
    if not DB_PATH.exists():
        return []
    with db.session(DB_PATH) as conn:
        events = maneuver_mod.scan(conn, min_magnitude=min_magnitude)  # type: ignore[arg-type]
    return [
        {
            "norad_id": e.norad_id,
            "name": e.name,
            "epoch_before": e.epoch_before,
            "epoch_after": e.epoch_after,
            "gap_days": e.gap_days,
            "altitude_before_km": e.altitude_before_km,
            "altitude_after_km": e.altitude_after_km,
            "delta_a_km": e.delta_a_km,
            "delta_a_km_per_day": e.delta_a_km_per_day,
            "direction": e.direction,
            "magnitude": e.magnitude,
        }
        for e in events
    ]


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_inspector_findings(
    min_severity: str,
    window_days: float,
    bucket_minute: int,
) -> list[dict]:
    """Inspector residual-scan results as cache-safe dicts."""
    del bucket_minute
    if not DB_PATH.exists():
        return []
    with db.session(DB_PATH) as conn:
        findings = inspector_mod.scan(
            conn,
            window_days=window_days,
            min_severity=min_severity,  # type: ignore[arg-type]
        )
    return [
        {
            "severity": f.severity,
            "norad_id": f.norad_id,
            "name": f.name,
            "element": f.element,
            "z_score": f.z_score,
            "observed": f.observed,
            "expected": f.expected,
            "sigma": f.sigma,
            "snapshots_used": f.snapshots_used,
            "window_days": f.window_days,
            "epoch": f.epoch,
        }
        for f in findings
    ]


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_anomaly_feed(
    types: tuple[str, ...],
    since_days: int,
    bucket_minute: int,
) -> list[dict]:
    """Persisted anomalies (written by `spacetrack scan`) as cache-safe dicts.

    Reads the accumulating ``anomalies`` table — the longitudinal record — not
    a live detector run, so the feed reflects everything flagged over time.
    """
    del bucket_minute
    if not _db_ready():
        return []
    since_unix = int(datetime.now(timezone.utc).timestamp()) - since_days * 86400
    with db.session(DB_PATH) as conn:
        records = recent_anomalies(
            conn,
            types=list(types) if types else None,
            since_unix=since_unix,
            limit=2000,
        )
    out: list[dict] = []
    for r in records:
        d = r.details or {}
        out.append({
            "detected_at": r.detected_at,
            "type": r.type,
            "severity": r.severity,
            "primary_id": r.primary_id,
            "secondary_id": r.secondary_id,
            "name": d.get("name") or d.get("primary_name") or "",
            "summary": _anomaly_summary(r.type, d),
        })
    return out


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_anomaly_counts(since_days: int, bucket_minute: int) -> dict[str, int]:
    del bucket_minute
    if not _db_ready():
        return {}
    since_unix = int(datetime.now(timezone.utc).timestamp()) - since_days * 86400
    with db.session(DB_PATH) as conn:
        return anomaly_counts_by_type(conn, since_unix=since_unix)


def _anomaly_summary(kind: str, d: dict) -> str:
    """One-line, analyst-readable description of a persisted anomaly."""
    if kind == "conjunction":
        return (
            f"vs {d.get('secondary_name', '?')} — miss "
            f"{d.get('miss_distance_km', '?')} km @ {d.get('relative_velocity_km_s', '?')} km/s"
        )
    if kind == "maneuver":
        return (
            f"{d.get('direction', '?')} "
            f"{d.get('altitude_before_km', '?')}→{d.get('altitude_after_km', '?')} km "
            f"(Δa {d.get('delta_a_km', '?')} km)"
        )
    if kind == "inspector":
        return (
            f"{d.get('element', '?')} z={d.get('z_score', '?')} "
            f"(obs {d.get('observed', '?')} vs exp {d.get('expected', '?')})"
        )
    if kind == "decay":
        dtr = d.get("days_to_reentry")
        tail = f", ~{dtr}d to reentry" if dtr is not None else ""
        return f"perigee {d.get('perigee_km', '?')} km{tail}"
    return ""


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_other_catalog_size() -> int:
    if not DB_PATH.exists():
        return 0
    with db.session(DB_PATH) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM satellites WHERE constellation = 'other'"
        ).fetchone()[0]


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_conjunctions_per_sat(
    norad_id: int,
    hours_ahead: float,
    max_distance_km: float,
    cap: int,
    step_seconds: float,
    bucket_minute: int,
) -> list[dict]:
    """Per-satellite conjunction scan against the external catalog."""
    del bucket_minute
    if not DB_PATH.exists():
        return []
    with db.session(DB_PATH) as conn:
        events = conjunction_mod.scan_satellite(
            conn, norad_id,
            forecast_hours=hours_ahead,
            step_seconds=step_seconds,
            max_distance_km=max_distance_km,
            cap=cap,
            secondary_constellation="other",
        )
    return [_conjunction_to_dict(e) for e in events]


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_conjunctions_scan(
    primary_limit: int,
    hours_ahead: float,
    max_distance_km: float,
    cap: int,
    step_seconds: float,
    bucket_minute: int,
) -> list[dict]:
    """Constellation-wide conjunction scan (capped for dashboard latency)."""
    del bucket_minute
    if not DB_PATH.exists():
        return []
    with db.session(DB_PATH) as conn:
        events = conjunction_mod.scan(
            conn,
            forecast_hours=hours_ahead,
            step_seconds=step_seconds,
            max_distance_km=max_distance_km,
            cap=cap,
            primary_limit=primary_limit,
        )
    return [_conjunction_to_dict(e) for e in events]


def _conjunction_to_dict(e: conjunction_mod.ConjunctionEvent) -> dict:
    return {
        "risk": e.risk,
        "primary_norad_id": e.primary_norad_id,
        "primary_name": e.primary_name,
        "secondary_norad_id": e.secondary_norad_id,
        "secondary_name": e.secondary_name,
        "tca_unix": e.tca_unix,
        "miss_distance_km": e.miss_distance_km,
        "relative_velocity_km_s": e.relative_velocity_km_s,
    }


def _jd_to_datetime(jd: float) -> datetime:
    """Convert a Julian Date (TLE epoch) to UTC datetime."""
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=jd - 2440587.5)


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_flagged_assessments(bucket_minute: int) -> list[dict]:
    """Decay scan results as plain dicts (cache-safe + table-ready)."""
    del bucket_minute
    if not DB_PATH.exists():
        return []
    with db.session(DB_PATH) as conn:
        flagged = decay_mod.scan(conn, min_risk="elevated")
    return [
        {
            "risk": a.risk,
            "norad_id": a.norad_id,
            "name": a.name,
            "perigee_km": round(a.perigee_km, 1),
            "apogee_km": round(a.apogee_km, 1),
            "altitude_km": round(a.altitude_km, 1),
            "decay_rate_km_per_day": (
                round(a.decay_rate_km_per_day, 3)
                if a.decay_rate_km_per_day is not None else None
            ),
            "days_to_reentry": (
                round(a.days_to_reentry, 1)
                if a.days_to_reentry is not None else None
            ),
        }
        for a in flagged
    ]


@st.cache_data(ttl=DATA_CACHE_SECONDS, show_spinner=False)
def cached_time_frames(limit: int, frames: int, step_min: float, bucket_minute: int):
    """Propagate the constellation at N future instants for animation."""
    del bucket_minute
    from datetime import timedelta
    tles = load_all_tles()[:limit]
    start = datetime.now(timezone.utc)
    return [
        (
            (start + timedelta(minutes=step_min * i)).strftime("%H:%M"),
            propagate_many(tles, when=start + timedelta(minutes=step_min * i)),
        )
        for i in range(frames)
    ]


def find_named_sat(name_substring: str) -> tuple[int, str, str, str] | None:
    """Resolve a name to (norad_id, name, line1, line2)."""
    if not DB_PATH.exists():
        return None
    with db.session(DB_PATH) as conn:
        norad_id = find_satellite(conn, name_substring)
        if norad_id is None:
            return None
        tle = get_latest_tle(conn, norad_id)
        if tle is None:
            return None
    return tle.norad_id, tle.name, tle.line1, tle.line2


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("Starlink Watch")
st.caption(
    "Live tracking of the Starlink constellation with observer-relative "
    "predictions.  ·  Data: CelesTrak  ·  Propagation: SGP4 via Skyfield"
)

# Bootstrap the catalog if the DB is empty (e.g. first run on a fresh deploy).
bootstrap_catalog_if_empty()

# Top-level stats strip
stats = db_stats()
last_update_ts = stats["last_update"]

if last_update_ts is not None:
    age_seconds = int(datetime.now(timezone.utc).timestamp() - last_update_ts)
    if age_seconds < 60:
        age_label = f"{age_seconds}s ago"
    elif age_seconds < 3600:
        age_label = f"{age_seconds // 60}m ago"
    elif age_seconds < 86400:
        age_label = f"{age_seconds // 3600}h {(age_seconds % 3600) // 60}m ago"
    else:
        age_label = f"{age_seconds // 86400}d ago"
    last_utc_full = datetime.fromtimestamp(last_update_ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    last_local_full = datetime.fromtimestamp(last_update_ts).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
else:
    age_label = "never"
    last_utc_full = "—"
    last_local_full = "—"

refresh_label = (
    f"{PAGE_REFRESH_SECONDS // 3600}h"
    if PAGE_REFRESH_SECONDS >= 3600
    else f"{PAGE_REFRESH_SECONDS}s"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracked sats", f"{stats['sats']:,}")
col2.metric("TLE snapshots", f"{stats['snapshots']:,}")
col3.metric("Last update", age_label)
col4.metric("Auto-refresh", refresh_label)

st.caption(
    f"Last update: **{last_utc_full}**  ·  Local time: {last_local_full}"
)

if stats["sats"] == 0:
    st.error(
        "No Starlink data available. The bootstrap fetch may have failed; "
        "check the logs or run `spacetrack update` from a terminal."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — observer controls
# ---------------------------------------------------------------------------

st.sidebar.header("Observer")
preset = st.sidebar.selectbox(
    "Preset location",
    ["Colorado Springs, CO (default)", "Custom"],
)
if preset.startswith("Colorado Springs"):
    site = COLORADO_SPRINGS
else:
    lat = st.sidebar.number_input("Latitude", value=COLORADO_SPRINGS.latitude, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=COLORADO_SPRINGS.longitude, format="%.4f")
    alt = st.sidebar.number_input("Altitude (m)", value=COLORADO_SPRINGS.altitude_m, format="%.0f")
    name = st.sidebar.text_input("Observer name", value="Custom site")
    site = ObserverLocation(name=name, latitude=lat, longitude=lon, altitude_m=alt)

st.sidebar.caption(
    f"**{site.name}**\n\n"
    f"`{site.latitude:.4f}°, {site.longitude:.4f}°, {site.altitude_m:.0f} m`"
)

st.sidebar.header("Globe")
sats_count = int(stats["sats"] or 0)
globe_limit = st.sidebar.slider(
    "Satellites to render",
    min_value=200, max_value=max(sats_count, 200),
    value=min(500, sats_count), step=100,
    help=(
        "The globe uses Scattergeo (SVG, not WebGL). Lower = smoother "
        "rotation; higher = more of the constellation visible."
    ),
)

st.sidebar.header("Observer pass scan")
hours_window = st.sidebar.slider("Look-ahead window (hours)", 1, 48, 12)
sat_query = st.sidebar.text_input(
    "Satellite (name or NORAD ID)",
    value="STARLINK-1008",
    help="Used for the sky plot and ground track tabs.",
)

# ---------------------------------------------------------------------------
# Tabs — globe / observer / track
# ---------------------------------------------------------------------------

(tab_globe, tab_feed, tab_decay, tab_maneuver, tab_conj, tab_inspect,
 tab_observer, tab_track) = st.tabs([
    "Constellation",
    "Anomaly Feed",
    "Decay Watch",
    "Maneuvers",
    "Conjunctions",
    "Inspector",
    "Observer · Sky plot",
    "Ground track",
])

now = datetime.now(timezone.utc)
minute_bucket = int(now.timestamp() // 60)

with tab_globe:
    st.subheader(f"Starlink positions around {now.strftime('%H:%M UTC')}")

    g1, g2, g3, g4 = st.columns([1.2, 1.0, 1.0, 1.0])
    color_by = g1.radio(
        "Color by", ["Altitude", "Decay risk"], horizontal=True,
        help="'Decay risk' overlays the Phase 3 anomaly detector.",
    )
    pulse_on = g2.checkbox(
        "Pulse imminent", value=True,
        disabled=color_by != "Decay risk",
        help="Throb imminent-risk markers (only with Decay risk coloring).",
    )
    animate_on = g3.checkbox(
        "Time slider", value=False,
        help="Propagate at N future instants. Heavier — render time scales with frames.",
    )
    thin_n = g4.slider(
        "Thin nominal sats", 1, 10, 4,
        help="Keep every Nth nominal-tier sat. Flagged sats always rendered.",
    )

    risk_map: dict[int, str] | None = None
    if color_by == "Decay risk":
        with st.spinner("Scanning constellation for decay risk..."):
            risk_map = cached_risk_map(minute_bucket)

    if animate_on:
        n_frames = st.slider("Animation frames", 3, 18, 9)
        step_min = st.slider("Minutes per frame", 5.0, 60.0, 15.0, step=5.0)
        with st.spinner(f"Propagating {n_frames} frames..."):
            time_frames = cached_time_frames(globe_limit, n_frames, step_min, minute_bucket)
        positions_now = time_frames[0][1]
    else:
        time_frames = None
        with st.spinner(f"Propagating {globe_limit:,} satellites..."):
            positions_now = cached_globe_positions(globe_limit, minute_bucket)

    fig = render_globe(
        positions_now,
        risk_map=risk_map,
        pulse=pulse_on and color_by == "Decay risk" and not animate_on,
        time_frames=time_frames,
        thin_nominal=thin_n,
    )
    st.plotly_chart(fig, width="stretch", height=720)

with tab_feed:
    st.subheader("Anomaly feed")
    st.caption(
        "The accumulating record of everything the four detectors have flagged "
        "over time — written by the scheduled `spacetrack scan`, not a live "
        "one-shot. This is the constellation's behavioural history: conjunctions, "
        "maneuvers, element residuals, and decay events as they were detected."
    )

    f1, f2 = st.columns([1.4, 1.0])
    window_label = f1.radio(
        "Time window", ["7 days", "30 days", "90 days", "All"],
        index=1, horizontal=True,
    )
    since_days = {"7 days": 7, "30 days": 30, "90 days": 90, "All": 36500}[window_label]
    type_choice = f2.multiselect(
        "Detector",
        ["conjunction", "maneuver", "inspector", "decay"],
        default=["conjunction", "maneuver", "inspector", "decay"],
        help="Filter the feed by detector type.",
    )

    counts = cached_anomaly_counts(since_days, minute_bucket)
    if not counts:
        st.info(
            "No anomalies recorded yet. The scheduled `spacetrack scan` "
            "populates this feed after each refresh — run "
            "`spacetrack scan` locally, or wait for the next GitHub Action run."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Conjunctions", counts.get("conjunction", 0))
        c2.metric("Maneuvers", counts.get("maneuver", 0))
        c3.metric("Inspector flags", counts.get("inspector", 0))
        c4.metric("Decay events", counts.get("decay", 0))

        feed = cached_anomaly_feed(tuple(sorted(type_choice)), since_days, minute_bucket)
        if not feed:
            st.info("No anomalies of the selected type(s) in this window.")
        else:
            import pandas as pd
            import plotly.express as px

            df = pd.DataFrame(feed)
            df["detected"] = pd.to_datetime(df["detected_at"], unit="s", utc=True)
            df["day"] = df["detected"].dt.floor("D")

            # Detections per day, stacked by detector type — the "bigger picture".
            timeline = (
                df.groupby(["day", "type"]).size().reset_index(name="count")
            )
            fig_tl = px.bar(
                timeline, x="day", y="count", color="type",
                title="Detections per day", height=320,
            )
            fig_tl.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                                 legend_title_text="")
            st.plotly_chart(fig_tl, width="stretch")

            st.markdown(f"**{len(df):,} events** in the last {window_label.lower()} "
                        "(most recent first)")
            table = df[[
                "detected", "type", "severity", "primary_id",
                "secondary_id", "name", "summary",
            ]].rename(columns={
                "detected": "Detected (UTC)", "type": "Type",
                "severity": "Severity", "primary_id": "Primary",
                "secondary_id": "Secondary", "name": "Name", "summary": "Detail",
            })
            st.dataframe(table, width="stretch", hide_index=True, height=460)

with tab_decay:
    st.subheader("Re-entry risk watch")
    st.caption(
        "Per-satellite decay assessment from the latest TLE mean motion. "
        "Risk tiers: **imminent** (perigee < 200 km) · **high** (< 300 km or "
        "−5 km/day) · **elevated** (< 450 km or −1 km/day). "
        "ETA assumes the current decay rate holds until 120 km."
    )

    with st.spinner("Running decay scan..."):
        risk_map_d = cached_risk_map(minute_bucket)
        flagged_rows = cached_flagged_assessments(minute_bucket)

    if not flagged_rows:
        st.info("No satellites flagged. Either everything's nominal or the DB is empty.")
    else:
        counts = {"imminent": 0, "high": 0, "elevated": 0}
        for r in flagged_rows:
            counts[r["risk"]] = counts.get(r["risk"], 0) + 1
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flagged total", f"{len(flagged_rows):,}")
        c2.metric("Imminent", counts.get("imminent", 0))
        c3.metric("High", counts.get("high", 0))
        c4.metric("Elevated", counts.get("elevated", 0))

        with st.spinner(f"Propagating {globe_limit:,} satellites..."):
            positions_now = cached_globe_positions(globe_limit, minute_bucket)
        fig_d = render_globe(
            positions_now, risk_map=risk_map_d, pulse=True, thin_nominal=4,
        )
        st.plotly_chart(fig_d, width="stretch", height=620)

        st.markdown("**Flagged satellites (sorted highest risk first):**")
        st.dataframe(
            flagged_rows,
            width="stretch",
            hide_index=True,
            column_config={
                "risk": st.column_config.TextColumn("Risk"),
                "norad_id": st.column_config.NumberColumn("NORAD", format="%d"),
                "name": st.column_config.TextColumn("Name"),
                "perigee_km": st.column_config.NumberColumn("Perigee (km)", format="%.1f"),
                "apogee_km": st.column_config.NumberColumn("Apogee (km)", format="%.1f"),
                "altitude_km": st.column_config.NumberColumn("Mean alt (km)", format="%.1f"),
                "decay_rate_km_per_day": st.column_config.NumberColumn(
                    "dh/dt (km/day)", format="%+.2f",
                ),
                "days_to_reentry": st.column_config.NumberColumn(
                    "ETA re-entry (days)", format="%.1f",
                ),
            },
        )

with tab_maneuver:
    st.subheader("Maneuver detection")
    st.caption(
        "Orbit raises and drops inferred from mean-motion jumps between "
        "consecutive TLE epochs. **Boost** = orbit raised; **Drop** = orbit "
        "lowered (often a deorbit phase). Magnitude tiers by |Δa|: "
        "**small** (0.5–5 km) · **medium** (5–15 km) · **large** (>15 km). "
        "Threshold rules exclude drag (max ~0.1 km/day at Starlink altitudes)."
    )

    m1, m2 = st.columns([1.2, 1.0])
    mnv_min_mag = m1.radio(
        "Minimum magnitude", ["small", "medium", "large"],
        index=1, horizontal=True,
        help="'small' is the rawest view; 'large' shows only deorbits and major boosts.",
    )
    since_days = m2.slider(
        "Look back (days)", 1, 30, 14,
        help="Filter to events whose latest epoch falls within this many days.",
    )

    with st.spinner("Running maneuver scan..."):
        all_events = cached_maneuver_events(mnv_min_mag, minute_bucket)

    cutoff_dt = now - timedelta(days=since_days)
    events = [
        e for e in all_events
        if _jd_to_datetime(e["epoch_after"]) >= cutoff_dt
    ]

    if not events:
        st.info(
            f"No maneuvers detected at magnitude ≥ **{mnv_min_mag}** in the last "
            f"{since_days} days. Loosen the magnitude or extend the window."
        )
    else:
        counts_mag = {"small": 0, "medium": 0, "large": 0}
        counts_dir = {"boost": 0, "drop": 0}
        for e in events:
            counts_mag[e["magnitude"]] = counts_mag.get(e["magnitude"], 0) + 1
            counts_dir[e["direction"]] = counts_dir.get(e["direction"], 0) + 1

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Events", f"{len(events):,}")
        c2.metric("Large", counts_mag["large"])
        c3.metric("Medium", counts_mag["medium"])
        c4.metric("Boosts", counts_dir["boost"])
        c5.metric("Drops", counts_dir["drop"])

        # Scatter: when each maneuver happened vs how big it was.
        import plotly.graph_objects as go

        boosts = [e for e in events if e["direction"] == "boost"]
        drops = [e for e in events if e["direction"] == "drop"]

        def _trace(rows: list[dict], color: str, name: str) -> go.Scatter:
            return go.Scatter(
                x=[_jd_to_datetime(r["epoch_after"]) for r in rows],
                y=[r["delta_a_km"] for r in rows],
                mode="markers",
                name=f"{name} ({len(rows):,})",
                marker=dict(
                    size=[min(28, 6 + abs(r["delta_a_km"])) for r in rows],
                    color=color,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                text=[
                    (
                        f"<b>{r['name']}</b> (NORAD {r['norad_id']})<br>"
                        f"{r['magnitude']} {r['direction']}<br>"
                        f"Δa = {r['delta_a_km']:+.2f} km over {r['gap_days']:.2f} d<br>"
                        f"alt {r['altitude_before_km']:.1f} → {r['altitude_after_km']:.1f} km"
                    )
                    for r in rows
                ],
                hoverinfo="text",
            )

        fig_m = go.Figure()
        if drops:
            fig_m.add_trace(_trace(drops, "#ff5a5f", "drop"))
            fig_m.add_trace(_trace(boosts, "#5af0a0", "boost"))
        elif boosts:
            fig_m.add_trace(_trace(boosts, "#5af0a0", "boost"))
        fig_m.add_hline(y=0, line_color="#3a4a60", line_width=1)
        fig_m.update_layout(
            paper_bgcolor="#06090f",
            plot_bgcolor="#06090f",
            font=dict(color="#dde6f1"),
            xaxis=dict(
                title="Epoch (UTC)", gridcolor="#1c2735",
                zerolinecolor="#2a3a4f",
            ),
            yaxis=dict(
                title="Δa (km)  ·  positive = boost",
                gridcolor="#1c2735",
                zerolinecolor="#2a3a4f",
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                bgcolor="rgba(6,9,15,0.7)",
                bordercolor="#2a3a4f", borderwidth=1,
            ),
            height=440,
        )
        st.plotly_chart(fig_m, width="stretch")

        st.markdown("**Detected events (most recent first):**")
        table_rows = [
            {
                "when": _jd_to_datetime(e["epoch_after"]).strftime("%Y-%m-%d %H:%M"),
                "magnitude": e["magnitude"],
                "direction": e["direction"],
                "norad_id": e["norad_id"],
                "name": e["name"],
                "delta_a_km": round(e["delta_a_km"], 2),
                "delta_a_km_per_day": round(e["delta_a_km_per_day"], 3),
                "gap_days": round(e["gap_days"], 2),
                "altitude_before_km": round(e["altitude_before_km"], 1),
                "altitude_after_km": round(e["altitude_after_km"], 1),
            }
            for e in events
        ]
        st.dataframe(
            table_rows,
            width="stretch",
            hide_index=True,
            column_config={
                "when": st.column_config.TextColumn("When (UTC)"),
                "magnitude": st.column_config.TextColumn("Mag"),
                "direction": st.column_config.TextColumn("Dir"),
                "norad_id": st.column_config.NumberColumn("NORAD", format="%d"),
                "name": st.column_config.TextColumn("Name"),
                "delta_a_km": st.column_config.NumberColumn("Δa (km)", format="%+.2f"),
                "delta_a_km_per_day": st.column_config.NumberColumn(
                    "rate (km/day)", format="%+.3f",
                ),
                "gap_days": st.column_config.NumberColumn("Gap (d)", format="%.2f"),
                "altitude_before_km": st.column_config.NumberColumn(
                    "Alt before (km)", format="%.1f",
                ),
                "altitude_after_km": st.column_config.NumberColumn(
                    "Alt after (km)", format="%.1f",
                ),
            },
        )

with tab_conj:
    st.subheader("Conjunction screening")
    st.caption(
        "Forecast close approaches between Starlink and CelesTrak's `active` "
        "(non-Starlink) catalog. Risk tiers by miss distance: "
        "**critical** (< 1 km) · **high** (< 2 km) · **medium** (< 5 km) · "
        "**low** (< 10 km). Pre-screen drops pairs whose altitude shells "
        "don't overlap; surviving pairs are SGP4-propagated and reduced to "
        "their minimum 3-D ECI separation."
    )

    other_count = cached_other_catalog_size()
    if other_count == 0:
        st.warning(
            "No external catalog loaded yet. Run "
            "`spacetrack update-catalog` from a terminal to pull CelesTrak's "
            "`active` group (≈ 5,000 non-Starlink sats). Conjunction "
            "screening needs that catalog as the secondary set."
        )
    else:
        st.caption(f"External catalog: **{other_count:,}** non-Starlink sats.")

        cc1, cc2 = st.columns([1.2, 1.0])
        mode = cc1.radio(
            "Mode", ["Single satellite", "Constellation scan (preview)"],
            horizontal=True,
            help=(
                "Single-sat: scan the satellite in the sidebar against the "
                "external catalog. Preview scan: walk the first N Starlinks; "
                "compute scales linearly with N."
            ),
        )
        max_dist = cc2.slider(
            "Max miss distance (km)", min_value=1.0, max_value=20.0,
            value=10.0, step=0.5,
            help="Drop pairs whose closest approach exceeds this.",
        )

        cc3, cc4, cc5 = st.columns(3)
        hours_ahead = cc3.slider(
            "Forecast window (h)", 1.0, 48.0, 24.0, step=1.0,
            help="How far ahead to propagate.",
        )
        cap = cc4.slider(
            "Candidates per primary", 5, 100, 25, step=5,
            help="Cap on the altitude-pre-filtered candidate pool per primary.",
        )
        step_seconds = cc5.select_slider(
            "Step (s)", options=[15, 30, 60, 120, 300], value=60,
            help="Finer = more accurate TCA; heavier compute.",
        )

        events: list[dict]
        if mode == "Single satellite":
            resolved = find_named_sat(sat_query)
            if resolved is None:
                st.warning(f"No satellite matches `{sat_query}` in the database.")
                events = []
            else:
                norad_id, sat_name, _, _ = resolved
                with st.spinner(
                    f"Screening {sat_name} vs {other_count:,} secondaries..."
                ):
                    events = cached_conjunctions_per_sat(
                        norad_id, hours_ahead, max_dist, cap, float(step_seconds),
                        minute_bucket,
                    )
                st.caption(
                    f"Primary: **{sat_name}** (NORAD {norad_id})  ·  "
                    f"forecast {hours_ahead:.0f}h, step {step_seconds}s"
                )
        else:
            preview_n = st.slider(
                "Primaries to scan", 5, 200, 25, step=5,
                help=(
                    "Full constellation scan is several minutes; the preview "
                    "walks the first N Starlinks by NORAD ID."
                ),
            )
            with st.spinner(
                f"Scanning {preview_n} Starlinks vs {other_count:,} secondaries..."
            ):
                events = cached_conjunctions_scan(
                    preview_n, hours_ahead, max_dist, cap, float(step_seconds),
                    minute_bucket,
                )
            st.caption(
                f"Preview scan: first **{preview_n}** Starlinks  ·  "
                f"forecast {hours_ahead:.0f}h, step {step_seconds}s"
            )

        if not events:
            st.info(
                "No conjunctions surfaced under these settings. Below-threshold "
                "close approaches against the active catalog are statistically "
                "rare per-satellite over a 24h window — try a wider miss "
                "distance, a longer window, or the constellation scan."
            )
        else:
            counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for e in events:
                counts[e["risk"]] = counts.get(e["risk"], 0) + 1
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Conjunctions", f"{len(events):,}")
            c2.metric("Critical", counts["critical"])
            c3.metric("High", counts["high"])
            c4.metric("Medium", counts["medium"])
            c5.metric("Low", counts["low"])

            import plotly.graph_objects as go

            _COLORS: dict[str, str] = {
                "critical": "#ff3344",
                "high":     "#ff8a5c",
                "medium":   "#ffd166",
                "low":      "#5af0a0",
            }

            fig_c = go.Figure()
            for tier in ("critical", "high", "medium", "low"):
                rows = [e for e in events if e["risk"] == tier]
                if not rows:
                    continue
                fig_c.add_trace(go.Scatter(
                    x=[datetime.fromtimestamp(r["tca_unix"], tz=timezone.utc)
                       for r in rows],
                    y=[r["miss_distance_km"] for r in rows],
                    mode="markers",
                    name=f"{tier} ({len(rows):,})",
                    marker=dict(
                        size=[max(8, 18 - 2 * r["miss_distance_km"]) for r in rows],
                        color=_COLORS[tier],
                        opacity=0.85,
                        line=dict(width=0),
                    ),
                    text=[
                        (
                            f"<b>{r['primary_name']}</b> vs "
                            f"{r['secondary_name']}<br>"
                            f"miss = {r['miss_distance_km']:.3f} km<br>"
                            f"rel v = {r['relative_velocity_km_s']:.3f} km/s"
                        )
                        for r in rows
                    ],
                    hoverinfo="text",
                ))
            fig_c.update_layout(
                paper_bgcolor="#06090f",
                plot_bgcolor="#06090f",
                font=dict(color="#dde6f1"),
                xaxis=dict(
                    title="Time of closest approach (UTC)",
                    gridcolor="#1c2735", zerolinecolor="#2a3a4f",
                ),
                yaxis=dict(
                    title="Miss distance (km)  ·  lower = closer",
                    gridcolor="#1c2735", zerolinecolor="#2a3a4f",
                    autorange="reversed",
                ),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(
                    bgcolor="rgba(6,9,15,0.7)",
                    bordercolor="#2a3a4f", borderwidth=1,
                ),
                height=440,
            )
            st.plotly_chart(fig_c, width="stretch")

            st.markdown("**Detected close approaches (most-critical first):**")
            table_rows = [
                {
                    "risk": e["risk"],
                    "primary": e["primary_name"],
                    "secondary": e["secondary_name"],
                    "miss_distance_km": round(e["miss_distance_km"], 3),
                    "relative_velocity_km_s": round(e["relative_velocity_km_s"], 3),
                    "tca_utc": datetime.fromtimestamp(
                        e["tca_unix"], tz=timezone.utc,
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                }
                for e in events
            ]
            st.dataframe(
                table_rows,
                width="stretch",
                hide_index=True,
                column_config={
                    "risk": st.column_config.TextColumn("Risk"),
                    "primary": st.column_config.TextColumn("Primary"),
                    "secondary": st.column_config.TextColumn("Secondary"),
                    "miss_distance_km": st.column_config.NumberColumn(
                        "Miss (km)", format="%.3f",
                    ),
                    "relative_velocity_km_s": st.column_config.NumberColumn(
                        "Rel v (km/s)", format="%.3f",
                    ),
                    "tca_utc": st.column_config.TextColumn("TCA (UTC)"),
                },
            )

with tab_inspect:
    st.subheader("Inspector — orbital-element residual scan")
    st.caption(
        "Per-satellite outlier test: a linear trend is fitted to each of "
        "four shape elements (mean motion, eccentricity, inclination, RAAN) "
        "across the last N days of TLE history *excluding* the latest "
        "snapshot. The latest snapshot is then scored against that baseline. "
        "Catches station-keeping burns and plane tweaks too small for the "
        "Maneuver detector. Severity tiers: **notable** (|z| ≥ 3) · "
        "**significant** (|z| ≥ 5) · **extreme** (|z| ≥ 8)."
    )

    i1, i2 = st.columns([1.2, 1.0])
    insp_min_sev = i1.radio(
        "Minimum severity", ["notable", "significant", "extreme"],
        index=1, horizontal=True,
        help="'notable' is the rawest view; 'extreme' shows only the strongest outliers.",
    )
    insp_window = i2.slider(
        "Baseline window (days)", 3.0, 14.0, 7.0, step=1.0,
        help="Span of TLE history used to fit the trend.",
    )

    with st.spinner("Running inspector scan..."):
        insp_rows = cached_inspector_findings(
            insp_min_sev, insp_window, minute_bucket,
        )

    if not insp_rows:
        st.info(
            f"No satellites flagged at severity ≥ **{insp_min_sev}** in the last "
            f"{insp_window:.0f} days of history. Loosen the severity or extend the window."
        )
    else:
        counts = {"extreme": 0, "significant": 0, "notable": 0}
        elem_counts: dict[str, int] = {}
        for r in insp_rows:
            counts[r["severity"]] = counts.get(r["severity"], 0) + 1
            elem_counts[r["element"]] = elem_counts.get(r["element"], 0) + 1

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flagged", f"{len(insp_rows):,}")
        c2.metric("Extreme", counts["extreme"])
        c3.metric("Significant", counts["significant"])
        c4.metric("Notable", counts["notable"])

        if elem_counts:
            top_elem = max(elem_counts, key=elem_counts.get)
            st.caption(
                f"Dominant flagged element: **{top_elem}** "
                f"({elem_counts[top_elem]} of {len(insp_rows)})."
            )

        import plotly.graph_objects as go

        _COLORS_INSP: dict[str, str] = {
            "extreme":     "#ff3344",
            "significant": "#ff8a5c",
            "notable":     "#ffd166",
        }

        fig_i = go.Figure()
        for tier in ("extreme", "significant", "notable"):
            rows = [r for r in insp_rows if r["severity"] == tier]
            if not rows:
                continue
            fig_i.add_trace(go.Scatter(
                x=[r["element"] for r in rows],
                y=[r["z_score"] for r in rows],
                mode="markers",
                name=f"{tier} ({len(rows):,})",
                marker=dict(
                    size=10,
                    color=_COLORS_INSP[tier],
                    opacity=0.75,
                    line=dict(width=0),
                ),
                text=[
                    (
                        f"<b>{r['name']}</b> (NORAD {r['norad_id']})<br>"
                        f"{r['element']} z = {r['z_score']:+.2f}<br>"
                        f"obs = {r['observed']:.6f}<br>"
                        f"exp = {r['expected']:.6f}<br>"
                        f"σ = {r['sigma']:.4g}"
                    )
                    for r in rows
                ],
                hoverinfo="text",
            ))
        fig_i.add_hline(y=0, line_color="#3a4a60", line_width=1)
        fig_i.update_layout(
            paper_bgcolor="#06090f",
            plot_bgcolor="#06090f",
            font=dict(color="#dde6f1"),
            xaxis=dict(
                title="Worst-offending element",
                gridcolor="#1c2735", zerolinecolor="#2a3a4f",
                categoryorder="array",
                categoryarray=list(inspector_mod.ELEMENTS),
            ),
            yaxis=dict(
                title="Z-score of latest residual vs prior-trend baseline",
                gridcolor="#1c2735", zerolinecolor="#2a3a4f",
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                bgcolor="rgba(6,9,15,0.7)",
                bordercolor="#2a3a4f", borderwidth=1,
            ),
            height=420,
        )
        st.plotly_chart(fig_i, width="stretch")

        st.markdown("**Flagged satellites (most-extreme first):**")
        table_rows = [
            {
                "severity": r["severity"],
                "norad_id": r["norad_id"],
                "name": r["name"],
                "element": r["element"],
                "z_score": round(r["z_score"], 2),
                "observed": round(r["observed"], 6),
                "expected": round(r["expected"], 6),
                "snapshots_used": r["snapshots_used"],
            }
            for r in insp_rows
        ]
        st.dataframe(
            table_rows,
            width="stretch",
            hide_index=True,
            column_config={
                "severity": st.column_config.TextColumn("Severity"),
                "norad_id": st.column_config.NumberColumn("NORAD", format="%d"),
                "name": st.column_config.TextColumn("Name"),
                "element": st.column_config.TextColumn("Element"),
                "z_score": st.column_config.NumberColumn("z-score", format="%+.2f"),
                "observed": st.column_config.NumberColumn("Observed", format="%.6f"),
                "expected": st.column_config.NumberColumn("Expected", format="%.6f"),
                "snapshots_used": st.column_config.NumberColumn("Snaps", format="%d"),
            },
        )

with tab_observer:
    resolved = find_named_sat(sat_query)
    if resolved is None:
        st.warning(f"No satellite matches `{sat_query}` in the database.")
    else:
        norad_id, sat_name, line1, line2 = resolved
        st.subheader(f"{sat_name} (NORAD {norad_id}) viewed from {site.name}")
        with st.spinner("Computing look angles..."):
            angles = look_angles(
                sat_name, line1, line2, site,
                start=now, minutes=hours_window * 60.0, step_seconds=15.0,
            )
        horizon_passes = find_passes(angles, HORIZON_DEG)
        practical_passes = find_passes(angles, PRACTICAL_DEG)

        c1, c2, c3 = st.columns(3)
        c1.metric("Window", f"{hours_window}h")
        c2.metric("Passes above horizon", len(horizon_passes))
        c3.metric("Passes above 10°", len(practical_passes))

        fig = render_sky(angles, site, sat_name=sat_name,
                         horizon_passes=horizon_passes,
                         practical_passes=practical_passes)
        st.plotly_chart(fig, width="stretch", height=600)

        if practical_passes:
            st.markdown("**Upcoming practical passes (above 10°):**")
            rows = [
                {
                    "Rise (UTC)": p.rise.when.strftime("%Y-%m-%d %H:%M:%S"),
                    "Peak (UTC)": p.peak.when.strftime("%H:%M:%S"),
                    "Peak elev (°)": round(p.peak.elevation, 1),
                    "Fall (UTC)": p.fall.when.strftime("%H:%M:%S"),
                    "Duration (s)": int((p.fall.when - p.rise.when).total_seconds()),
                }
                for p in practical_passes
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No practical passes (above 10°) in this window.")

with tab_track:
    resolved = find_named_sat(sat_query)
    if resolved is None:
        st.warning(f"No satellite matches `{sat_query}` in the database.")
    else:
        norad_id, sat_name, line1, line2 = resolved
        orbits = st.slider("Orbits to trace", 1, 6, 2)
        with st.spinner("Propagating ground track..."):
            track = propagate_track(
                sat_name, line1, line2,
                start=now, minutes=orbits * 100.0, step_seconds=15.0,
            )
        st.subheader(f"{sat_name} — ground track for the next {orbits} orbits")
        fig = render_ground_track(track)
        st.plotly_chart(fig, width="stretch", height=600)

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

st.caption(
    f"Page auto-refreshes every {refresh_label}. "
    f"Press **R** for a manual rerun (gets fresh positions in seconds). "
    f"Catalog refreshes only when `spacetrack update` runs."
)

# Streamlit's autorefresh helper requires an extra package; the simpler
# approach is meta-refresh via injected HTML.
st.markdown(
    f"<meta http-equiv='refresh' content='{PAGE_REFRESH_SECONDS}'>",
    unsafe_allow_html=True,
)
