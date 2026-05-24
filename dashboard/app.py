"""Streamlit dashboard — live view of the Starlink constellation.

Run with:
    streamlit run dashboard/app.py
or via the CLI:
    spacetrack dashboard
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
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
from spacetrack.storage.queries import find_satellite, get_latest_tle
from spacetrack.storage.snapshot import write_snapshots
from spacetrack.tle.fetcher import (
    FetchError,
    NoNewData,
    fetch_starlink,
    load_bundled_seed,
    now_unix,
)
from spacetrack.anomaly import decay as decay_mod
from spacetrack.viz.globe3d import render_globe
from spacetrack.viz.globe_deck import render_globe_deck
from spacetrack.viz.groundtrack import render_ground_track
from spacetrack.viz.skyplot import HORIZON_DEG, PRACTICAL_DEG, render_sky

DB_PATH = Path("data/spacetrack.db")

# How often the page auto-reloads via meta-refresh. Long enough that user
# interaction (3D rotation, scroll position, dropdown state) isn't constantly
# wiped, short enough that an unattended dashboard stays roughly current.
PAGE_REFRESH_SECONDS = 3600  # 1 hour

# How long cached data (TLE rows, DB stats) is reused across reruns. Kept
# short so pressing R for a manual rerun gives you fresh propagation.
DATA_CACHE_SECONDS = 60

# CelesTrak refreshes the Starlink GP feed roughly every 2 hours. Re-pull
# the catalog when the freshest snapshot in the DB is older than this so
# the dashboard reflects live TLEs without needing a manual `spacetrack
# update`. A small buffer past 2h avoids hammering CelesTrak with 403s.
CATALOG_MAX_AGE_SECONDS = 2 * 3600 + 600  # 2h 10m

st.set_page_config(
    page_title="Starlink Watch",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def ensure_fresh_catalog() -> None:
    """Bootstrap the catalog on first run and refresh stale TLEs from CelesTrak.

    Two cases:
      * DB empty (fresh checkout or first deploy) — fetch once and persist,
        falling back to the bundled seed if CelesTrak is unreachable.
      * Most recent snapshot is older than CATALOG_MAX_AGE_SECONDS — pull
        a new snapshot so propagated positions track the live constellation.
        Failures here are soft: keep serving the stale catalog rather than
        breaking the dashboard.
    """
    db.init_db(DB_PATH)
    with db.session(DB_PATH) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM satellites WHERE constellation = 'starlink'"
        ).fetchone()[0]
        last_fetched = conn.execute(
            "SELECT MAX(fetched_at) FROM tle_snapshots"
        ).fetchone()[0]

    is_empty = count == 0
    is_stale = (
        last_fetched is not None
        and (now_unix() - int(last_fetched)) > CATALOG_MAX_AGE_SECONDS
    )

    if not is_empty and not is_stale:
        return

    spinner_msg = (
        "First run: fetching the current Starlink catalog from CelesTrak..."
        if is_empty
        else "Catalog is stale — refreshing live TLEs from CelesTrak..."
    )

    with st.spinner(spinner_msg):
        try:
            tles = fetch_starlink()
        except NoNewData:
            # CelesTrak hasn't published a new catalog yet; current DB is
            # already up to date relative to upstream, so nothing to do.
            return
        except FetchError as exc:
            if not is_empty:
                # Stale refresh failed — keep serving what we have.
                st.warning(
                    f"Couldn't refresh from CelesTrak ({exc}). Showing the "
                    f"last cached snapshot."
                )
                return
            # First run with no data: fall back to the bundled seed snapshot.
            try:
                tles = load_bundled_seed()
                st.warning(
                    "CelesTrak is unreachable from this host — showing a bundled "
                    "snapshot instead. Positions are propagated from the most recent "
                    "TLEs committed to the repo.",
                )
            except Exception as seed_exc:
                st.error(
                    f"Couldn't fetch live data and the bundled seed failed to load.\n\n"
                    f"`{exc}`\n\n`{seed_exc}`"
                )
                st.stop()
        with db.session(DB_PATH) as conn:
            write_snapshots(conn, tles, fetched_at=now_unix(), constellation="starlink")
    # Reset cached stats so the metrics strip reflects the new data.
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

# Bootstrap on first run, and auto-refresh from CelesTrak when the catalog
# has gone stale (older than CelesTrak's ~2h update cadence).
ensure_fresh_catalog()

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

tab_globe, tab_decay, tab_bluemarble, tab_observer, tab_track = st.tabs([
    "Constellation",
    "Decay Watch",
    "Blue Marble (deck.gl)",
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

with tab_bluemarble:
    st.subheader("GPU globe — NASA Blue Marble")
    st.caption(
        "deck.gl GlobeView with a photorealistic Earth backdrop. "
        "GPU-rendered — smooth rotation even at full constellation scale. "
        "Markers color-coded by decay risk; hover for details."
    )
    with st.spinner(f"Propagating {globe_limit:,} satellites..."):
        positions_bm = cached_globe_positions(globe_limit, minute_bucket)
        risk_map_bm = cached_risk_map(minute_bucket)
    deck = render_globe_deck(positions_bm, risk_map=risk_map_bm)
    st.pydeck_chart(deck, height=720)
    st.caption(
        "Earth texture is loaded from NASA's `eoimages.gsfc.nasa.gov`. "
        "If you see a dark globe with only dots, the image is still loading "
        "or the CDN is blocked from your network."
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
    f"Catalog auto-refreshes from CelesTrak when it goes stale "
    f"(>{CATALOG_MAX_AGE_SECONDS // 3600}h old)."
)

# Streamlit's autorefresh helper requires an extra package; the simpler
# approach is meta-refresh via injected HTML.
st.markdown(
    f"<meta http-equiv='refresh' content='{PAGE_REFRESH_SECONDS}'>",
    unsafe_allow_html=True,
)
