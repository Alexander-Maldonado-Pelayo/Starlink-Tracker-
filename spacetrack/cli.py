"""Command-line entry point for spacetrack."""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from spacetrack import __version__
from spacetrack.anomaly import conjunction as conjunction_mod
from spacetrack.anomaly import decay as decay_mod
from spacetrack.anomaly import inspector as inspector_mod
from spacetrack.anomaly import maneuver as maneuver_mod
from spacetrack.anomaly import persist as persist_mod
from spacetrack.live import live_sample
from spacetrack.observer.visibility import (
    ObserverLocation,
    default_observer,
    find_passes,
    look_angles,
)
from spacetrack.propagate.sgp4_engine import (
    propagate,
    propagate_many,
    propagate_track,
)
from spacetrack.storage import db
from spacetrack.storage.queries import find_satellite, get_latest_tle
from spacetrack.storage.snapshot import write_external_catalog, write_snapshots
from spacetrack.tle.fetcher import (
    DEFAULT_CATALOG_GROUPS,
    NoNewData,
    fetch_catalog_groups,
    fetch_starlink,
    now_unix,
)
from spacetrack.tle.spacetrack_fetcher import (
    SpaceTrackAuthError,
    SpaceTrackError,
    fetch_starlink_for_date,
    fetch_starlink_for_range,
)

DEFAULT_DB = Path("data/spacetrack.db")


def _force_utf8_stdio() -> None:
    """Make Windows consoles render unicode (e.g. ° symbols) correctly."""
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if reconfig is not None:
            try:
                reconfig(encoding="utf-8")
            except (ValueError, OSError):
                pass


def _setup_logging(verbose: bool) -> None:
    _force_utf8_stdio()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__, prog_name="spacetrack")
@click.option("--db-path", type=click.Path(path_type=Path), default=DEFAULT_DB,
              help="SQLite database path.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, db_path: Path, verbose: bool) -> None:
    """Track Starlink and detect orbital anomalies."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    db.init_db(db_path)


@main.command()
@click.pass_context
def update(ctx: click.Context) -> None:
    """Fetch the latest Starlink TLEs from CelesTrak and persist them."""
    db_path: Path = ctx.obj["db_path"]
    fetched_at = now_unix()

    try:
        tles = fetch_starlink()
    except NoNewData as exc:
        click.echo(str(exc))
        return
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    with db.session(db_path) as conn:
        new_count, total = write_snapshots(
            conn, tles, fetched_at=fetched_at, constellation="starlink"
        )

    click.echo(f"Fetched {total} Starlink TLEs ({new_count} new since last update).")
    click.echo(f"Database: {db_path}")


@main.command()
@click.argument("date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--days", default=1, type=int, show_default=True,
              help="Window length in days. Pulls [date, date+days).")
@click.pass_context
def backfill(ctx: click.Context, date: datetime, days: int) -> None:
    """Backfill historical Starlink TLEs from Space-Track for a date range.

    DATE is a UTC calendar day (YYYY-MM-DD). Pulls every Starlink TLE whose
    epoch falls in [date, date + days). Requires SPACETRACK_IDENTITY and
    SPACETRACK_PASSWORD env vars set to a Space-Track.org account.

    Idempotent: snapshots dedupe on (norad_id, epoch), so re-running over
    a window that overlaps existing data only inserts truly new rows.
    """
    if days < 1:
        click.echo("--days must be >= 1.", err=True)
        sys.exit(2)

    db_path: Path = ctx.obj["db_path"]
    start = date.replace(tzinfo=timezone.utc)
    click.echo(
        f"Backfilling Starlink TLEs from Space-Track: "
        f"{start.date()} (+{days}d window)..."
    )

    try:
        tles = fetch_starlink_for_date(start, window_days=days)
    except SpaceTrackAuthError as exc:
        click.echo(f"auth error: {exc}", err=True)
        sys.exit(1)
    except SpaceTrackError as exc:
        click.echo(f"Space-Track error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    if not tles:
        click.echo(
            f"No TLEs returned in [{start.date()}, +{days}d). The constellation "
            "may not have had epochs in that window, or your filter is too narrow."
        )
        return

    fetched_at = now_unix()
    with db.session(db_path) as conn:
        new_count, total = write_snapshots(
            conn, tles, fetched_at=fetched_at, constellation="starlink"
        )
    click.echo(
        f"Backfill complete: {total:,} TLEs received, "
        f"{new_count:,} new (rest were already in the DB)."
    )
    click.echo(f"Database: {db_path}")


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show DB stats: tracked sats, snapshots, last update."""
    db_path: Path = ctx.obj["db_path"]
    with db.session(db_path) as conn:
        sats = conn.execute(
            "SELECT COUNT(*) FROM satellites WHERE constellation = 'starlink'"
        ).fetchone()[0]
        snaps = conn.execute("SELECT COUNT(*) FROM tle_snapshots").fetchone()[0]
        last = conn.execute(
            "SELECT MAX(fetched_at) FROM tle_snapshots"
        ).fetchone()[0]

    click.echo(f"Tracked Starlink satellites: {sats}")
    click.echo(f"TLE snapshots stored:        {snaps}")
    if last:
        from datetime import datetime, timezone
        ts = datetime.fromtimestamp(last, tz=timezone.utc).isoformat()
        click.echo(f"Last update:                 {ts}")
    else:
        click.echo("Last update:                 never")


def _parse_when(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise click.BadParameter(
            f"could not parse {value!r} as ISO-8601 datetime"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@main.command()
@click.argument("identifier")
@click.option("--at", "when_str", default=None,
              help="ISO-8601 UTC time (default: now). Example: 2026-05-10T20:00:00Z")
@click.pass_context
def where(ctx: click.Context, identifier: str, when_str: str | None) -> None:
    """Show a satellite's current (or specified) position.

    IDENTIFIER can be a NORAD ID (e.g. 44713) or a satellite name (e.g. STARLINK-1007).
    """
    db_path: Path = ctx.obj["db_path"]
    when = _parse_when(when_str)

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(
                f"error: no satellite matches {identifier!r}. "
                f"Run `spacetrack update` to refresh the catalog.",
                err=True,
            )
            sys.exit(1)

        tle = get_latest_tle(conn, norad_id)
        if tle is None:
            click.echo(f"error: no TLE stored for NORAD {norad_id}", err=True)
            sys.exit(1)

    pos = propagate(tle.name, tle.line1, tle.line2, when=when)

    lat_dir = "N" if pos.latitude >= 0 else "S"
    lon_dir = "E" if pos.longitude >= 0 else "W"
    speed_kmh = pos.speed_km_s * 3600.0

    click.echo(f"{pos.name} (NORAD {pos.norad_id})")
    click.echo(f"  position at {when.isoformat()}:")
    click.echo(f"    latitude:    {abs(pos.latitude):7.3f}° {lat_dir}")
    click.echo(f"    longitude:   {abs(pos.longitude):7.3f}° {lon_dir}")
    click.echo(f"    altitude:    {pos.altitude_km:7.1f} km")
    click.echo(f"    speed:       {pos.speed_km_s:7.3f} km/s ({speed_kmh:,.0f} km/h)")


@main.command()
@click.option("--output", "-o", type=click.Path(path_type=Path),
              default=Path("globe.html"),
              help="Output HTML file (default: globe.html)")
@click.option("--at", "when_str", default=None,
              help="ISO-8601 UTC time (default: now)")
@click.option("--limit", type=int, default=None,
              help="Only render the first N satellites (debug aid)")
@click.option("--color-by", type=click.Choice(["altitude", "risk"]), default="altitude",
              help="Marker coloring: 'altitude' (default) or 'risk' (decay tiers).")
@click.option("--pulse", is_flag=True,
              help="Pulse imminent markers (requires --color-by risk).")
@click.option("--animate", is_flag=True,
              help="Add a time slider stepping through future positions.")
@click.option("--frames", type=int, default=9,
              help="Number of slider frames when --animate is set (default: 9).")
@click.option("--step-min", type=float, default=15.0,
              help="Minutes between frames when --animate is set (default: 15).")
@click.option("--thin", type=int, default=1,
              help="Keep every Nth nominal-tier sat to speed up rotation "
                   "(flagged sats always kept; default 1 = no thinning).")
@click.option("--open", "open_in_browser", is_flag=True,
              help="Open the rendered HTML in your default browser")
@click.pass_context
def globe(ctx: click.Context, output: Path, when_str: str | None,
          limit: int | None, color_by: str, pulse: bool,
          animate: bool, frames: int, step_min: float, thin: int,
          open_in_browser: bool) -> None:
    """Render all tracked Starlink satellites on a 3D Earth (HTML output)."""
    from datetime import timedelta
    from spacetrack.viz.globe3d import render_globe

    db_path: Path = ctx.obj["db_path"]
    when = _parse_when(when_str)

    with db.session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.norad_id, s.name, t.line1, t.line2
            FROM satellites s
            JOIN tle_snapshots t ON t.norad_id = s.norad_id
            WHERE s.constellation = 'starlink'
              AND t.epoch = (
                  SELECT MAX(epoch) FROM tle_snapshots WHERE norad_id = s.norad_id
              )
            """
        ).fetchall()
        risk_map: dict[int, str] | None = None
        if color_by == "risk":
            scan = decay_mod.scan(conn, min_risk="elevated")
            risk_map = {a.norad_id: a.risk for a in scan}

    tles = [(r["name"], r["line1"], r["line2"]) for r in rows]
    if limit is not None:
        tles = tles[:limit]
    if not tles:
        click.echo("error: no Starlink TLEs in DB. Run `spacetrack update` first.", err=True)
        sys.exit(1)

    click.echo(f"Propagating {len(tles)} satellites to {when.isoformat()}...")
    positions = propagate_many(tles, when=when)

    time_frames = None
    if animate:
        click.echo(f"Building {frames} animation frames at {step_min:.0f}-min intervals...")
        time_frames = []
        for i in range(frames):
            t = when + timedelta(minutes=step_min * i)
            frame_positions = propagate_many(tles, when=t)
            label = t.strftime("%H:%M")
            time_frames.append((label, frame_positions))

    click.echo(f"Rendering to {output}...")
    fig = render_globe(
        positions,
        risk_map=risk_map,
        pulse=pulse,
        time_frames=time_frames,
        thin_nominal=thin,
    )
    fig.write_html(str(output), include_plotlyjs="cdn", auto_open=False)
    click.echo(f"Wrote {output}")
    if open_in_browser:
        import webbrowser
        webbrowser.open(output.resolve().as_uri())


@main.command()
@click.argument("identifier")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output HTML file (default: track-<NAME>.html)")
@click.option("--orbits", type=float, default=2.0,
              help="How many orbits forward to plot (default: 2)")
@click.option("--start", "start_str", default=None,
              help="Start time, ISO-8601 UTC (default: now)")
@click.option("--open", "open_in_browser", is_flag=True,
              help="Open the rendered HTML in your default browser")
@click.pass_context
def track(ctx: click.Context, identifier: str, output: Path | None,
          orbits: float, start_str: str | None, open_in_browser: bool) -> None:
    """Render a 2D ground track for one satellite over its next N orbits."""
    from spacetrack.viz.groundtrack import render_ground_track

    db_path: Path = ctx.obj["db_path"]
    start = _parse_when(start_str)

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(f"error: no satellite matches {identifier!r}", err=True)
            sys.exit(1)
        tle = get_latest_tle(conn, norad_id)
        if tle is None:
            click.echo(f"error: no TLE stored for NORAD {norad_id}", err=True)
            sys.exit(1)

    # Starlink orbital period is ~95 min; use 100 to be safe for the V2 shell.
    minutes = orbits * 100.0
    positions = propagate_track(
        tle.name, tle.line1, tle.line2, start=start, minutes=minutes, step_seconds=15.0
    )

    if output is None:
        safe = tle.name.replace(" ", "_").replace("/", "_")
        output = Path(f"track-{safe}.html")

    click.echo(
        f"Rendering {len(positions)} samples ({minutes:.0f} min) for {tle.name}..."
    )
    fig = render_ground_track(positions)
    fig.write_html(str(output), include_plotlyjs="cdn", auto_open=False)
    click.echo(f"Wrote {output}")
    if open_in_browser:
        import webbrowser
        webbrowser.open(output.resolve().as_uri())


@main.command()
@click.argument("identifier")
@click.option("--lat", type=float, default=None, help="Observer latitude (default: Colorado Springs)")
@click.option("--lon", type=float, default=None, help="Observer longitude (default: Colorado Springs)")
@click.option("--alt", type=float, default=None, help="Observer altitude in meters (default: 1839)")
@click.option("--observer-name", default=None, help="Label for the observer site")
@click.option("--hours", type=float, default=6.0, help="Time window in hours (default: 6)")
@click.option("--step", type=float, default=15.0, help="Sample step in seconds (default: 15)")
@click.option("--start", "start_str", default=None,
              help="Start time, ISO-8601 UTC (default: now)")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output HTML file (default: observe-<NAME>.html)")
@click.option("--open", "open_in_browser", is_flag=True,
              help="Open the rendered HTML in your default browser")
@click.pass_context
def observe(
    ctx: click.Context, identifier: str,
    lat: float | None, lon: float | None, alt: float | None, observer_name: str | None,
    hours: float, step: float, start_str: str | None,
    output: Path | None, open_in_browser: bool,
) -> None:
    """Render sky plot + elevation timeline for a satellite as seen from a location.

    Defaults to Colorado Springs, CO. Override with --lat/--lon/--alt.
    """
    from spacetrack.viz.skyplot import HORIZON_DEG, PRACTICAL_DEG, render_sky

    db_path: Path = ctx.obj["db_path"]
    start = _parse_when(start_str)

    site = default_observer()
    if lat is not None or lon is not None or alt is not None or observer_name is not None:
        site = ObserverLocation(
            name=observer_name or site.name,
            latitude=lat if lat is not None else site.latitude,
            longitude=lon if lon is not None else site.longitude,
            altitude_m=alt if alt is not None else site.altitude_m,
        )

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(f"error: no satellite matches {identifier!r}", err=True)
            sys.exit(1)
        tle = get_latest_tle(conn, norad_id)
        if tle is None:
            click.echo(f"error: no TLE stored for NORAD {norad_id}", err=True)
            sys.exit(1)

    angles = look_angles(
        tle.name, tle.line1, tle.line2, site,
        start=start, minutes=hours * 60.0, step_seconds=step,
    )
    horizon = find_passes(angles, threshold_deg=HORIZON_DEG)
    practical = find_passes(angles, threshold_deg=PRACTICAL_DEG)

    click.echo(f"Observer: {site.name} ({site.latitude:.4f}, {site.longitude:.4f}, {site.altitude_m:.0f} m)")
    click.echo(f"Window:   {start.isoformat()} for {hours:.1f} h, step {step:.0f} s")
    click.echo(f"Passes above horizon (0°):    {len(horizon)}")
    click.echo(f"Passes above practical (10°): {len(practical)}")

    for i, p in enumerate(practical, 1):
        click.echo(
            f"  Pass {i}: rise {p.rise.when.strftime('%H:%M:%S')} "
            f"-> peak {p.peak.elevation:5.1f}° "
            f"({p.peak.when.strftime('%H:%M:%S')}) "
            f"-> set {p.fall.when.strftime('%H:%M:%S')}"
        )

    if output is None:
        safe = tle.name.replace(" ", "_").replace("/", "_")
        output = Path(f"observe-{safe}.html")

    fig = render_sky(angles, site, sat_name=tle.name,
                     horizon_passes=horizon, practical_passes=practical)
    fig.write_html(str(output), include_plotlyjs="cdn", auto_open=False)
    click.echo(f"Wrote {output}")
    if open_in_browser:
        import webbrowser
        webbrowser.open(output.resolve().as_uri())


@main.command()
@click.argument("identifier")
@click.option("--lat", type=float, default=None, help="Observer latitude (default: Colorado Springs)")
@click.option("--lon", type=float, default=None, help="Observer longitude (default: Colorado Springs)")
@click.option("--alt", type=float, default=None, help="Observer altitude in meters (default: 1839)")
@click.option("--observer-name", default=None, help="Label for the observer site")
@click.option("--interval", type=float, default=2.0,
              help="Seconds between updates (default: 2.0)")
@click.pass_context
def live(
    ctx: click.Context, identifier: str,
    lat: float | None, lon: float | None, alt: float | None,
    observer_name: str | None, interval: float,
) -> None:
    """Stream a satellite's current position to the terminal until Ctrl+C.

    Reuses the last stored TLE — propagation runs on every tick, but no new
    network calls. To refresh TLEs, run `spacetrack update` in another window.
    """
    db_path: Path = ctx.obj["db_path"]

    site = default_observer()
    if lat is not None or lon is not None or alt is not None or observer_name is not None:
        site = ObserverLocation(
            name=observer_name or site.name,
            latitude=lat if lat is not None else site.latitude,
            longitude=lon if lon is not None else site.longitude,
            altitude_m=alt if alt is not None else site.altitude_m,
        )

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(f"error: no satellite matches {identifier!r}", err=True)
            sys.exit(1)
        tle = get_latest_tle(conn, norad_id)
        if tle is None:
            click.echo(f"error: no TLE stored for NORAD {norad_id}", err=True)
            sys.exit(1)

    click.echo(f"Tracking {tle.name} (NORAD {tle.norad_id}) from {site.name}")
    click.echo(f"Updating every {interval:.1f}s — press Ctrl+C to stop.\n")

    try:
        while True:
            sample = live_sample(tle.name, tle.line1, tle.line2, site)
            p, look = sample.position, sample.look
            lat_dir = "N" if p.latitude >= 0 else "S"
            lon_dir = "E" if p.longitude >= 0 else "W"
            visible = "visible " if look.elevation >= 10.0 else \
                      "above hz" if look.elevation >= 0.0 else "below hz"
            click.echo(
                f"{p.when.strftime('%H:%M:%S')} UTC  "
                f"{abs(p.latitude):6.2f}°{lat_dir}  "
                f"{abs(p.longitude):7.2f}°{lon_dir}  "
                f"alt {p.altitude_km:6.1f} km  "
                f"| az {look.azimuth:5.1f}°  el {look.elevation:+6.2f}°  "
                f"range {look.range_km:6.0f} km  [{visible}]"
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nStopped.")


@main.command("globe-deck")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              default=Path("globe-deck.html"),
              help="Output HTML file (default: globe-deck.html)")
@click.option("--at", "when_str", default=None,
              help="ISO-8601 UTC time (default: now)")
@click.option("--limit", type=int, default=None,
              help="Only render the first N satellites (debug aid)")
@click.option("--color-by", type=click.Choice(["altitude", "risk"]), default="risk",
              help="Marker coloring: 'risk' (default) or 'altitude' (not yet wired).")
@click.option("--open", "open_in_browser", is_flag=True,
              help="Open the rendered HTML in your default browser")
@click.pass_context
def globe_deck(ctx: click.Context, output: Path, when_str: str | None,
               limit: int | None, color_by: str, open_in_browser: bool) -> None:
    """Render the constellation on a deck.gl GlobeView with NASA Blue Marble.

    GPU-accelerated; smoother than the Plotly globe at full constellation
    scale. Real photographic Earth backdrop via NASA Blue Marble bitmap.
    """
    from spacetrack.viz.globe_deck import render_globe_deck

    db_path: Path = ctx.obj["db_path"]
    when = _parse_when(when_str)

    with db.session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT s.norad_id, s.name, t.line1, t.line2
            FROM satellites s
            JOIN tle_snapshots t ON t.norad_id = s.norad_id
            WHERE s.constellation = 'starlink'
              AND t.epoch = (
                  SELECT MAX(epoch) FROM tle_snapshots WHERE norad_id = s.norad_id
              )
            """
        ).fetchall()
        risk_map: dict[int, str] | None = None
        if color_by == "risk":
            scan = decay_mod.scan(conn, min_risk="elevated")
            risk_map = {a.norad_id: a.risk for a in scan}

    tles = [(r["name"], r["line1"], r["line2"]) for r in rows]
    if limit is not None:
        tles = tles[:limit]
    if not tles:
        click.echo("error: no Starlink TLEs in DB. Run `spacetrack update` first.", err=True)
        sys.exit(1)

    click.echo(f"Propagating {len(tles)} satellites to {when.isoformat()}...")
    positions = propagate_many(tles, when=when)
    click.echo(f"Rendering deck.gl globe to {output}...")
    deck = render_globe_deck(positions, risk_map=risk_map)
    deck.to_html(str(output), notebook_display=False)
    click.echo(f"Wrote {output}")
    if open_in_browser:
        import webbrowser
        webbrowser.open(output.resolve().as_uri())


@main.command()
@click.option("--port", type=int, default=8501, help="Streamlit port (default: 8501)")
@click.option("--host", default="localhost", help="Host to bind (default: localhost)")
def dashboard(port: int, host: str) -> None:
    """Launch the Streamlit web dashboard.

    Equivalent to: streamlit run dashboard/app.py
    """
    import subprocess

    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    if not app_path.exists():
        click.echo(f"error: dashboard not found at {app_path}", err=True)
        sys.exit(1)

    click.echo(f"Starting dashboard at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop.")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", str(app_path),
                "--server.port", str(port),
                "--server.address", host,
                "--browser.gatherUsageStats", "false",
            ],
            check=False,
        )
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")


_RISK_SYMBOL = {
    "nominal": "·",
    "elevated": "!",
    "high": "!!",
    "imminent": "!!!",
}


def _format_assessment(a: decay_mod.DecayAssessment) -> str:
    lines = [
        f"{a.name} (NORAD {a.norad_id})  [{a.risk.upper()}]",
        f"  epoch (JD):     {a.epoch:.5f}",
        f"  mean motion:    {a.mean_motion:.6f} rev/day",
        f"  eccentricity:   {a.eccentricity:.6f}",
        f"  semi-major:     {a.semi_major_axis_km:7.1f} km",
        f"  perigee × apogee: {a.perigee_km:7.1f} × {a.apogee_km:7.1f} km",
        f"  mean altitude:  {a.altitude_km:7.1f} km",
    ]
    if a.decay_rate_km_per_day is not None:
        lines.append(f"  decay rate:     {a.decay_rate_km_per_day:+.3f} km/day")
    else:
        lines.append("  decay rate:     (needs ≥2 snapshots)")
    if a.days_to_reentry is not None:
        lines.append(f"  ETA re-entry:   ~{a.days_to_reentry:.1f} days (rough)")
    return "\n".join(lines)


@main.command()
@click.argument("identifier", required=False)
@click.option("--scan", is_flag=True,
              help="Scan the whole tracked constellation and list flagged sats.")
@click.option("--min-risk",
              type=click.Choice(["nominal", "elevated", "high", "imminent"]),
              default="elevated",
              help="Minimum risk to report under --scan (default: elevated).")
@click.option("--limit", type=int, default=None,
              help="Cap on --scan results (default: unlimited).")
@click.pass_context
def decay(
    ctx: click.Context,
    identifier: str | None,
    scan: bool,
    min_risk: str,
    limit: int | None,
) -> None:
    """Assess re-entry risk from TLE mean motion.

    Without --scan, IDENTIFIER must be a NORAD ID or satellite name and the
    command prints a single assessment. With --scan, IDENTIFIER is ignored
    and every tracked Starlink sat is evaluated.
    """
    db_path: Path = ctx.obj["db_path"]

    if scan:
        with db.session(db_path) as conn:
            flagged = decay_mod.scan(conn, min_risk=min_risk)  # type: ignore[arg-type]
        if not flagged:
            click.echo(f"No satellites at risk >= {min_risk}.")
            return
        shown = flagged[:limit] if limit else flagged
        click.echo(
            f"{len(flagged)} satellite(s) at risk >= {min_risk}"
            + (f" (showing first {len(shown)})" if limit and len(shown) < len(flagged) else "")
            + ":"
        )
        click.echo(
            f"  {'RISK':<9} {'NORAD':>6}  {'NAME':<20} "
            f"{'PERI(km)':>9} {'APO(km)':>9} {'dh/dt':>9}"
        )
        for a in shown:
            rate = (
                f"{a.decay_rate_km_per_day:+7.2f}"
                if a.decay_rate_km_per_day is not None
                else "    n/a"
            )
            click.echo(
                f"  {a.risk:<9} {a.norad_id:>6}  {a.name:<20} "
                f"{a.perigee_km:>9.1f} {a.apogee_km:>9.1f} {rate:>9}"
            )
        return

    if not identifier:
        click.echo("error: provide IDENTIFIER or use --scan", err=True)
        sys.exit(1)

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(f"error: no satellite matches {identifier!r}.", err=True)
            sys.exit(1)
        assessment = decay_mod.assess_satellite(conn, norad_id)

    if assessment is None:
        click.echo(f"error: no TLE stored for NORAD {norad_id}", err=True)
        sys.exit(1)
    click.echo(_format_assessment(assessment))


@main.command()
@click.argument("identifier", required=False)
@click.option("--scan", is_flag=True,
              help="Scan every tracked satellite instead of one IDENTIFIER.")
@click.option("--min-magnitude",
              type=click.Choice(["small", "medium", "large"]),
              default="small", show_default=True,
              help="Minimum maneuver size to report under --scan.")
@click.option("--min-delta-km", type=float,
              default=maneuver_mod.DEFAULT_MIN_DELTA_KM, show_default=True,
              help="Minimum |Δa| (km) to consider a maneuver vs. noise.")
@click.option("--max-gap-days", type=float,
              default=maneuver_mod.DEFAULT_MAX_GAP_DAYS, show_default=True,
              help="Skip epoch pairs separated by more than this many days.")
@click.option("--limit", type=int, default=None,
              help="Cap on --scan results (default: unlimited).")
@click.pass_context
def maneuver(
    ctx: click.Context,
    identifier: str | None,
    scan: bool,
    min_magnitude: str,
    min_delta_km: float,
    max_gap_days: float,
    limit: int | None,
) -> None:
    """Detect orbit-raise / orbit-drop maneuvers from TLE mean-motion jumps.

    Without --scan, IDENTIFIER must be a NORAD ID or satellite name and the
    command prints every detected event in that satellite's history. With
    --scan, IDENTIFIER is ignored and every tracked Starlink is checked.

    A "maneuver" is a consecutive-epoch pair whose change in semi-major axis
    exceeds drag's typical ceiling — both in absolute size and per-day rate.
    """
    db_path: Path = ctx.obj["db_path"]

    if scan:
        with db.session(db_path) as conn:
            events = maneuver_mod.scan(
                conn,
                min_magnitude=min_magnitude,  # type: ignore[arg-type]
                min_delta_km=min_delta_km,
                max_gap_days=max_gap_days,
            )
        if not events:
            click.echo(f"No maneuvers detected at magnitude >= {min_magnitude}.")
            return
        shown = events[:limit] if limit else events
        click.echo(
            f"{len(events)} maneuver event(s) at magnitude >= {min_magnitude}"
            + (f" (showing first {len(shown)})" if limit and len(shown) < len(events) else "")
            + ":"
        )
        click.echo(
            f"  {'MAG':<7} {'DIR':<5} {'NORAD':>6}  {'NAME':<20} "
            f"{'Δa(km)':>8} {'rate(km/d)':>11} {'gap(d)':>7}"
        )
        for ev in shown:
            click.echo(
                f"  {ev.magnitude:<7} {ev.direction:<5} {ev.norad_id:>6}  "
                f"{ev.name:<20} {ev.delta_a_km:>+8.2f} "
                f"{ev.delta_a_km_per_day:>+11.3f} {ev.gap_days:>7.2f}"
            )
        return

    if not identifier:
        click.echo("error: provide IDENTIFIER or use --scan", err=True)
        sys.exit(1)

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(f"error: no satellite matches {identifier!r}.", err=True)
            sys.exit(1)
        events = maneuver_mod.scan_satellite(
            conn, norad_id,
            min_delta_km=min_delta_km, max_gap_days=max_gap_days,
        )

    if not events:
        click.echo(f"No maneuvers detected for NORAD {norad_id}.")
        return

    click.echo(f"{len(events)} maneuver event(s) for NORAD {norad_id} ({events[0].name}):")
    for ev in events:
        click.echo(
            f"  [{ev.magnitude}/{ev.direction}] "
            f"Δa = {ev.delta_a_km:+.2f} km over {ev.gap_days:.2f} d "
            f"(rate {ev.delta_a_km_per_day:+.3f} km/d) "
            f"alt {ev.altitude_before_km:.1f} → {ev.altitude_after_km:.1f} km"
        )


@main.command("update-catalog")
@click.option("--debris/--no-debris", default=True, show_default=True,
              help="Include the major LEO debris clouds (Cosmos-1408, "
                   "Fengyun-1C, Iridium-33, Cosmos-2251) alongside the active "
                   "catalog — the objects that dominate real conjunction risk.")
@click.pass_context
def update_catalog(ctx: click.Context, debris: bool) -> None:
    """Fetch the non-Starlink catalog (CelesTrak) for conjunction screening.

    By default pulls operational satellites (`active`) plus the major tracked
    debris clouds, merged and deduped. Stored under constellation='other';
    skips NORAD IDs already tagged 'starlink' so the upsert doesn't clobber the
    constellation label. Use --no-debris for the operational-only catalog.
    """
    db_path: Path = ctx.obj["db_path"]
    fetched_at = now_unix()

    groups = DEFAULT_CATALOG_GROUPS if debris else ("active",)
    try:
        tles = fetch_catalog_groups(groups)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    if not tles:
        click.echo("No catalog data received (all groups unchanged or failed).")
        return

    with db.session(db_path) as conn:
        new_count, total, skipped = write_external_catalog(
            conn, tles, fetched_at=fetched_at, constellation="other",
        )

    scope = "active + debris clouds" if debris else "active only"
    click.echo(
        f"Catalog refresh ({scope}): {total:,} TLEs received, "
        f"{new_count:,} new ({skipped:,} skipped as already-tracked Starlink)."
    )


@main.command()
@click.argument("identifier", required=False)
@click.option("--scan", is_flag=True,
              help="Scan every tracked Starlink instead of one IDENTIFIER.")
@click.option("--min-risk",
              type=click.Choice(["low", "medium", "high", "critical"]),
              default="low", show_default=True,
              help="Minimum miss-distance tier to report.")
@click.option("--hours-ahead", type=float,
              default=conjunction_mod.DEFAULT_FORECAST_HOURS, show_default=True,
              help="Forecast window in hours.")
@click.option("--step-seconds", type=float,
              default=conjunction_mod.DEFAULT_STEP_SECONDS, show_default=True,
              help="Propagation step inside the window.")
@click.option("--max-distance-km", type=float,
              default=conjunction_mod.DEFAULT_MAX_DISTANCE_KM, show_default=True,
              help="Drop conjunctions whose closest approach exceeds this.")
@click.option("--cap", type=int,
              default=conjunction_mod.DEFAULT_MAX_CANDIDATES_PER_PRIMARY, show_default=True,
              help="Max candidate secondaries per primary after altitude pre-filter.")
@click.option("--primary-limit", type=int, default=None,
              help="With --scan, cap on primary satellites (for quick previews).")
@click.option("--limit", type=int, default=None,
              help="Cap on rows printed (default: unlimited).")
@click.pass_context
def conjunction(
    ctx: click.Context,
    identifier: str | None,
    scan: bool,
    min_risk: str,
    hours_ahead: float,
    step_seconds: float,
    max_distance_km: float,
    cap: int,
    primary_limit: int | None,
    limit: int | None,
) -> None:
    """Screen for close approaches between Starlink and the external catalog.

    Requires the external catalog to be populated via `spacetrack update-catalog`.
    Without --scan, IDENTIFIER must be a NORAD ID or satellite name and the
    command finds every close approach for that one satellite. With --scan,
    every tracked Starlink is screened against the 'other' constellation.
    """
    db_path: Path = ctx.obj["db_path"]

    if scan:
        with db.session(db_path) as conn:
            other_count = conn.execute(
                "SELECT COUNT(*) FROM satellites WHERE constellation = 'other'"
            ).fetchone()[0]
            if other_count == 0:
                click.echo(
                    "error: no external catalog loaded. "
                    "Run `spacetrack update-catalog` first.",
                    err=True,
                )
                sys.exit(1)
            events = conjunction_mod.scan(
                conn,
                forecast_hours=hours_ahead,
                step_seconds=step_seconds,
                max_distance_km=max_distance_km,
                cap=cap,
                min_risk=min_risk,  # type: ignore[arg-type]
                primary_limit=primary_limit,
            )
        if not events:
            click.echo(
                f"No conjunctions at risk >= {min_risk} within {hours_ahead:.0f}h."
            )
            return
        shown = events[:limit] if limit else events
        click.echo(
            f"{len(events)} conjunction(s) at risk >= {min_risk}"
            + (f" (showing first {len(shown)})" if limit and len(shown) < len(events) else "")
            + ":"
        )
        click.echo(
            f"  {'RISK':<9} {'PRIMARY':<22} {'SECONDARY':<22} "
            f"{'MISS(km)':>9} {'REL_V(km/s)':>11} {'TCA(UTC)':<19}"
        )
        for ev in shown:
            tca_iso = datetime.fromtimestamp(ev.tca_unix, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            click.echo(
                f"  {ev.risk:<9} {ev.primary_name[:22]:<22} "
                f"{ev.secondary_name[:22]:<22} "
                f"{ev.miss_distance_km:>9.3f} "
                f"{ev.relative_velocity_km_s:>11.3f} {tca_iso:<19}"
            )
        return

    if not identifier:
        click.echo("error: provide IDENTIFIER or use --scan", err=True)
        sys.exit(1)

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(f"error: no satellite matches {identifier!r}.", err=True)
            sys.exit(1)
        events = conjunction_mod.scan_satellite(
            conn, norad_id,
            forecast_hours=hours_ahead,
            step_seconds=step_seconds,
            max_distance_km=max_distance_km,
            cap=cap,
        )

    if not events:
        click.echo(
            f"No conjunctions within {hours_ahead:.0f}h for NORAD {norad_id} "
            f"(threshold {max_distance_km:.1f} km)."
        )
        return

    click.echo(
        f"{len(events)} conjunction(s) within {hours_ahead:.0f}h for "
        f"NORAD {norad_id} ({events[0].primary_name}):"
    )
    for ev in events:
        tca_iso = datetime.fromtimestamp(ev.tca_unix, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        click.echo(
            f"  [{ev.risk}] vs {ev.secondary_name} (NORAD {ev.secondary_norad_id}) "
            f"miss {ev.miss_distance_km:.3f} km @ {tca_iso} "
            f"(rel v {ev.relative_velocity_km_s:.3f} km/s)"
        )


@main.command()
@click.argument("identifier", required=False)
@click.option("--scan", is_flag=True,
              help="Scan every tracked Starlink instead of one IDENTIFIER.")
@click.option("--min-severity",
              type=click.Choice(["notable", "significant", "extreme"]),
              default="notable", show_default=True,
              help="Minimum severity tier to report under --scan.")
@click.option("--window-days", type=float,
              default=inspector_mod.DEFAULT_WINDOW_DAYS, show_default=True,
              help="History window length (days) used to fit the baseline trend.")
@click.option("--min-snapshots", type=int,
              default=inspector_mod.DEFAULT_MIN_SNAPSHOTS, show_default=True,
              help="Skip sats with fewer snapshots in the window.")
@click.option("--threshold", "notable_threshold", type=float,
              default=inspector_mod.DEFAULT_NOTABLE_THRESHOLD, show_default=True,
              help="Z-score threshold for the 'notable' tier.")
@click.option("--limit", type=int, default=None,
              help="Cap on rows printed (default: unlimited).")
@click.pass_context
def inspector(
    ctx: click.Context,
    identifier: str | None,
    scan: bool,
    min_severity: str,
    window_days: float,
    min_snapshots: int,
    notable_threshold: float,
    limit: int | None,
) -> None:
    """Flag satellites whose latest TLE elements depart from a 7-day baseline.

    Fits a linear trend per element (mean motion, eccentricity, inclination,
    RAAN) over the last N days of TLE history, excluding the latest snapshot.
    Reports the satellite with a z-score for the worst-offending element.

    Without --scan, IDENTIFIER selects one satellite. With --scan, every
    tracked Starlink with ≥ min-snapshots history is evaluated.
    """
    db_path: Path = ctx.obj["db_path"]

    if scan:
        with db.session(db_path) as conn:
            findings = inspector_mod.scan(
                conn,
                window_days=window_days,
                min_snapshots=min_snapshots,
                notable_threshold=notable_threshold,
                min_severity=min_severity,  # type: ignore[arg-type]
            )
        if not findings:
            click.echo(f"No satellites flagged at severity >= {min_severity}.")
            return
        shown = findings[:limit] if limit else findings
        click.echo(
            f"{len(findings)} satellite(s) flagged at severity >= {min_severity}"
            + (f" (showing first {len(shown)})" if limit and len(shown) < len(findings) else "")
            + ":"
        )
        click.echo(
            f"  {'SEVERITY':<12} {'NORAD':>6}  {'NAME':<20} "
            f"{'ELEMENT':<13} {'Z':>7} {'OBSERVED':>12} {'EXPECTED':>12}"
        )
        for f in shown:
            click.echo(
                f"  {f.severity:<12} {f.norad_id:>6}  {f.name:<20} "
                f"{f.element:<13} {f.z_score:>+7.2f} "
                f"{f.observed:>12.6f} {f.expected:>12.6f}"
            )
        return

    if not identifier:
        click.echo("error: provide IDENTIFIER or use --scan", err=True)
        sys.exit(1)

    with db.session(db_path) as conn:
        norad_id = find_satellite(conn, identifier)
        if norad_id is None:
            click.echo(f"error: no satellite matches {identifier!r}.", err=True)
            sys.exit(1)
        finding = inspector_mod.assess_satellite(
            conn, norad_id,
            window_days=window_days,
            min_snapshots=min_snapshots,
            notable_threshold=notable_threshold,
        )

    if finding is None:
        click.echo(
            f"NORAD {norad_id}: insufficient history "
            f"(need ≥ {min_snapshots} snapshots in {window_days:.0f}d window)."
        )
        return

    click.echo(f"{finding.name} (NORAD {finding.norad_id})  [{finding.severity.upper()}]")
    click.echo(f"  window:         {finding.window_days:.2f}d, {finding.snapshots_used} snapshots")
    click.echo(f"  worst element:  {finding.element}")
    click.echo(f"  z-score:        {finding.z_score:+.3f}")
    click.echo(f"  observed:       {finding.observed:.6f}")
    click.echo(f"  expected:       {finding.expected:.6f}")
    click.echo(f"  sigma used:     {finding.sigma:.4g}")


@main.command()
@click.option("--conjunction/--no-conjunction", "do_conjunction", default=True,
              show_default=True, help="Run the conjunction detector (heaviest).")
@click.option("--maneuver/--no-maneuver", "do_maneuver", default=True,
              show_default=True, help="Run the maneuver detector.")
@click.option("--inspector/--no-inspector", "do_inspector", default=True,
              show_default=True, help="Run the inspector (residual) detector.")
@click.option("--decay/--no-decay", "do_decay", default=True,
              show_default=True, help="Run the decay detector.")
@click.option("--conjunction-min-risk",
              type=click.Choice(["low", "medium", "high", "critical"]),
              default="medium", show_default=True,
              help="Minimum conjunction risk tier to record.")
@click.option("--conjunction-primary-limit", type=int, default=None,
              help="Cap primaries screened for conjunctions (faster preview "
                   "scans; omit to screen the whole constellation).")
@click.option("--conjunction-primary-offset", type=int, default=0,
              show_default=True,
              help="Start the conjunction primary window at this offset "
                   "(wraps around). Advance each scheduled run for rolling "
                   "full-constellation coverage.")
@click.option("--maneuver-min-magnitude",
              type=click.Choice(["small", "medium", "large"]),
              default="small", show_default=True,
              help="Minimum maneuver magnitude to record.")
@click.option("--inspector-min-severity",
              type=click.Choice(["notable", "significant", "extreme"]),
              default="notable", show_default=True,
              help="Minimum inspector severity to record.")
@click.option("--decay-min-risk",
              type=click.Choice(["elevated", "high", "imminent"]),
              default="elevated", show_default=True,
              help="Minimum decay risk tier to record.")
@click.pass_context
def scan(
    ctx: click.Context,
    do_conjunction: bool,
    do_maneuver: bool,
    do_inspector: bool,
    do_decay: bool,
    conjunction_min_risk: str,
    conjunction_primary_limit: int | None,
    conjunction_primary_offset: int,
    maneuver_min_magnitude: str,
    inspector_min_severity: str,
    decay_min_risk: str,
) -> None:
    """Run every detector and persist findings to the anomalies table.

    This is the command that turns point-in-time detection into an
    accumulating record. Designed to run on a schedule (e.g. the refresh
    GitHub Action) after each TLE update: findings carry a stable fingerprint
    so re-running on the same data is idempotent — the timeline grows only
    when genuinely new events appear.
    """
    db_path: Path = ctx.obj["db_path"]
    detected_at = now_unix()

    # Ensure the fingerprint column/index exists on pre-existing databases.
    db.init_db(db_path)

    with db.session(db_path) as conn:
        summary = persist_mod.run_scan(
            conn,
            detected_at=detected_at,
            run_conjunction=do_conjunction,
            run_maneuver=do_maneuver,
            run_inspector=do_inspector,
            run_decay=do_decay,
            conjunction_min_risk=conjunction_min_risk,
            conjunction_primary_limit=conjunction_primary_limit,
            conjunction_primary_offset=conjunction_primary_offset,
            maneuver_min_magnitude=maneuver_min_magnitude,
            inspector_min_severity=inspector_min_severity,
            decay_min_risk=decay_min_risk,
        )

    stamp = datetime.fromtimestamp(detected_at, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    click.echo(f"Scan @ {stamp}")
    click.echo(f"  {'DETECTOR':<13} {'FOUND':>7} {'NEW':>7}")
    for det in ("conjunction", "maneuver", "inspector", "decay"):
        if det not in summary.found_by_type:
            continue
        click.echo(
            f"  {det:<13} {summary.found_by_type[det]:>7} "
            f"{summary.new_by_type.get(det, 0):>7}"
        )
    click.echo(f"  {'-'*29}")
    click.echo(f"  {'total new':<13} {'':>7} {summary.total_new:>7}")


if __name__ == "__main__":
    main()
