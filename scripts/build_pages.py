"""Build the self-contained GitHub Pages demo (docs/index.html).

Renders a fully offline, no-external-requests interactive world map of the
Starlink constellation, colored by decay risk, and writes it to
``docs/index.html`` for GitHub Pages to serve.

Why not ``spacetrack globe``: Plotly's ``Scattergeo`` fetches its basemap
topojson from a CDN at render time, and the deck.gl globe pulls tiles/imagery
from the network — either goes blank on a locked-down network (exactly the
environment a security team may run). And a WebGL 3D globe can't be rendered
by headless CI for verification. This build instead draws coastlines from the
bundled Natural Earth GeoJSON as plain SVG line traces (``go.Scatter``, not
``Scattergeo``) with the constellation as SVG markers and Plotly.js inlined —
so the page renders from a single file, with no outbound requests, on any
browser or network.

Usage:
    python scripts/build_pages.py            # uses data/spacetrack.db
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go

from spacetrack.anomaly import decay as decay_mod
from spacetrack.propagate.sgp4_engine import propagate_many
from spacetrack.storage import db
from spacetrack.viz import coastlines
from spacetrack.viz.globe3d import RISK_STYLE

DB_PATH = Path("data/spacetrack.db")

OCEAN = "#0a1626"   # plot area — a deep blue-black "ocean"
COAST = "#33506e"   # coastline outlines
GRID = "#14233a"    # graticule
BG = "#06090f"      # page ground
INK = "#dde6f1"


def _load_latest_tles(conn) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        """
        SELECT s.name, t.line1, t.line2
        FROM satellites s
        JOIN tle_snapshots t ON t.norad_id = s.norad_id
        WHERE s.constellation = 'starlink'
          AND t.epoch = (SELECT MAX(epoch) FROM tle_snapshots WHERE norad_id = s.norad_id)
        """
    ).fetchall()
    return [(r["name"], r["line1"], r["line2"]) for r in rows]


def _coastline_trace() -> go.Scatter:
    """Coastlines as a single SVG polyline trace (NaN-separated segments)."""
    xs: list[float] = []
    ys: list[float] = []
    for segment in coastlines.load_segments():          # list of (lat, lon)
        for lat, lon in segment:
            xs.append(lon)
            ys.append(lat)
        xs.append(None)  # break so segments aren't joined across the map
        ys.append(None)
    return go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color=COAST, width=1),
        hoverinfo="skip", showlegend=False, name="coastline",
    )


def _sat_traces(positions, risk_map) -> list[go.Scatter]:
    buckets: dict[str, list] = {tier: [] for tier in RISK_STYLE}
    for p in positions:
        buckets.setdefault(risk_map.get(p.norad_id, "nominal"), []).append(p)

    sizes = {"nominal": 3, "elevated": 6, "high": 9, "imminent": 13}
    traces: list[go.Scatter] = []
    for tier in sorted(RISK_STYLE, key=lambda t: RISK_STYLE[t]["order"]):
        sats = buckets.get(tier, [])
        total = len(sats)
        if tier == "nominal":
            sats = sats[::2]  # thin the backdrop; keep every flagged sat
        if not sats:
            continue
        style = RISK_STYLE[tier]
        traces.append(go.Scatter(
            x=[p.longitude for p in sats],
            y=[p.latitude for p in sats],
            mode="markers",
            marker=dict(size=sizes[tier], color=style["color"],
                        opacity=style["opacity"], line=dict(width=0)),
            text=[
                f"{p.name} (NORAD {p.norad_id})<br>risk: <b>{tier}</b><br>"
                f"alt {p.altitude_km:.1f} km<br>lat {p.latitude:.2f}°, lon {p.longitude:.2f}°"
                for p in sats
            ],
            hoverinfo="text",
            name=f"{tier} ({total:,})", showlegend=True,
        ))
    return traces


def build_figure() -> tuple[go.Figure, dict]:
    with db.session(DB_PATH) as conn:
        tles = _load_latest_tles(conn)
    positions = propagate_many(tles, when=datetime.now(timezone.utc))
    with db.session(DB_PATH) as conn:
        flagged = decay_mod.scan(conn, min_risk="elevated")
    risk_map = {a.norad_id: a.risk for a in flagged}

    fig = go.Figure(data=[_coastline_trace(), *_sat_traces(positions, risk_map)])
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=OCEAN, font=dict(color=INK),
        xaxis=dict(range=[-180, 180], dtick=30, gridcolor=GRID, zeroline=False,
                   showticklabels=False, ticks="", constrain="domain"),
        yaxis=dict(range=[-90, 90], dtick=30, gridcolor=GRID, zeroline=False,
                   showticklabels=False, ticks="",
                   scaleanchor="x", scaleratio=1.0, constrain="domain"),
        legend=dict(bgcolor="rgba(6,9,15,0.72)", bordercolor="#2a3a4f",
                    borderwidth=1, font=dict(color=INK), x=0.5, y=-0.02,
                    xanchor="center", yanchor="top", orientation="h"),
        margin=dict(l=6, r=6, t=6, b=6),
    )
    from collections import Counter
    tiers = Counter(risk_map.values())
    stats = {"n": len(positions), "elevated": tiers.get("elevated", 0),
             "high": tiers.get("high", 0), "imminent": tiers.get("imminent", 0)}
    return fig, stats


HEAD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Starlink Watch — Live Demo</title>
<meta name="description" content="Interactive world map of the Starlink constellation, colored by orbital decay risk.">
</head>
<body>
"""

STYLE = r"""<style>
  :root{
    --ground:#06090f; --panel:#0b1119; --line:#1c2735;
    --ink:#e6edf6; --muted:#8595ab; --faint:#5a6b82;
    --signal:#3fd0c9; --elevated:#ffcc44; --high:#ff8833; --imminent:#ff3344;
    --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,"Liberation Mono",monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);-webkit-font-smoothing:antialiased;}
  .wrap{max-width:1180px;margin:0 auto;padding:26px 22px 44px;display:flex;flex-direction:column;gap:18px;min-height:100vh;}
  header{display:flex;flex-direction:column;gap:12px;}
  .eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.24em;text-transform:uppercase;color:var(--signal);display:flex;align-items:center;gap:10px;}
  .eyebrow::before{content:"";width:7px;height:7px;border-radius:50%;background:var(--signal);box-shadow:0 0 0 0 rgba(63,208,201,.55);animation:pulse 2.6s ease-out infinite;}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,208,201,.5);}70%{box-shadow:0 0 0 9px rgba(63,208,201,0);}100%{box-shadow:0 0 0 0 rgba(63,208,201,0);}}
  @media (prefers-reduced-motion:reduce){.eyebrow::before{animation:none;}}
  h1{margin:0;font-size:clamp(30px,5vw,46px);font-weight:650;letter-spacing:-.02em;text-wrap:balance;line-height:1.02;}
  .lede{margin:0;max-width:62ch;color:var(--muted);font-size:16px;line-height:1.55;}
  .lede b{color:var(--ink);font-weight:600;}
  .strip{display:flex;flex-wrap:wrap;gap:10px;}
  .stat{flex:1 1 150px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px;display:flex;flex-direction:column;gap:3px;}
  .stat .n{font-family:var(--mono);font-size:24px;font-weight:600;font-variant-numeric:tabular-nums;letter-spacing:-.01em;}
  .stat .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--faint);}
  .stat.tier{border-left-width:3px;}
  .stat.el{border-left-color:var(--elevated);} .stat.el .n{color:var(--elevated);}
  .stat.hi{border-left-color:var(--high);} .stat.hi .n{color:var(--high);}
  .stat.im{border-left-color:var(--imminent);} .stat.im .n{color:var(--imminent);}
  .legend{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);}
  .legend span{display:inline-flex;align-items:center;gap:7px;}
  .dot{width:9px;height:9px;border-radius:50%;}
  .dot.el{background:var(--elevated);} .dot.hi{background:var(--high);} .dot.im{background:var(--imminent);} .dot.nom{background:#6a7a90;}
  .globe{background:var(--ground);border:1px solid var(--line);border-radius:12px;overflow:hidden;height:70vh;min-height:460px;position:relative;}
  .globe .plotly-graph-div{width:100%!important;height:100%!important;}
  .hint{position:absolute;right:14px;top:12px;z-index:5;font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);pointer-events:none;}
  .foot{display:flex;flex-wrap:wrap;gap:8px 22px;justify-content:space-between;align-items:baseline;border-top:1px solid var(--line);padding-top:16px;color:var(--faint);font-size:13px;line-height:1.5;}
  .foot a{color:var(--signal);text-decoration:none;}
  .foot a:hover,.foot a:focus{text-decoration:underline;}
  .foot .prov{font-family:var(--mono);font-size:11.5px;letter-spacing:.04em;}
</style>
"""

INNER = r"""<div class="wrap">
  <header>
    <div class="eyebrow">Space Domain Awareness · Live demo</div>
    <h1>Starlink Watch</h1>
    <p class="lede">Every operational Starlink, propagated from the latest orbital snapshot and
      screened for <b>orbital decay risk</b> &mdash; the same question a Space Domain Awareness
      operator asks: out of thousands of satellites, which ones are dropping toward re-entry?
      Each dot is one satellite at its sub-satellite point; hover for its altitude and NORAD ID.</p>
  </header>
  <div class="strip">
    <div class="stat"><span class="n">__N__</span><span class="k">Tracked satellites</span></div>
    <div class="stat tier el"><span class="n">__EL__</span><span class="k">Elevated risk</span></div>
    <div class="stat tier hi"><span class="n">__HI__</span><span class="k">High risk</span></div>
    <div class="stat tier im"><span class="n">__IM__</span><span class="k">Imminent re-entry</span></div>
  </div>
  <div class="legend">
    <span><i class="dot nom"></i>Nominal</span>
    <span><i class="dot el"></i>Elevated &middot; perigee &lt; 450 km</span>
    <span><i class="dot hi"></i>High &middot; &lt; 300 km</span>
    <span><i class="dot im"></i>Imminent &middot; &lt; 200 km</span>
  </div>
  <div class="globe">
    <div class="hint">Hover a satellite for details &middot; drag to zoom</div>
    __FRAG__
  </div>
  <div class="foot">
    <span>One view of an eight-tab dashboard &mdash; conjunction screening, maneuver detection,
      inspector-residual scans, observer sky-plots and ground tracks run alongside this map.
      Run the full interactive tool from the
      <a href="https://github.com/Alexander-Maldonado-Pelayo/Starlink-Tracker-">repository</a>.</span>
    <span class="prov">Data: CelesTrak snapshot &middot; Propagation: SGP4 / Skyfield</span>
  </div>
</div>
"""


def _fill(template: str, stats: dict, frag: str) -> str:
    return (template
            .replace("__N__", f"{stats['n']:,}")
            .replace("__EL__", f"{stats['elevated']:,}")
            .replace("__HI__", f"{stats['high']:,}")
            .replace("__IM__", f"{stats['imminent']:,}")
            .replace("__FRAG__", frag))


def main() -> None:
    fig, stats = build_figure()
    frag = fig.to_html(full_html=False, include_plotlyjs="inline",
                       config={"responsive": True, "displaylogo": False})

    # 1) Standalone page for GitHub Pages (full HTML document).
    page = HEAD + _fill(STYLE + INNER, stats, frag) + "</body>\n</html>\n"
    out = Path("docs/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(page, encoding="utf-8")
    Path("docs/.nojekyll").write_text("")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) — "
          f"{stats['n']:,} sats, flagged e/h/i = "
          f"{stats['elevated']}/{stats['high']}/{stats['imminent']}")

    # 2) Body-only fragment for the Claude artifact (no <head>/<body> wrappers).
    artifact_dir = Path(
        "/tmp/claude-0/-home-user-Starlink-Tracker-/"
        "6fde5f0e-1ef5-5670-94ea-99bdfb32ace0/scratchpad"
    )
    if artifact_dir.is_dir():
        (artifact_dir / "starlink-globe.html").write_text(
            _fill(STYLE + INNER, stats, frag), encoding="utf-8"
        )
        print(f"wrote artifact body to {artifact_dir / 'starlink-globe.html'}")


if __name__ == "__main__":
    main()
