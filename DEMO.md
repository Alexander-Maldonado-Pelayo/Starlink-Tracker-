# Running the demo locally

A self-contained, no-account, works-offline way to show **Starlink Watch** — the
live constellation globe plus the four cybersecurity-style anomaly detectors
(conjunctions, maneuvers, inspector residuals, decay watch) — running on your
own machine.

The demo seeds itself from a **bundled snapshot of ~10,000 Starlink satellites**
committed to this repo, so it renders instantly and needs no network access, no
API key, and no hardware.

---

## One command

**Windows (PowerShell):**

```powershell
.\scripts\demo.ps1
```

**macOS / Linux:**

```bash
./scripts/demo.sh
```

The script creates a virtualenv, installs dependencies, seeds the local
database, and opens the dashboard. When it finishes, browse to:

**http://localhost:8501**

Press `Ctrl+C` in the terminal to stop it. Re-running the script is safe — it
reuses the existing virtualenv and database.

---

## Manual steps (if you'd rather not use the script)

```bash
# 1. Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# 2. Install the package plus the dashboard extras
pip install -e ".[dashboard]"

# 3. Seed the local DB from the bundled snapshot (offline, ~10k sats)
python scripts/seed_demo.py

# 4. Launch the dashboard
spacetrack dashboard                 # → http://localhost:8501
```

---

## What you'll see

The dashboard opens on the **Constellation** tab — a rotatable 3D globe of the
whole Starlink fleet, colorable by altitude or by decay risk. The other tabs
each drive one detector:

| Tab | What it shows |
| --- | --- |
| **Constellation** | 3D globe of all tracked Starlinks; optional time-slider animation |
| **Anomaly Feed** | Accumulating timeline of everything the detectors have flagged |
| **Decay Watch** | Per-satellite re-entry risk from mean-motion decay |
| **Maneuvers** | Orbit raises/drops inferred from mean-motion jumps between epochs |
| **Conjunctions** | Forecast close approaches vs the external catalog* |
| **Inspector** | Orbital-element outlier scan (station-keeping / plane tweaks) |
| **Observer · Sky plot** | Sky view + pass predictions from a ground site |
| **Ground track** | Sub-satellite track over the next few orbits |

\* The **Conjunctions** tab needs the non-Starlink catalog, which is fetched
live from CelesTrak. On a network with outbound access run
`spacetrack update-catalog` first; on an offline/blocked network the other
seven tabs still work fully from the bundled snapshot.

---

## Give someone a link via GitHub Pages (no install, no account)

The repo ships a **self-contained interactive map** of the whole constellation
at [`docs/index.html`](docs/index.html) — coastlines and all ~10,000 satellites,
colored by decay risk, in a single HTML file with **no external requests**
(works on any network, including one that blocks CDNs). GitHub can host it as a
public web page straight from the repo:

1. In the repo on GitHub, open **Settings ▸ Pages**.
2. Under **Build and deployment ▸ Source**, choose **GitHub Actions**.
3. That's it — the included workflow ([`.github/workflows/pages.yml`](.github/workflows/pages.yml))
   publishes `docs/` on every push. When it finishes (Actions tab), your public
   URL appears at **Settings ▸ Pages**, typically:
   `https://alexander-maldonado-pelayo.github.io/Starlink-Tracker-/`

Send that URL to anyone — it opens in any browser with no login and no install.

> Prefer not to use Actions? Under **Settings ▸ Pages ▸ Source** pick
> **Deploy from a branch**, choose this branch and the **`/docs`** folder.

To regenerate the page from the current data (after a `spacetrack update` or
`spacetrack scan`):

```bash
python scripts/build_pages.py     # rewrites docs/index.html, then commit + push
```

The Pages map is a static snapshot. For a link that's *interactive* end-to-end
(all eight tabs, live re-propagation), deploy the full app to Streamlit Cloud
below.

## Share a link so others can access it

To hand someone a URL they can open in a browser — no install, no laptop of
yours required — deploy to **[Streamlit Community Cloud](https://share.streamlit.io)**
(free). This repo is already set up for it.

1. Make sure this repo is pushed to your GitHub (it is, on your working branch).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. **New app** → pick this repository, the branch, and set the main file to
   `dashboard/app.py`.
4. Open **Advanced settings ▸ Secrets** and add one line so the deploy boots
   instantly from the bundled snapshot instead of depending on outbound
   network access:
   ```toml
   STARLINK_WATCH_SEED_ONLY = "1"
   ```
5. **Deploy.** You'll get a public `https://<your-app>.streamlit.app` URL to
   send along.

Notes:
- With `STARLINK_WATCH_SEED_ONLY = "1"` the app serves the committed
  ~10,000-satellite snapshot — reliable and fast, ideal for a demo link.
- Drop that secret and the deploy will instead pull **live** TLEs from
  CelesTrak on first load (when its network can reach CelesTrak).
- For a link that stays continuously current *and* accumulates a real anomaly
  timeline, follow the **Production deploy (Turso + scheduled refresh)** section
  in the [README](README.md#production-deploy-turso--scheduled-refresh).

## Populate the Maneuvers & Inspector tabs (real history)

The **Maneuvers** and **Inspector** detectors compare orbital elements across
*multiple* TLE epochs, so they need history — a single snapshot leaves them
empty. To light them up with genuine detected events (the credible thing to
show a technical audience), backfill real history from
[Space-Track.org](https://www.space-track.org) (free account):

```bash
# One-time: set your Space-Track credentials
export SPACETRACK_IDENTITY="you@example.com"     # Windows: $env:SPACETRACK_IDENTITY="..."
export SPACETRACK_PASSWORD="your-password"        # Windows: $env:SPACETRACK_PASSWORD="..."

# Pull a week of real Starlink TLE history, then run the detectors
spacetrack backfill 2026-07-08 --days 7
spacetrack scan

# Launch (or refresh) the dashboard — Maneuvers, Inspector, and the
# Anomaly Feed are now populated with real events
spacetrack dashboard
```

Starlink maneuvers constantly for station-keeping, so a week of history
surfaces plenty of real boosts, drops, and element residuals. Running
`spacetrack update` on a schedule (see
[`scripts/register_scheduled_update.ps1`](scripts/register_scheduled_update.ps1))
keeps that history growing over time.

## Notes

- The bundled snapshot is a single point in time, so the **Maneuvers** and
  **Inspector** tabs (which need multi-epoch history) may be sparse. To get a
  richer, current dataset on a machine with internet access, run
  `spacetrack update` (live TLEs from CelesTrak) before launching.
- The demo writes to a local file DB at `data/spacetrack.db`. Delete that file
  to start clean.
- Full CLI reference: [notes/10-cli-reference.md](notes/10-cli-reference.md).
