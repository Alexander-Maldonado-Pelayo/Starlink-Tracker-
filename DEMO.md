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

## Notes

- The bundled snapshot is a single point in time, so the **Maneuvers** and
  **Inspector** tabs (which need multi-epoch history) may be sparse. To get a
  richer, current dataset on a machine with internet access, run
  `spacetrack update` (live TLEs from CelesTrak) before launching.
- The demo writes to a local file DB at `data/spacetrack.db`. Delete that file
  to start clean.
- Full CLI reference: [notes/10-cli-reference.md](notes/10-cli-reference.md).
