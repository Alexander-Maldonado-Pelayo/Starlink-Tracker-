# Live modes — keeping the project running

Three independent ways to make Starlink Watch feel "alive" rather than a
one-shot script.

| Mode | What it does | Run with |
|---|---|---|
| **Scheduled updates** | Refreshes the TLE catalog every 6h in the background | Windows Task Scheduler |
| **Terminal stream** | Prints a single satellite's position + look angles every few seconds | `spacetrack live <sat>` |
| **Web dashboard** | Auto-refreshing browser page: globe, sky plot, pass schedule | `spacetrack dashboard` |

Each layer is optional and they work fine in any combination.

---

## 1. Scheduled updates (Windows Task Scheduler)

Fresh TLEs from CelesTrak, on a 6-hour cadence, with no manual intervention.
Critical for Phase 3 anomaly detection — maneuver detection compares
successive TLE snapshots, so the more snapshots accumulated, the better the
detector gets.

### Install

```powershell
.\scripts\register_scheduled_update.ps1
```

This registers a Scheduled Task called `SpacetrackUpdate` that:
- Runs `python -m spacetrack.cli update` every 6 hours
- Uses the project's venv (no system-Python pollution)
- Runs under your user account (no admin required)
- Doesn't run while the laptop is on battery in low-power state
- Has a 5-minute timeout so it can't hang the system

### Useful commands

```powershell
# View the task
Get-ScheduledTask -TaskName SpacetrackUpdate

# Run it once on demand (great for testing)
Start-ScheduledTask -TaskName SpacetrackUpdate

# Check last run result
Get-ScheduledTaskInfo -TaskName SpacetrackUpdate

# Remove it
.\scripts\register_scheduled_update.ps1 -Uninstall
```

### Logs

Each run appends to `logs/scheduled_update.log`. To make the task actually
write there, redirect output inside the action — the current script just
runs `spacetrack.cli`, which logs to stderr. For full file logging, edit the
PowerShell script's `-Argument` to wrap the call in a redirect, e.g.
`-c "python -m spacetrack.cli update >> logs\\scheduled_update.log 2>&1"`.

### Caveats

- The task only runs when you're logged on (default). To run when logged off,
  register it via Task Scheduler GUI and check **Run whether user is logged
  on or not** that requires saving your password.
- If your laptop is asleep when a fire time comes around, the task is skipped
  for that interval but fires at the next one.

---

## 2. Terminal stream — `spacetrack live`

A continuously updating one-line status for a single satellite.

```powershell
spacetrack live STARLINK-1008
```

Output (one new line every ~2 seconds):
```
22:55:12 UTC   38.05°N   105.21°W  alt  447.3 km | az  88.4°  el -23.41°  range  3500 km  [below hz]
22:55:14 UTC   38.13°N   105.07°W  alt  447.5 km | az  88.5°  el -23.30°  range  3491 km  [below hz]
22:55:16 UTC   38.21°N   104.93°W  alt  447.7 km | az  88.6°  el -23.19°  range  3482 km  [below hz]
```

The `[below hz]` / `[above hz]` / `[visible ]` tag at the end tells you
whether the satellite is currently below the horizon, above 0° (geometrically
visible), or above 10° (practically visible from your location).

### Options

```
--lat / --lon / --alt          Override the observer location
--observer-name                Custom label for the observer
--interval N                   Seconds between updates (default 2.0)
```

Press **Ctrl+C** to stop.

### What it does NOT do

- It does not refetch TLEs every tick. It uses the most recent stored TLE
  and re-propagates it to "now" each second. To get a fresh TLE, run
  `spacetrack update` (or let the scheduled task do it).

---

## 3. Web dashboard — `spacetrack dashboard`

A Streamlit web app at `http://localhost:8501` with three tabs:

| Tab | Contents |
|---|---|
| **Constellation** | 3D Earth with all (or sampled) Starlink satellites at "now" |
| **Observer · Sky plot** | Polar sky plot + elevation timeline for one satellite from any location |
| **Ground track** | 2D ground track for one satellite over the next N orbits |

The page auto-refreshes every 1 hour via an HTML meta-refresh — long enough
that interactive state (3D rotation, scroll position, dropdown selections)
isn't constantly wiped. For a faster manual refresh, press **R** in the
browser; that reruns the page in a couple of seconds with fresh propagation.

Cache TTL is decoupled at 60 seconds, so a manual rerun always gets recently
re-propagated positions even though the page itself doesn't auto-reload that
often.

### Run

```powershell
spacetrack dashboard
```

Equivalent to:
```powershell
streamlit run dashboard/app.py
```

### Sidebar controls

- **Observer preset / custom** — switch between Colorado Springs default and
  a custom lat/lon/altitude
- **Globe sample size** — fewer sats = faster render (try 500–2000 for a
  smooth experience, 10k+ for full fidelity)
- **Look-ahead window** — how many hours of passes to scan
- **Satellite query** — name or NORAD ID for the observer / track tabs

### Stop

`Ctrl+C` in the terminal that launched it.

### Caveats

- The dashboard reads from `data/spacetrack.db`. If that file doesn't exist
  yet, it tells you to run `spacetrack update` first.
- TLEs don't refresh automatically while the dashboard runs — the
  auto-refresh just re-propagates against whatever's in the DB. Pair with
  the scheduled task above for truly live data.

---

## Putting it all together

A realistic "production-like" setup:

1. `.\scripts\register_scheduled_update.ps1` — TLEs stay fresh forever
2. `spacetrack dashboard` in one terminal — open `http://localhost:8501`
3. `spacetrack live STARLINK-XXXX` in another terminal — focused stream

That's the project running indefinitely. The portfolio demo: leave the
dashboard open with the constellation slowly rotating, the live terminal
streaming a single-sat view, and screenshots showing both.
