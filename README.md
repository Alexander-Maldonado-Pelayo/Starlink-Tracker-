# Starlink Watch

A Python tool that tracks the Starlink constellation in near-real-time and applies
cybersecurity-style anomaly detection to spot conjunctions, unannounced maneuvers,
suspicious proximity events, and orbital decay.

**Status:** Phases 1 (foundation), 2 (visualization), and 2.5 (live modes — dashboard, terminal stream, scheduled updates) complete. Phase 3 (anomaly detection) is next. See [notes/07-status.md](notes/07-status.md) for the running log.

## Live demo

**[https://starlink-watch.streamlit.app](https://starlink-watch.streamlit.app)**

Public deployment on Streamlit Cloud — no install required. Data is
propagated from the most recent TLE snapshot.

To run locally, launch with `spacetrack dashboard` and open:

**[http://localhost:8501](http://localhost:8501)**

## Quickstart

```powershell
# from the project root, with the venv activated:
spacetrack update                    # pull 10,315 live Starlink TLEs
spacetrack stats                     # see what's stored
spacetrack where STARLINK-1008       # current position of one sat
spacetrack globe --open              # render the whole constellation in 3D
spacetrack track STARLINK-1008 --open   # ground track over the next 2 orbits
spacetrack observe STARLINK-1008 --open # sky view + elevation timeline from Colorado Springs
spacetrack live STARLINK-1008           # continuous terminal stream of position
spacetrack dashboard                    # launch Streamlit web dashboard at localhost:8501
```

## The pitch

Most "satellite tracker" projects stop at "show pretty 3D globe." This one borrows
the playbook of Space Domain Awareness (SDA) the discipline used by orgs like
LeoLabs, Slingshot, and US Space Force and asks the harder question:

> Out of 6,500 Starlink satellites, which ones are doing something they shouldn't be?

By focusing on a single homogeneous constellation, "normal" is well-defined, and
deviations become tractable to detect with public data alone.

## Goals

1. Pull live Starlink TLE data from CelesTrak (no API key, no hardware).
2. Propagate orbits with SGP4 and answer "where is X right now / when does it pass overhead?"
3. Store TLE history over time so we can diff orbital elements and detect maneuvers.
4. Run four anomaly detectors:
   - Conjunctions (close approaches with non-Starlink objects)
   - Maneuver detection (unannounced burns)
   - Proximity / inspector-style events (one sat closing on another)
   - Decay / re-entry tracking
5. CLI first; Streamlit dashboard as a Phase 4 add-on.

## Non-goals (for now)

- Live SDR signal capture (no hardware).
- Spoofing / jamming detection (different project, would need RF data).
- Tracking the full ~10k object public catalog (Starlink-only keeps scope tight).

## Production deploy (Turso + scheduled refresh)

The Streamlit Cloud deploy reads from [Turso](https://turso.tech) (cloud
SQLite), and a GitHub Action repopulates that database every two hours from
CelesTrak. Users never wait on a live Space-Track fetch — the dashboard's
DB is always already current. To set this up on a fork:

1. **Create a Turso DB** — `turso db create starlink-watch`, then capture
   the URL (`turso db show starlink-watch`) and an auth token
   (`turso db tokens create starlink-watch`).
2. **Add four GitHub repo secrets** (Settings ▸ Secrets and variables ▸
   Actions):
   - `TURSO_URL` — `libsql://your-db.turso.io`
   - `TURSO_AUTH_TOKEN` — the token from step 1
   - `SPACETRACK_IDENTITY` — Space-Track.org login (used by the action as a
     fallback if CelesTrak fails)
   - `SPACETRACK_PASSWORD`
3. **Add the same four to Streamlit Cloud Secrets** so the dashboard reads
   from Turso instead of a local file.
4. **Migrate existing local data** (one-shot, optional):
   ```powershell
   $env:TURSO_URL = "libsql://..."
   $env:TURSO_AUTH_TOKEN = "eyJh..."
   python scripts/migrate_to_turso.py
   ```
5. **Trigger the first refresh** — Actions ▸ Refresh TLE catalog ▸ Run
   workflow. Subsequent runs fire automatically every two hours.

Local development keeps using a file-based SQLite DB at `data/spacetrack.db`
— the Turso path activates only when both `TURSO_URL` and
`TURSO_AUTH_TOKEN` are set in the environment.

## Project layout (planned)

See [notes/01-architecture.md](notes/01-architecture.md) for the full breakdown.

## Notes / docs

- [notes/01-architecture.md](notes/01-architecture.md) — module layout, tech stack, data schema
- [notes/02-starlink.md](notes/02-starlink.md) — why Starlink specifically, what makes it interesting
- [notes/03-anomalies.md](notes/03-anomalies.md) — the security/SDA angle, real-world examples
- [notes/04-phases.md](notes/04-phases.md) — phased rollout plan
- [notes/05-references.md](notes/05-references.md) — APIs, libraries, reading list
- [notes/06-open-questions.md](notes/06-open-questions.md) — decisions still to make
- [notes/07-status.md](notes/07-status.md) — running status log of what's shipped
- [notes/08-observer.md](notes/08-observer.md) — observer view feature (sky plot + elevation timeline)
- [notes/09-live.md](notes/09-live.md) — live modes: scheduled updates, terminal stream, web dashboard
