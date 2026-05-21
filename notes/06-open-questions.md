# Open questions

Decisions we haven't made yet. Revisit before / during Phase 1 implementation.

## Resolved (2026-05-09)

- **Git from day one?** → Not yet. Hold off until there's code worth committing.
- **Python version?** → 3.11+.
- **Snapshot frequency?** → Every 1 hour.
- **Tests from the start?** → Yes, pytest with synthetic TLE fixtures.

## Environment / tooling

- **Windows-only or also WSL/Linux?** Affects whether we worry about path
  separators, line endings, and which scheduler we use for the snapshot writer
  (Task Scheduler vs. cron).
- **Git from day one?** Recommended the commit history itself becomes part
  of the portfolio story.
- **GitHub repo public or private until Phase 3 is done?** Public + a
  "WIP / portfolio project" README banner is a reasonable default.
- **Python version target?** 3.11+ recommended (better error messages, faster).

## Scope refinements

- **Snapshot frequency for TLE history?** CelesTrak updates several times daily.
  Pulling every 1 hour is plenty for maneuver detection and keeps the DB small.
- **How far back to keep TLE history?** 90 days covers plenty of anomaly windows
  without bloating SQLite. Older data can be archived to CSV.
- **Include non-Starlink "neighbors" for conjunction detection?** Yes pull the
  full active LEO catalog for conjunction context, but only run maneuver/inspector
  detection on Starlink itself. Keeps cost manageable.
- **Geographic scope for pass prediction?** Default to "anywhere the user
  configures"; ship with no built-in default location.

## Engineering choices to revisit later

- **C/C++ propagation engine?** Premature optimization right now. Profile the
  Python SGP4 batch propagation first; only drop into C if it's actually slow.
- **Test coverage strategy?** Synthetic TLEs (known orbits, known answers) are
  more reliable than network-dependent tests. Write a fixture generator.
- **Logging vs. print?** Use `logging` module from day one Phase 4 dashboard
  will want structured logs.

## Portfolio presentation

- **README screenshots — real Starlink events or synthetic?** Real events from
  the last 90 days are way more compelling. Plan to capture screenshots once
  the detectors actually catch something.
- **Live demo?** A Streamlit Cloud or fly.io deployment is cheap and lets
  recruiters click around. Worth doing in Phase 4.
- **Writeup / blog post?** A "what I learned building Starlink Watch" post
  compounds the portfolio value. Worth budgeting time for.

## Things to discuss with future-you

- If the project takes off, do you want to add OneWeb / Iridium / Kuiper next?
  The schema supports it but the validation story (no FCC ground-truth equivalent
  for those) is weaker.
- Real-time alerting via Discord webhook is fun but requires a 24/7 runner.
  Cheap VPS ($5/mo) is the easy path.
