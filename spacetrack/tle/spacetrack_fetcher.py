"""Pull historical TLEs from Space-Track.org's ``gp_history`` endpoint.

Complements ``fetcher.py`` (CelesTrak, current-catalog only) by giving us a
way to backfill missed days and read TLE history for any past date.

Auth: requires a free Space-Track.org account. Credentials are read from
``SPACETRACK_IDENTITY`` and ``SPACETRACK_PASSWORD`` environment variables.

Rate limits: Space-Track caps anonymous-ish use at ~30 requests/minute and
~300/hour. We default to a conservative 2-second gap between calls; a
single date-range query usually returns the entire constellation in one
response, so backfilling a missed day is a single API call.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests

from spacetrack.tle.parser import ParsedTLE, parse

BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
QUERY_BASE = f"{BASE_URL}/basicspacedata/query"

# Conservative pacing: 30 req/min limit -> 2.0s between calls leaves headroom.
MIN_REQUEST_INTERVAL_SEC = 2.0
USER_AGENT = "spacetrack/0.1 (+https://github.com/xRainbowdartx/Starlink-Tracker-)"

log = logging.getLogger(__name__)


class SpaceTrackAuthError(RuntimeError):
    """Credentials missing, rejected, or session expired."""


class SpaceTrackError(RuntimeError):
    """Non-auth Space-Track API failure (HTTP, malformed response, etc.)."""


def _credentials() -> tuple[str, str]:
    identity = os.environ.get("SPACETRACK_IDENTITY")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        raise SpaceTrackAuthError(
            "Set SPACETRACK_IDENTITY and SPACETRACK_PASSWORD env vars to a "
            "valid Space-Track.org account (free at space-track.org)."
        )
    return identity, password


def _login(session: requests.Session) -> None:
    identity, password = _credentials()
    log.debug("Logging in to Space-Track as %s", identity)
    resp = session.post(
        LOGIN_URL,
        data={"identity": identity, "password": password},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    if resp.status_code != 200:
        raise SpaceTrackAuthError(
            f"Login returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    body = resp.text.strip()
    # Space-Track returns empty body or "{}" on success. Errors return
    # JSON like {"Login":"Failed"}.
    if body and "Failed" in body:
        raise SpaceTrackAuthError(f"Login refused by Space-Track: {body[:200]}")


def open_session() -> requests.Session:
    """Open and authenticate a Space-Track session.

    Reuse the returned session for multiple queries within a run — Space-Track
    keeps you logged in for the session cookie's lifetime.
    """
    session = requests.Session()
    _login(session)
    return session


def fetch_starlink_for_range(
    start_utc: datetime,
    end_utc: datetime,
    *,
    session: requests.Session | None = None,
) -> list[ParsedTLE]:
    """Pull every Starlink TLE whose EPOCH falls in ``[start_utc, end_utc)``.

    Both timestamps are interpreted in UTC. ``start_utc`` is inclusive,
    ``end_utc`` is exclusive (matches Python's range semantics and lets you
    backfill a single calendar day by passing ``date`` and ``date + 1 day``).
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)
    if end_utc <= start_utc:
        raise ValueError("end_utc must be after start_utc")

    owned_session = session is None
    if session is None:
        session = open_session()

    # Space-Track's EPOCH filter syntax: "YYYY-MM-DD--YYYY-MM-DD" (range,
    # inclusive both ends at midnight UTC). To get a [start, end) window
    # we query with start..end and accept the small overlap at end's
    # midnight — duplicates dedupe naturally on insert.
    epoch_filter = (
        f"{start_utc.strftime('%Y-%m-%d')}--{end_utc.strftime('%Y-%m-%d')}"
    )

    # ~~STARLINK~~ is Space-Track's substring match on OBJECT_NAME.
    query = (
        f"{QUERY_BASE}/class/gp_history"
        f"/EPOCH/{epoch_filter}"
        f"/OBJECT_NAME/~~STARLINK~~"
        f"/orderby/NORAD_CAT_ID%20asc,EPOCH%20desc"
        f"/format/json"
    )
    log.info("Space-Track gp_history query: %s", query)

    time.sleep(MIN_REQUEST_INTERVAL_SEC)
    try:
        resp = session.get(query, headers={"User-Agent": USER_AGENT}, timeout=180)
    finally:
        if owned_session:
            # We opened it; let GC close it after we're done parsing.
            pass

    if resp.status_code in (401, 403):
        raise SpaceTrackAuthError(
            f"Space-Track rejected the query (HTTP {resp.status_code}): "
            f"{resp.text[:200]}"
        )
    if resp.status_code == 429:
        raise SpaceTrackError(
            "Rate-limited by Space-Track (HTTP 429). Wait a minute and retry."
        )
    if resp.status_code != 200:
        raise SpaceTrackError(
            f"gp_history returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    try:
        rows = resp.json()
    except ValueError as exc:
        raise SpaceTrackError(
            f"gp_history returned non-JSON body: {resp.text[:200]}"
        ) from exc

    log.info("Received %d gp_history rows for %s", len(rows), epoch_filter)

    parsed: list[ParsedTLE] = []
    for row in rows:
        # TLE_LINE0 looks like "0 STARLINK-1234"; strip the leading "0 ".
        line0 = (row.get("TLE_LINE0") or "").strip()
        name = line0[2:].strip() if line0.startswith("0 ") else (
            line0 or (row.get("OBJECT_NAME") or "").strip()
        )
        line1 = (row.get("TLE_LINE1") or "").rstrip()
        line2 = (row.get("TLE_LINE2") or "").rstrip()
        if not name or not line1 or not line2:
            continue
        try:
            parsed.append(parse(name, line1, line2))
        except Exception as exc:
            log.warning("Skipping malformed TLE for %s: %s", name, exc)
    return parsed


def fetch_starlink_for_date(
    date_utc: datetime,
    *,
    window_days: int = 1,
    session: requests.Session | None = None,
) -> list[ParsedTLE]:
    """Convenience wrapper: backfill a single UTC calendar date.

    ``date_utc`` is treated as the start of a UTC day; the query window is
    ``[date_utc, date_utc + window_days)``.
    """
    start = date_utc.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = start + timedelta(days=window_days)
    return fetch_starlink_for_range(start, end, session=session)
