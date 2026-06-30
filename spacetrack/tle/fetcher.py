"""Fetch TLE data from CelesTrak.

CelesTrak's GP feed is the canonical free source for current TLEs. The Starlink
group endpoint returns a 3-line-element block (name + line1 + line2 per sat).
"""

from __future__ import annotations

import logging
import time
from importlib.resources import files

import requests

from spacetrack.tle.parser import ParsedTLE, parse_block

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
USER_AGENT = "spacetrack/0.1 (+https://github.com/xRainbowdartx/Starlink-Tracker-)"

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


class NoNewData(Exception):
    """CelesTrak 403: data unchanged since last download."""
    pass


def fetch_group(
    group: str = "starlink",
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    params = {"GROUP": group, "FORMAT": "tle"}
    headers = {"User-Agent": USER_AGENT}

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        log.debug("Fetching CelesTrak group=%s (attempt %d/%d)", group, attempt, retries)
        try:
            resp = requests.get(
                CELESTRAK_GP_URL, params=params, headers=headers, timeout=timeout
            )
        except requests.RequestException as exc:
            last_exc = exc
            log.warning("network error on attempt %d: %s", attempt, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
            raise FetchError(f"network error fetching {group}: {exc}") from exc

        if resp.status_code == 403:
            # CelesTrak returns 403 when data hasn't changed since last download.
            if "has not updated since" in resp.text:
                import re
                m = re.search(r"at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC)", resp.text)
                since = m.group(1) if m else "recently"
                raise NoNewData(f"Catalog already current as of {since}. CelesTrak updates every 2 hours.")
            raise FetchError(f"CelesTrak returned HTTP 403")

        if resp.status_code != 200:
            if resp.status_code in (429, 502, 503, 504) and attempt < retries:
                log.warning("CelesTrak HTTP %d, retrying", resp.status_code)
                time.sleep(backoff * attempt)
                continue
            raise FetchError(f"CelesTrak returned HTTP {resp.status_code}")

        text = resp.text.strip()
        if not text or text.lower().startswith("no gp data"):
            raise FetchError(f"CelesTrak returned no data for group={group!r}")

        return text

    raise FetchError(f"exhausted retries fetching {group}: {last_exc}")


def fetch_starlink(*, timeout: float = 30.0) -> list[ParsedTLE]:
    raw = fetch_group("starlink", timeout=timeout)
    parsed = parse_block(raw)
    log.info("Parsed %d Starlink TLEs", len(parsed))
    return parsed


def fetch_active(*, timeout: float = 30.0) -> list[ParsedTLE]:
    """Fetch CelesTrak's `active` group: every operational satellite.

    Includes Starlink — callers building a non-Starlink catalog should filter
    by NORAD ID against the Starlink set.
    """
    raw = fetch_group("active", timeout=timeout)
    parsed = parse_block(raw)
    log.info("Parsed %d active-catalog TLEs", len(parsed))
    return parsed


# Major tracked debris/collision clouds in LEO. These are the objects that
# dominate real conjunction risk for a constellation in the 500–600 km shells —
# the ASAT-test and accidental-collision debris fields that Space Domain
# Awareness operators screen against. Operational satellites ('active') alone
# miss the bulk of the actual hazard population.
DEBRIS_GROUPS: tuple[str, ...] = (
    "cosmos-1408-debris",   # 2021 Russian ASAT test (~1500 tracked pieces, LEO)
    "fengyun-1c-debris",    # 2007 Chinese ASAT test (largest tracked cloud)
    "iridium-33-debris",    # 2009 Iridium–Cosmos collision
    "cosmos-2251-debris",   # 2009 Iridium–Cosmos collision (other body)
)

# Default catalog used for conjunction screening: operational sats + the major
# debris clouds.
DEFAULT_CATALOG_GROUPS: tuple[str, ...] = ("active",) + DEBRIS_GROUPS


def fetch_catalog_groups(
    groups: tuple[str, ...] = DEFAULT_CATALOG_GROUPS,
    *,
    timeout: float = 30.0,
) -> list[ParsedTLE]:
    """Fetch several CelesTrak groups and merge into one deduped catalog.

    Each group is fetched independently; a group that 403s (NoNewData) or
    errors is logged and skipped rather than aborting the whole catalog — a
    single stale or renamed debris group shouldn't sink the refresh. TLEs are
    deduped by NORAD id (first occurrence wins, so 'active' takes precedence
    over a debris-group duplicate).
    """
    by_norad: dict[int, ParsedTLE] = {}
    fetched_groups: list[str] = []
    for group in groups:
        try:
            raw = fetch_group(group, timeout=timeout)
        except NoNewData as exc:
            log.info("Group %s unchanged, skipping: %s", group, exc)
            continue
        except FetchError as exc:
            log.warning("Group %s failed, skipping: %s", group, exc)
            continue
        added = 0
        for tle in parse_block(raw):
            if tle.norad_id not in by_norad:
                by_norad[tle.norad_id] = tle
                added += 1
        fetched_groups.append(f"{group}(+{added})")

    log.info(
        "Catalog merge: %d unique objects from %d/%d groups [%s]",
        len(by_norad), len(fetched_groups), len(groups), ", ".join(fetched_groups),
    )
    return list(by_norad.values())


def load_bundled_seed() -> list[ParsedTLE]:
    """Return the bundled Starlink TLE snapshot.

    Used as a last-resort fallback for environments that can't reach
    CelesTrak (e.g. Streamlit Cloud's egress IPs). The snapshot is
    refreshed by re-running `spacetrack update` locally and re-extracting
    the seed file.
    """
    raw = files("spacetrack.data").joinpath("starlink_seed.tle").read_text(encoding="utf-8")
    parsed = parse_block(raw)
    log.info("Loaded %d Starlink TLEs from bundled seed", len(parsed))
    return parsed


def now_unix() -> int:
    return int(time.time())
