"""SGP4 propagation via Skyfield.

Given a TLE and a UTC datetime, compute the satellite's geographic position
(latitude, longitude, altitude) and its inertial speed. Skyfield wraps the
`sgp4` C extension and handles all the frame conversions
(TEME -> ECI -> ECEF -> WGS84 geodetic) for us.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from skyfield.api import EarthSatellite, load, wgs84


class PropagationError(RuntimeError):
    """SGP4/Skyfield could not produce a valid position for this satellite.

    Most commonly raised when a satellite is in active re-entry: its perigee
    has dropped low enough that the SGP4 model's atmospheric assumptions
    break down and the propagated position degrades to NaN. Callers that
    iterate over many TLEs (``propagate_many``) catch and skip; callers
    that follow one satellite (``live_sample``, ``where``) surface it.
    """


@dataclass(frozen=True)
class SatPosition:
    norad_id: int
    name: str
    when: datetime              # UTC
    latitude: float             # degrees, +N / -S
    longitude: float            # degrees, +E / -W
    altitude_km: float          # height above WGS84 ellipsoid
    speed_km_s: float           # magnitude of inertial velocity


@lru_cache(maxsize=1)
def _timescale():
    # Skyfield needs a Timescale; building one is non-trivial, so cache it.
    return load.timescale()


def propagate(
    name: str, line1: str, line2: str, when: datetime | None = None
) -> SatPosition:
    if when is None:
        when = datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware (UTC)")

    ts = _timescale()
    sat = EarthSatellite(line1, line2, name, ts)
    t = ts.from_datetime(when.astimezone(timezone.utc))

    geocentric = sat.at(t)
    subpoint = wgs84.subpoint_of(geocentric)
    altitude_km = wgs84.height_of(geocentric).km

    # geocentric.velocity is in AU/day; convert to km/s.
    vx, vy, vz = geocentric.velocity.km_per_s
    speed = (vx * vx + vy * vy + vz * vz) ** 0.5

    lat = subpoint.latitude.degrees
    lon = subpoint.longitude.degrees
    # Skyfield silently emits NaN when SGP4 hits its validity edge (very
    # low-perigee deorbit phase). Surface that as a real error instead.
    if not all(math.isfinite(v) for v in (lat, lon, altitude_km, speed)):
        raise PropagationError(
            f"SGP4 produced NaN/inf for {name} — the satellite has likely "
            "decayed below the model's validity range."
        )

    return SatPosition(
        norad_id=sat.model.satnum,
        name=name,
        when=when,
        latitude=lat,
        longitude=lon,
        altitude_km=altitude_km,
        speed_km_s=speed,
    )


def propagate_many(
    tles: Iterable[tuple[str, str, str]],
    when: datetime | None = None,
) -> list[SatPosition]:
    """Propagate many TLEs to the same instant. Skips ones SGP4 rejects."""
    if when is None:
        when = datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("`when` must be timezone-aware (UTC)")

    out: list[SatPosition] = []
    for name, line1, line2 in tles:
        try:
            out.append(propagate(name, line1, line2, when=when))
        except Exception:
            # A handful of TLEs in any large feed fail SGP4's plausibility checks.
            # Skip them rather than abort the whole batch.
            continue
    return out


def propagate_track(
    name: str,
    line1: str,
    line2: str,
    *,
    start: datetime | None = None,
    minutes: float = 200.0,
    step_seconds: float = 30.0,
) -> list[SatPosition]:
    """Propagate a single TLE forward in time to build a ground-track path."""
    if start is None:
        start = datetime.now(timezone.utc)
    if start.tzinfo is None:
        raise ValueError("`start` must be timezone-aware (UTC)")

    ts = _timescale()
    sat = EarthSatellite(line1, line2, name, ts)

    samples = max(2, int(minutes * 60.0 / step_seconds) + 1)
    seconds = [i * step_seconds for i in range(samples)]
    start_utc = start.astimezone(timezone.utc)
    datetimes = [start_utc + timedelta(seconds=s) for s in seconds]
    t = ts.from_datetimes(datetimes)

    geocentric = sat.at(t)
    subpoint = wgs84.subpoint_of(geocentric)
    heights_km = wgs84.height_of(geocentric).km

    base_ts = start.astimezone(timezone.utc).timestamp()
    lats = subpoint.latitude.degrees
    lons = subpoint.longitude.degrees
    out: list[SatPosition] = []
    for i in range(samples):
        out.append(
            SatPosition(
                norad_id=sat.model.satnum,
                name=name,
                when=datetime.fromtimestamp(base_ts + seconds[i], tz=timezone.utc),
                latitude=float(lats[i]),
                longitude=float(lons[i]),
                altitude_km=float(heights_km[i]),
                speed_km_s=0.0,
            )
        )
    return out
