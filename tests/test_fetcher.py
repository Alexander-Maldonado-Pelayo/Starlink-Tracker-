"""Tests for the multi-group catalog merge (debris-inclusive coverage)."""

from __future__ import annotations

import pytest

from spacetrack.tle import fetcher
from spacetrack.tle.fetcher import FetchError, NoNewData
from spacetrack.tle.parser import ParsedTLE


def _tle(norad_id: int) -> ParsedTLE:
    return ParsedTLE(
        name=f"OBJ-{norad_id}", norad_id=norad_id,
        line1="L1", line2="L2", epoch_jd=2461170.0,
        inclination=53.0, raan=100.0, eccentricity=0.0001,
        arg_perigee=90.0, mean_anomaly=270.0, mean_motion=15.07,
    )


def test_catalog_merge_dedupes_first_wins(monkeypatch):
    # active: 1,2 ; debris group: 2 (dup),3 — merged should be {1,2,3}.
    group_objs = {
        "active": [_tle(1), _tle(2)],
        "cosmos-1408-debris": [_tle(2), _tle(3)],
    }
    monkeypatch.setattr(fetcher, "fetch_group", lambda g, **k: g)
    monkeypatch.setattr(fetcher, "parse_block", lambda raw: group_objs[raw])

    merged = fetcher.fetch_catalog_groups(("active", "cosmos-1408-debris"))
    ids = sorted(t.norad_id for t in merged)
    assert ids == [1, 2, 3]


def test_catalog_merge_tolerates_group_failures(monkeypatch):
    def fake_fetch(group, **kwargs):
        if group == "active":
            return "active"
        if group == "stale-debris":
            raise NoNewData("unchanged")
        if group == "broken-debris":
            raise FetchError("HTTP 500")
        raise AssertionError(group)

    monkeypatch.setattr(fetcher, "fetch_group", fake_fetch)
    monkeypatch.setattr(fetcher, "parse_block", lambda raw: [_tle(1), _tle(2)])

    # One good group + one 403 + one hard error -> still returns the good data.
    merged = fetcher.fetch_catalog_groups(
        ("active", "stale-debris", "broken-debris")
    )
    assert sorted(t.norad_id for t in merged) == [1, 2]


def test_default_catalog_groups_include_debris():
    assert fetcher.DEFAULT_CATALOG_GROUPS[0] == "active"
    assert "cosmos-1408-debris" in fetcher.DEFAULT_CATALOG_GROUPS
    assert set(fetcher.DEBRIS_GROUPS).issubset(set(fetcher.DEFAULT_CATALOG_GROUPS))
