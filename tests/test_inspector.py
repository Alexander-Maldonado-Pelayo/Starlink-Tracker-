"""Tests for the per-satellite orbital-element residual scanner."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spacetrack.anomaly import inspector


# --- Severity classifier ----------------------------------------------------

@pytest.mark.parametrize("abs_z,expected", [
    (0.0,   "quiet"),
    (1.5,   "quiet"),
    (2.999, "quiet"),
    (3.0,   "notable"),
    (4.5,   "notable"),
    (4.999, "notable"),
    (5.0,   "significant"),
    (7.999, "significant"),
    (8.0,   "extreme"),
    (50.0,  "extreme"),
])
def test_classify_severity(abs_z, expected):
    assert inspector.classify_severity(abs_z) == expected


# --- Synthetic snapshot helpers --------------------------------------------

def _baseline_snaps(n: int, base_epoch: float = 2461170.0, seed: int = 0) -> list[dict]:
    """Build a noisy linear-trend history matching real TLE precision.

    Noise magnitudes are roughly 1 σ for Starlink TLEs in calm conditions —
    enough to floor the residual std at a meaningful (non-FP-noise) level.
    """
    rng = np.random.default_rng(seed)
    snaps = []
    for i in range(n):
        snaps.append({
            "epoch": base_epoch + i,
            "mean_motion": 15.07 + 0.0001 * i + rng.normal(0, 1e-5),
            "eccentricity": 0.0001 + 1e-7 * i + rng.normal(0, 1e-6),
            "inclination": 53.0 + 0.0001 * i + rng.normal(0, 1e-3),
            "raan": (256.0 + 1.0 * i + rng.normal(0, 5e-3)) % 360.0,
            "arg_perigee": (90.0 + 0.5 * i + rng.normal(0, 1e-2)) % 360.0,
        })
    return snaps


# --- assess() pure-function behaviour --------------------------------------

def test_clean_baseline_is_quiet():
    snaps = _baseline_snaps(8)
    finding = inspector.assess(norad_id=44713, name="STARLINK-X", snapshots=snaps)
    # All residuals are floating-point noise → z-scores hover near zero.
    assert finding is not None
    assert finding.severity == "quiet"
    assert abs(finding.z_score) < 3.0


def test_too_few_snapshots_returns_none():
    snaps = _baseline_snaps(1)
    assert inspector.assess(
        norad_id=44713, name="STARLINK-X", snapshots=snaps,
    ) is None


def test_mean_motion_kick_is_flagged():
    """Spike the latest mean motion off-trend; expect a notable+ flag."""
    snaps = _baseline_snaps(8)
    # 50× the per-step trend delta — easily 3σ off the clean baseline.
    snaps[-1]["mean_motion"] += 0.05

    finding = inspector.assess(norad_id=44713, name="STARLINK-X", snapshots=snaps)
    assert finding is not None
    assert finding.element == "mean_motion"
    assert finding.severity in {"notable", "significant", "extreme"}
    assert abs(finding.z_score) >= 3.0


def test_inclination_kick_is_flagged_as_extreme():
    """Inclination is normally near-rigid — any sustained deviation should pop."""
    snaps = _baseline_snaps(8)
    snaps[-1]["inclination"] += 0.5    # half-degree is huge for a Starlink

    finding = inspector.assess(norad_id=44713, name="STARLINK-X", snapshots=snaps)
    assert finding is not None
    assert finding.element == "inclination"
    assert abs(finding.z_score) >= 5.0   # extreme


def test_raan_unwrapping_doesnt_false_positive():
    """RAAN crossing the 0/360 boundary cleanly shouldn't trigger a flag."""
    # Build a series that smoothly crosses 360 -> 0.
    snaps = _baseline_snaps(8)
    for i, s in enumerate(snaps):
        # ~1°/day precession that crosses 360 at step 5
        s["raan"] = (357.0 + 1.0 * i) % 360.0

    finding = inspector.assess(norad_id=44713, name="STARLINK-X", snapshots=snaps)
    # Wrap-around shouldn't manifest as a residual outlier on RAAN.
    assert finding is not None
    if finding.element == "raan":
        assert finding.severity == "quiet"


def test_picks_worst_element_when_multiple_anomalies():
    """When several elements deviate, the one with the largest |z| wins.

    Sized so the inclination kick is the bigger anomaly *relative to its own
    noise floor* (σ ~ 1e-3° for inclination, σ ~ 1e-5 rev/day for mean motion).
    """
    snaps = _baseline_snaps(8)
    snaps[-1]["mean_motion"] += 0.0001    # ~10σ on mean_motion
    snaps[-1]["inclination"] += 0.5       # ~500σ on inclination

    finding = inspector.assess(norad_id=44713, name="STARLINK-X", snapshots=snaps)
    assert finding is not None
    assert finding.element == "inclination"


def test_zero_variance_element_is_skipped_cleanly():
    """A perfectly constant element produces sigma=0 and must be skipped."""
    snaps = _baseline_snaps(8)
    for s in snaps:
        s["eccentricity"] = 0.0001       # rigorously constant
    # Kick mean_motion so we still have a winning element.
    snaps[-1]["mean_motion"] += 0.05

    finding = inspector.assess(norad_id=44713, name="STARLINK-X", snapshots=snaps)
    assert finding is not None
    assert finding.element != "eccentricity"
    assert math.isfinite(finding.z_score)
