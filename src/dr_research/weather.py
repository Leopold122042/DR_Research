"""Fixed, reproducible heating-season weeks from the public Nanjing CSWD EPW.

EPW dry-bulb observations are indexed at the ending hour, 01:00 through 24:00.
The controller samples the linearly interpolated profile at 00:00, 00:30, ... .
The annual boundary is periodic, as appropriate for this synthetic typical year.
The source month/day is retained in metadata; every experimental week starts on
an assigned Monday. This calendar assignment is independent of source EPW years.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parent / "data" / "weather"


@lru_cache(maxsize=1)
def _dataset() -> tuple[np.ndarray, dict]:
    manifest = json.loads((_DATA / "manifest.json").read_text(encoding="utf-8"))
    path = _DATA / manifest["derived_csv"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["derived_csv_sha256"]:
        raise ValueError("weather data SHA256 does not match manifest")
    values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=3)
    if values.shape != (8760,) or not np.isfinite(values).all() or np.any(np.abs(values) > 80):
        raise ValueError("weather data must contain 8760 finite dry-bulb observations")
    weeks = manifest["weeks"]
    if len(weeks) != 11 or [row["week_id"] for row in weeks] != list(range(11)):
        raise ValueError("weather manifest must contain week IDs 0 through 10")
    values.setflags(write=False)
    return values, manifest


def weather_manifest() -> dict:
    """Return independent provenance, physical-screening and split metadata."""
    return json.loads(json.dumps(_dataset()[1]))


def load_weather_week(week_id: int) -> tuple[np.ndarray, dict]:
    """Return 336 half-hour dry-bulb values and an independent metadata dict."""
    if isinstance(week_id, (bool, np.bool_)) or not isinstance(week_id, (int, np.integer)) or not 0 <= week_id < 11:
        raise ValueError("weather_week must be an integer from 0 through 10")
    hourly, manifest = _dataset()
    row = manifest["weeks"][int(week_id)]
    target_hours = (row["start_hour_of_year"] + np.arange(336)*0.5) % 8760
    # At hour zero use the preceding December 31 hour-ending-24 observation.
    ambient = np.interp(target_hours, np.arange(8761), np.r_[hourly[-1], hourly])
    metadata = dict(row, provider="nanjing_cswd", source_epw_sha256=manifest["source_epw_sha256"],
                    derived_csv_sha256=manifest["derived_csv_sha256"])
    return ambient, metadata


def residential_occupancy(hours: np.ndarray, transition_hours: float = 1.0) -> np.ndarray:
    """Monday-origin weekly occupancy with linear, centered transitions.

    Weekdays are occupied before 07:00 and from 20:00; weekends are occupied all
    day. A one-hour transition ramps over 06:30--07:30 and 19:30--20:30. The
    multiplicative demand response to this schedule is a research assumption,
    rather than an indoor-temperature or internal-heat-gain model.
    """
    hours = np.asarray(hours, dtype=float)
    if not np.isfinite(hours).all() or not np.isfinite(transition_hours) or not 0 <= transition_hours <= 6:
        raise ValueError("occupancy hours must be finite and transition must lie in [0,6]")
    wrapped = hours % 168
    if transition_hours == 0:
        return (((wrapped // 24) >= 5) | ((wrapped % 24) < 7) | ((wrapped % 24) >= 20)).astype(float)
    knots, values = [0.0], [1.0]
    half = transition_hours / 2
    for day in range(5):
        for event, before, after in ((24*day+7, 1.0, 0.0), (24*day+20, 0.0, 1.0)):
            knots.extend([event-half, event+half])
            values.extend([before, after])
    knots.append(168.0)
    values.append(1.0)
    return np.interp(wrapped, knots, values)
