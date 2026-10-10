"""Versioned, non-pickle scenario persistence and content fingerprints."""
from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .types import Network, Scenario

SCHEMA_VERSION = 1


def write_json(path: str | Path, value: dict) -> None:
    """Exclusive create: repeated experiments never silently replace results."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _parts(s: Scenario) -> tuple[dict, dict]:
    arrays, scalars = {}, {}
    for prefix, obj in (("network", s.network), ("scenario", s)):
        for item in fields(obj):
            if item.name == "network":
                continue
            value = getattr(obj, item.name)
            key = f"{prefix}.{item.name}"
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                scalars[key] = value
    return arrays, scalars


def scenario_fingerprint(s: Scenario) -> str:
    arrays, scalars = _parts(s)
    digest = sha256(json.dumps(scalars, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode())
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(json.dumps([key, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_scenario(path: str | Path, s: Scenario) -> str:
    from .physics import validate_scenario
    validate_scenario(s)
    arrays, scalars = _parts(s)
    fingerprint = scenario_fingerprint(s)
    metadata = {"schema_version": SCHEMA_VERSION, "fingerprint": fingerprint,
                "scalars": scalars}
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.asarray(
            json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
    return fingerprint


def load_scenario(path: str | Path) -> Scenario:
    from .physics import validate_scenario
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported scenario schema version")
        values = dict(metadata["scalars"])
        for key in archive.files:
            if key != "metadata":
                values[key] = archive[key].copy()
    network_values = {k.removeprefix("network."): v for k, v in values.items()
                      if k.startswith("network.")}
    network_values["pump_models"] = tuple(network_values["pump_models"])
    scenario_values = {k.removeprefix("scenario."): v for k, v in values.items()
                       if k.startswith("scenario.")}
    scenario = Scenario(network=Network(**network_values), **scenario_values)
    validate_scenario(scenario)
    if scenario_fingerprint(scenario) != metadata["fingerprint"]:
        raise ValueError("Scenario fingerprint mismatch: file content changed")
    return scenario
