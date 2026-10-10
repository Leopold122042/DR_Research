"""Strict experiment configuration loading; scenario seeds are call arguments."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tomllib

SCALES = {"S1": (8, 2, 1), "S2": (32, 8, 4), "S3": (128, 32, 16),
          "S4": (512, 128, 64), "S5": (2048, 512, 256)}

_FIELDS = {
    "simulation": {"horizon", "dt_hours"},
    "weather": {"provider", "default_week"},
    "graph": {"buildings_per_station", "pumps_per_tank", "tanks_per_building", "neighbor_radius", "extra_edge_probability"},
    "tank": {"model", "volume_m3", "water_heat_capacity_kwh_per_m3_k", "reference_c", "maximum_c", "initial_c", "terminal_c", "eta_charge", "eta_discharge", "ua_kw_per_k", "environment_c", "charge_capacity_fraction", "discharge_capacity_fraction"},
    "operation": {"availability", "minimum_load_fraction", "ramp_fraction_per_step", "initial_load_fraction", "power_band_center_fraction", "power_band_half_width_fraction", "source_capacity_fraction"},
    "demand": {"indoor_c", "outdoor_mean_c", "outdoor_amplitude_c", "outdoor_phase_hour", "mean_load_fraction", "building_factor_half_width", "edge_capacity_fraction", "occupancy_gain_gamma", "occupancy_transition_hours", "reference_degree_difference_c"},
    "tariff": {"valley", "flat", "peak", "critical_peak", "hourly_periods"},
    "provenance": {"pumps_url", "tank_url", "note"},
}


def _keys(value, expected, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a table")
    unknown, missing = set(value) - expected, expected - set(value)
    if unknown or missing:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}; missing fields {sorted(missing)}")


def _number(value, path, low=None, high=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected a number")
    if not math.isfinite(value):
        raise ValueError(f"{path}: must be finite")
    if integer and not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{path}: outside [{low}, {high}]")


def validate_config(config: dict) -> None:
    """Reject typos and unsupported values before allocating a graph."""
    if not isinstance(config, dict):
        raise ValueError("config: expected a table")
    version = config.get("schema_version")
    if type(version) is not int or version != 2:
        raise ValueError("schema_version: expected 2 for the weekly experiment")
    sections = {key: set(value) for key, value in _FIELDS.items()}
    _keys(config, set(sections) | {"schema_version", "scales", "pump_models"}, "config")
    for section, fields in sections.items():
        _keys(config[section], fields, section)
    _keys(config["scales"], set(SCALES), "scales")
    for name, dimensions in SCALES.items():
        values = config["scales"][name]
        if not isinstance(values, list) or len(values) != 3 or any(type(v) is not int for v in values) or tuple(values) != dimensions:
            raise ValueError(f"scales.{name}: must be {dimensions}")
    sim, graph, tank, op, demand, tariff = (config[k] for k in ("simulation", "graph", "tank", "operation", "demand", "tariff"))
    _number(sim["horizon"], "simulation.horizon", 1, 10000, integer=True)
    _number(sim["dt_hours"], "simulation.dt_hours", 1e-6, 24)
    if sim["horizon"] != 336 or sim["dt_hours"] != 0.5:
        raise ValueError("simulation: requires 336 half-hour steps")
    weather = config["weather"]
    if weather["provider"] not in ("nanjing_cswd", "synthetic_week"):
        raise ValueError("weather.provider: expected nanjing_cswd or synthetic_week")
    _number(weather["default_week"], "weather.default_week", 0, 10, integer=True)
    _number(demand["occupancy_gain_gamma"], "demand.occupancy_gain_gamma", 0, 1)
    _number(demand["occupancy_transition_hours"], "demand.occupancy_transition_hours", 0, 6)
    _number(demand["reference_degree_difference_c"], "demand.reference_degree_difference_c", 1e-9)
    for key in ("buildings_per_station", "pumps_per_tank", "tanks_per_building", "neighbor_radius"):
        _number(graph[key], f"graph.{key}", 1, 4, integer=True)
    if graph["buildings_per_station"] > 4 or graph["pumps_per_tank"] != 4 or graph["tanks_per_building"] != 2:
        raise ValueError("graph: requires four pumps per tank, two backbone tanks per building, at most four buildings per station")
    _number(graph["extra_edge_probability"], "graph.extra_edge_probability", 0, 1)
    models = config["pump_models"]
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("pump_models: expected exactly three model records")
    for i, model in enumerate(models):
        _keys(model, {"name", "heat_max_kw", "cop", "fraction"}, f"pump_models[{i}]")
        if not isinstance(model["name"], str) or not model["name"].strip():
            raise ValueError(f"pump_models[{i}].name: expected a nonempty string")
        _number(model["heat_max_kw"], f"pump_models[{i}].heat_max_kw", 1e-6)
        _number(model["cop"], f"pump_models[{i}].cop", 1.000001)
        _number(model["fraction"], f"pump_models[{i}].fraction", 0, 1)
    if len({m["name"] for m in models}) != len(models) or not math.isclose(sum(m["fraction"] for m in models), 1, abs_tol=1e-12):
        raise ValueError("pump_models: names must be unique and fractions must sum to one")
    for _, (h, _, _) in SCALES.items():
        if any(not math.isclose(h*m["fraction"], round(h*m["fraction"]), abs_tol=1e-10) for m in models):
            raise ValueError("pump_models.fraction: model counts must be integral at every scale")
    if not isinstance(tank["model"], str) or not tank["model"].strip():
        raise ValueError("tank.model: expected a nonempty string")
    for key, value in tank.items():
        if key != "model":
            _number(value, f"tank.{key}")
    for key in ("volume_m3", "water_heat_capacity_kwh_per_m3_k", "charge_capacity_fraction", "discharge_capacity_fraction"):
        _number(tank[key], f"tank.{key}", 1e-9)
    for key in ("eta_charge", "eta_discharge"):
        _number(tank[key], f"tank.{key}", 1e-9, 1)
    _number(tank["ua_kw_per_k"], "tank.ua_kw_per_k", 0)
    if not 0 <= tank["reference_c"] < tank["maximum_c"] <= 110:
        raise ValueError("tank: reference < maximum <= device limit 110 C required")
    if any(not tank["reference_c"] <= tank[k] <= tank["maximum_c"] for k in ("initial_c", "terminal_c")):
        raise ValueError("tank: initial and terminal temperatures must be in the operating range")
    if sim["dt_hours"] * tank["ua_kw_per_k"] >= tank["volume_m3"]*tank["water_heat_capacity_kwh_per_m3_k"]:
        raise ValueError("tank: time step is too large for the explicit heat-loss model")
    if type(op["availability"]) is not bool or not op["availability"]:
        raise ValueError("operation.availability: this fixed-mode experiment requires true")
    for key, value in op.items():
        if key != "availability":
            _number(value, f"operation.{key}", 0, 1)
    if op["ramp_fraction_per_step"] <= 0 or op["source_capacity_fraction"] <= 0:
        raise ValueError("operation: ramp and source fractions must be positive")
    if op["initial_load_fraction"] < op["minimum_load_fraction"]:
        raise ValueError("operation.initial_load_fraction: below minimum load")
    center, width = op["power_band_center_fraction"], op["power_band_half_width_fraction"]
    if center-width < 0 or center+width > 1 or not center-width <= op["initial_load_fraction"] <= center+width:
        raise ValueError("operation: power band must lie in [0,1] and contain the initial load fraction")
    for key, value in demand.items():
        _number(value, f"demand.{key}")
    _number(demand["outdoor_amplitude_c"], "demand.outdoor_amplitude_c", 0)
    _number(demand["outdoor_phase_hour"], "demand.outdoor_phase_hour", 0, 24)
    _number(demand["mean_load_fraction"], "demand.mean_load_fraction", 1e-9, 1)
    _number(demand["building_factor_half_width"], "demand.building_factor_half_width", 0, 0.2)
    _number(demand["edge_capacity_fraction"], "demand.edge_capacity_fraction", 1e-9)
    periods = {"valley", "flat", "peak", "critical_peak"}
    for key in periods:
        _number(tariff[key], f"tariff.{key}", 0)
    if not isinstance(tariff["hourly_periods"], list) or len(tariff["hourly_periods"]) != 24 or any(not isinstance(p, str) or p not in periods for p in tariff["hourly_periods"]):
        raise ValueError("tariff.hourly_periods: expected 24 valid period names")
    for key, value in config["provenance"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"provenance.{key}: expected a nonempty string")


def load_config(path: str | Path) -> dict:
    """Read TOML, reject unknown fields and return an independent validated dict."""
    try:
        with Path(path).open("rb") as handle:
            config = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc
    validate_config(config)
    return copy.deepcopy(config)
