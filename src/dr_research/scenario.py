"""Reproducible station-local heterogeneous graphs and winter scenarios."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .config import SCALES, validate_config
from .physics import validate_scenario
from .types import Network, Scenario
from .weather import load_weather_week, residential_occupancy


def generate_scenario(config: dict, scale: str, seed: int, weather_week: int | None = None) -> Scenario:
    """Generate one explicitly seeded physical case; no random global state."""
    validate_config(config)
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}; expected one of {tuple(SCALES)}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    seed = int(seed)
    if weather_week is not None and (isinstance(weather_week, (bool, np.bool_)) or not isinstance(weather_week, (int, np.integer)) or not 0 <= weather_week < 11):
        raise ValueError("weather_week must be an integer from 0 through 10")
    rng = np.random.default_rng(seed)
    h, s, b = SCALES[scale]
    graph, tank, op, demand = (config[k] for k in ("graph", "tank", "operation", "demand"))
    pump_to_tank = np.repeat(np.arange(s, dtype=np.int64), graph["pumps_per_tank"])
    backbone_building = np.arange(s, dtype=np.int64) // graph["tanks_per_building"]
    building_station = np.arange(b, dtype=np.int64) // graph["buildings_per_station"]
    tank_station = building_station[backbone_building]
    pump_station = tank_station[pump_to_tank]
    stations = int(building_station[-1])+1

    model_ids = np.concatenate([np.full(round(h*m["fraction"]), i, dtype=np.int64)
                                for i, m in enumerate(config["pump_models"])])
    rng.shuffle(model_ids)
    models = config["pump_models"]
    heat_max = np.asarray([models[i]["heat_max_kw"] for i in model_ids], dtype=float)
    cop = np.asarray([models[i]["cop"] for i in model_ids], dtype=float)
    power_max = heat_max/cop
    tank_nominal_heat = np.bincount(pump_to_tank, weights=heat_max, minlength=s)
    building_nominal_heat = np.bincount(backbone_building, weights=tank_nominal_heat, minlength=b)

    # Every tank has its own backbone building. Extra short-range edges stay local.
    edges = {(int(j), int(backbone_building[j])) for j in range(s)}
    for j in range(s):
        owner = int(backbone_building[j])
        first = max(0, owner-graph["neighbor_radius"])
        last = min(b, owner+graph["neighbor_radius"]+1)
        for k in range(first, last):
            if k != owner and tank_station[j] == building_station[k] and rng.random() < graph["extra_edge_probability"]:
                edges.add((j, k))
    edge_index = np.asarray(sorted(edges), dtype=np.int64).T.copy()
    edge_capacity = demand["edge_capacity_fraction"]*tank_nominal_heat[edge_index[0]]

    heat_capacity = tank["volume_m3"]*tank["water_heat_capacity_kwh_per_m3_k"]
    network = Network(
        pump_to_tank=pump_to_tank, delivery_edges=edge_index,
        pump_station=pump_station, tank_station=tank_station, building_station=building_station,
        pump_cop=cop, pump_max_kw=power_max,
        pump_min_kw=op["minimum_load_fraction"]*power_max,
        pump_ramp_kw=op["ramp_fraction_per_step"]*power_max,
        tank_capacity_kwh=np.full(s, heat_capacity*(tank["maximum_c"]-tank["reference_c"])),
        tank_charge_limit_kw=tank["charge_capacity_fraction"]*tank_nominal_heat,
        tank_discharge_limit_kw=tank["discharge_capacity_fraction"]*tank_nominal_heat,
        tank_ua_kw_per_k=np.full(s, tank["ua_kw_per_k"]),
        tank_heat_capacity_kwh_per_k=np.full(s, heat_capacity),
        tank_reference_c=float(tank["reference_c"]),
        eta_charge=np.full(s, tank["eta_charge"]), eta_discharge=np.full(s, tank["eta_discharge"]),
        edge_capacity_kw=edge_capacity,
        pump_models=tuple(models[i]["name"] for i in model_ids),
    )
    horizon, dt = config["simulation"]["horizon"], config["simulation"]["dt_hours"]
    hours = np.arange(horizon)*dt
    selected_week = config["weather"]["default_week"] if weather_week is None else int(weather_week)
    if config["weather"]["provider"] == "nanjing_cswd":
        ambient, weather_metadata = load_weather_week(selected_week)
    else:
        ambient = demand["outdoor_mean_c"] + demand["outdoor_amplitude_c"]*np.sin(2*np.pi*(hours-demand["outdoor_phase_hour"])/24)
        weather_metadata = {"provider": "synthetic_week", "week_id": selected_week}
    occupancy = residential_occupancy(hours, demand["occupancy_transition_hours"])
    daily_shape = (np.maximum(demand["indoor_c"]-ambient, 0) / demand["reference_degree_difference_c"]
                   * (1+demand["occupancy_gain_gamma"]*occupancy))
    factors = rng.uniform(1-demand["building_factor_half_width"], 1+demand["building_factor_half_width"], b)
    # Normalize within each station so heterogeneity cannot change its mean load.
    for station in range(stations):
        mask = building_station == station
        factors[mask] /= np.average(factors[mask], weights=building_nominal_heat[mask])
    demand_kw = daily_shape[:, None]*(demand["mean_load_fraction"]*building_nominal_heat*factors)[None, :]
    tariff = config["tariff"]
    price = np.asarray([tariff[tariff["hourly_periods"][int(hour) % 24]] for hour in hours], dtype=float)
    total_max = power_max.sum()
    source_max = np.bincount(pump_station, weights=heat_max-power_max, minlength=stations)
    digest_content = {"config": config, "weather": weather_metadata}
    digest = hashlib.sha256(json.dumps(digest_content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    weather_tag = f"-week{weather_metadata['week_id']}"
    result = Scenario(
        network=network, dt_hours=float(dt), price_yuan_per_kwh=price,
        heat_demand_kw=demand_kw, ambient_c=ambient,
        tank_environment_c=np.full((horizon, s), tank["environment_c"]),
        source_limit_kw=np.broadcast_to(op["source_capacity_fraction"]*source_max, (horizon, stations)).copy(),
        power_lower_kw=np.full(horizon, (op["power_band_center_fraction"]-op["power_band_half_width_fraction"])*total_max),
        power_upper_kw=np.full(horizon, (op["power_band_center_fraction"]+op["power_band_half_width_fraction"])*total_max),
        availability=np.full((horizon, h), op["availability"], dtype=bool),
        initial_energy_kwh=np.full(s, heat_capacity*(tank["initial_c"]-tank["reference_c"])),
        initial_power_kw=op["initial_load_fraction"]*power_max,
        terminal_energy_kwh=np.full(s, heat_capacity*(tank["terminal_c"]-tank["reference_c"])),
        scenario_id=f"{scale}-seed{seed}{weather_tag}-{digest[:12]}", seed=seed, config_digest=digest,
    )
    validate_scenario(result)
    return result
