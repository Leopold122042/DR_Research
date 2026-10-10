from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dr_research.config import SCALES, load_config, validate_config
from dr_research.scenario import generate_scenario, validate_scenario
from dr_research.weather import load_weather_week, residential_occupancy

CONFIG = Path(__file__).resolve().parents[1]/"configs"/"experiment.toml"


def test_exact_scales_repeatability_and_device_mix():
    config = load_config(CONFIG)
    s = generate_scenario(config, "S2", 901)
    again = generate_scenario(config, "S2", 901)
    assert (s.network.n_pumps, s.network.n_tanks, s.network.n_buildings) == SCALES["S2"]
    np.testing.assert_array_equal(s.network.delivery_edges, again.network.delivery_edges)
    np.testing.assert_array_equal(s.heat_demand_kw, again.heat_demand_kw)
    np.testing.assert_array_equal(s.network.pump_cop, again.network.pump_cop)
    assert s.scenario_id == again.scenario_id
    assert len(s.config_digest) == 64
    counts = {name: s.network.pump_models.count(name) for name in set(s.network.pump_models)}
    assert sorted(counts.values()) == [8, 8, 16]
    np.testing.assert_allclose(s.network.pump_max_kw*s.network.pump_cop,
                               [next(m["heat_max_kw"] for m in config["pump_models"] if m["name"] == name) for name in s.network.pump_models])
    changed = generate_scenario(config, "S2", 902)
    assert changed.scenario_id != s.scenario_id
    assert not np.array_equal(changed.heat_demand_kw, s.heat_demand_kw)
    state = s.initial_state()
    state.energy_kwh[:] = 0
    assert np.all(s.initial_energy_kwh > 0)


def test_s5_graph_is_station_local_sparse_and_all_buildings_reachable():
    config = load_config(CONFIG)
    s = generate_scenario(config, "S5", 44)
    n = s.network
    assert (n.n_pumps, n.n_tanks, n.n_buildings, n.n_stations) == (2048, 512, 256, 64)
    np.testing.assert_array_equal(np.bincount(n.pump_to_tank), np.full(512, 4))
    assert np.all(n.tank_station[n.delivery_edges[0]] == n.building_station[n.delivery_edges[1]])
    assert np.all(np.bincount(n.delivery_edges[1], minlength=256) >= 2)
    assert n.n_edges <= 4*n.n_tanks
    edges = set(map(tuple, n.delivery_edges.T.tolist()))
    assert all((tank, tank//2) in edges for tank in range(n.n_tanks))
    assert np.all(np.bincount(n.building_station) <= 4)
    assert np.any(np.bincount(n.delivery_edges[0]) > 1)
    heat_max = n.pump_max_kw*n.pump_cop
    demand = config["demand"]
    occupancy = residential_occupancy(np.arange(s.horizon)*s.dt_hours,
                                       demand["occupancy_transition_hours"])
    shape = (np.maximum(demand["indoor_c"]-s.ambient_c, 0)
             / demand["reference_degree_difference_c"]
             * (1+demand["occupancy_gain_gamma"]*occupancy))
    for station in range(n.n_stations):
        actual = s.heat_demand_kw[:, n.building_station == station].sum(axis=1)
        expected = demand["mean_load_fraction"]*heat_max[n.pump_station == station].sum()*shape
        np.testing.assert_allclose(actual, expected)


def test_half_hour_tariff_boundary_values_and_temperatures():
    config = load_config(CONFIG)
    s = generate_scenario(config, "S1", 17)
    expected = {0: .3815, 11: .3815, 12: .6756, 21: .6756, 22: .3815,
                25: .3815, 26: .6756, 27: .6756, 28: 1.0376, 35: 1.0376,
                36: 1.2005, 39: 1.2005, 40: 1.0376, 43: 1.0376,
                44: .6756, 47: .6756}
    for t, price in expected.items():
        np.testing.assert_array_equal(s.price_yuan_per_kwh[t::48], price)
    weather, _ = load_weather_week(config["weather"]["default_week"])
    np.testing.assert_array_equal(s.ambient_c, weather)
    np.testing.assert_allclose(s.network.tank_capacity_kwh, 1163)
    np.testing.assert_allclose(s.initial_energy_kwh, 581.5)
    np.testing.assert_allclose(s.tank_environment_c, 10)


@pytest.mark.parametrize("mutate", [
    lambda c: c.update({"seed": 42}),
    lambda c: c["tank"].update({"typo_ua": .1}),
    lambda c: c["tank"].update({"ua_kw_per_k": float("nan")}),
    lambda c: c["pump_models"][0].update({"cop": float("inf")}),
    lambda c: c["pump_models"][0].update({"cop": 1}),
    lambda c: c["simulation"].update({"horizon": True}),
    lambda c: c["simulation"].update({"dt_hours": -1}),
    lambda c: c["tank"].update({"initial_c": 120}),
    lambda c: c["graph"].update({"buildings_per_station": 5}),
    lambda c: c["operation"].update({"power_band_half_width_fraction": .8}),
    lambda c: c["tariff"]["hourly_periods"].append("flat"),
    lambda c: c["scales"].update({"S5": [2048, 512, 255]}),
])
def test_invalid_configuration_rejected(mutate):
    config = deepcopy(load_config(CONFIG))
    mutate(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_corrupted_scenario_and_bad_seed_rejected():
    config = load_config(CONFIG)
    for seed in (-1, True, 1.2):
        with pytest.raises(ValueError):
            generate_scenario(config, "S1", seed)
    s = generate_scenario(config, "S3", 1)
    bad_edges = s.network.delivery_edges.copy()
    bad_edges[1, 0] = 4  # tank 0 is at station 0; building 4 is at station 1.
    with pytest.raises(ValueError, match="crosses a station"):
        validate_scenario(replace(s, network=replace(s.network, delivery_edges=bad_edges)))
    with pytest.raises(ValueError, match="shape"):
        validate_scenario(replace(s, heat_demand_kw=s.heat_demand_kw[:, :-1]))
    bad_price = s.price_yuan_per_kwh.copy()
    bad_price[0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        validate_scenario(replace(s, price_yuan_per_kwh=bad_price))
