"""Weekly data, calendar, demand assumptions, and physical feasibility checks."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dr_research.config import SCALES, load_config, validate_config
from dr_research.control import construct_reference
from dr_research.physics import transition, validate_action
from dr_research.scenario import generate_scenario
from dr_research.types import State
from dr_research.weather import load_weather_week, residential_occupancy, weather_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment.toml"


def test_default_week_is_336_half_hours_and_reproducible():
    config = load_config(CONFIG)
    a = generate_scenario(config, "S2", 42)
    b = generate_scenario(config, "S2", 42, weather_week=10)
    assert config["schema_version"] == 2
    assert a.horizon == 336 and a.dt_hours == .5
    assert a.scenario_id == b.scenario_id
    for field in ("ambient_c", "heat_demand_kw", "price_yuan_per_kwh"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(a.network.delivery_edges, b.network.delivery_edges)
    different = generate_scenario(config, "S2", 42, weather_week=9)
    assert different.config_digest != a.config_digest
    assert different.scenario_id != a.scenario_id
    assert not np.array_equal(a.ambient_c, different.ambient_c)


def test_same_weather_is_independent_of_scale_graph_seed_and_cwd(tmp_path, monkeypatch):
    config = load_config(CONFIG)
    expected, _ = load_weather_week(4)
    monkeypatch.chdir(tmp_path)
    for index, scale in enumerate(SCALES):
        case = generate_scenario(config, scale, seed=42+index, weather_week=4)
        np.testing.assert_array_equal(case.ambient_c, expected)


def test_weekend_daytime_demand_and_smooth_transition_at_equal_temperature():
    config = load_config(CONFIG)
    config["weather"]["provider"] = "synthetic_week"
    config["demand"]["outdoor_amplitude_c"] = 0.0
    case = generate_scenario(config, "S1", 42)
    # Monday noon is away, Saturday noon occupied; temperature is held equal.
    assert case.heat_demand_kw[5*48+24, 0] / case.heat_demand_kw[24, 0] == pytest.approx(1.1)
    assert case.heat_demand_kw[5*48, 0] == pytest.approx(case.heat_demand_kw[0, 0])
    sampled = residential_occupancy(np.array([6.5, 7.0, 7.5, 19.5, 20.0, 20.5, 132.0]))
    np.testing.assert_allclose(sampled, [1, .5, 0, 0, .5, 1, 1])
    # There is no weekly renormalization that erases the calendar load increase.
    nominal = np.sum(case.network.pump_cop*case.network.pump_max_kw)
    assert case.heat_demand_kw[24, 0] == pytest.approx(.45*nominal)


def test_tank_state_continues_across_midnight_without_reset():
    case = generate_scenario(load_config(CONFIG), "S1", 42)
    reference = construct_reference(case)
    state = State(47, case.initial_energy_kwh+17, reference.power_kw[46].copy())
    action = reference.action(47)
    assert validate_action(case, state, action).feasible
    midnight = transition(case, state, action)
    assert midnight.t == 48
    assert np.all(midnight.energy_kwh > case.initial_energy_kwh+16)
    next_state = transition(case, midnight, reference.action(48))
    assert next_state.t == 49
    assert np.all(next_state.energy_kwh > case.initial_energy_kwh+16)


def test_weather_split_is_fixed_and_seven_day_blocks_do_not_overlap():
    manifest = weather_manifest()
    weeks = manifest["weeks"]
    assert [week["split"] for week in weeks] == ["train"]*8+["validation"]*2+["test"]
    covered = set()
    for week in weeks:
        slots = set((week["start_hour_of_year"]+i) % 8760 for i in range(168))
        assert not covered.intersection(slots)
        covered.update(slots)
        assert week["reference_feasible_all_scales_seed42"]
        values, metadata = load_weather_week(week["week_id"])
        assert len(values) == 336 and np.isfinite(values).all()
        assert metadata["source_epw_sha256"] == manifest["source_epw_sha256"]
    # Public metadata is independent: callers cannot change the cached dataset.
    manifest["weeks"].clear()
    assert len(weather_manifest()["weeks"]) == 11


@pytest.mark.parametrize("scale", list(SCALES))
@pytest.mark.parametrize("week", range(11))
def test_reference_is_feasible_for_every_published_week_and_scale(scale, week):
    case = generate_scenario(load_config(CONFIG), scale, 42, weather_week=week)
    # construct_reference includes independent full-trajectory physical replay.
    reference = construct_reference(case)
    assert reference.power_kw.shape == (336, case.network.n_pumps)
    np.testing.assert_allclose(reference.energy_kwh[-1], case.terminal_energy_kwh)


def test_synthetic_week_preserves_sinusoidal_weather_and_weekend_occupancy():
    config = load_config(CONFIG)
    config["weather"]["provider"] = "synthetic_week"
    case = generate_scenario(config, "S1", 17)
    assert case.horizon == 336
    np.testing.assert_allclose(case.ambient_c[12::48], 0, atol=1e-12)
    np.testing.assert_allclose(case.ambient_c[36::48], 10, atol=1e-12)
    qmax = np.sum(case.network.pump_max_kw*case.network.pump_cop)
    # Monday noon has no occupancy increment; Saturday noon has ten percent.
    assert case.heat_demand_kw[24, 0] == pytest.approx(.45*qmax)
    assert case.heat_demand_kw[5*48+24, 0] == pytest.approx(.45*qmax*1.1)
    other = generate_scenario(config, "S1", 17, weather_week=0)
    np.testing.assert_array_equal(case.ambient_c, other.ambient_c)
    assert case.scenario_id != other.scenario_id


def test_building_random_factor_is_fixed_for_the_week_and_s1_normalizes_to_one():
    config = load_config(CONFIG)
    # Fixed weather and occupancy expose any unintended time-dependent noise.
    config["weather"]["provider"] = "synthetic_week"
    config["demand"]["outdoor_amplitude_c"] = 0.0
    config["demand"]["occupancy_gain_gamma"] = 0.0
    s2 = generate_scenario(config, "S2", 42)
    np.testing.assert_array_equal(s2.heat_demand_kw,
                                  np.broadcast_to(s2.heat_demand_kw[0], s2.heat_demand_kw.shape))
    s1 = generate_scenario(config, "S1", 42)
    nominal = np.sum(s1.network.pump_cop*s1.network.pump_max_kw)
    expected = config["demand"]["mean_load_fraction"]*nominal
    np.testing.assert_allclose(s1.heat_demand_kw, expected, rtol=1e-14)
    config["demand"]["building_factor_half_width"] = 0.0
    no_factors = generate_scenario(config, "S2", 42)
    assert not np.allclose(s2.heat_demand_kw, no_factors.heat_demand_kw)
    np.testing.assert_allclose(s2.heat_demand_kw.sum(axis=1),
                               no_factors.heat_demand_kw.sum(axis=1), rtol=1e-14)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(schema_version=1),
    lambda c: c.update(schema_version=3),
    lambda c: c["simulation"].update(horizon=48),
    lambda c: c["weather"].update(default_week=11),
    lambda c: c["weather"].update(default_week=True),
    lambda c: c["weather"].update(provider="unknown"),
    lambda c: c["weather"].update(typo=1),
    lambda c: c["demand"].update(occupancy_transition_hours=-1),
    lambda c: c["demand"].update(occupancy_gain_gamma=float("nan")),
    lambda c: c["demand"].update(reference_degree_difference_c=0),
])
def test_schema_two_rejects_invalid_parameters(mutation):
    config = deepcopy(load_config(CONFIG))
    mutation(config)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("value", [True, -1, 11, 1.5])
def test_invalid_weather_week_is_rejected(value):
    with pytest.raises(ValueError, match="weather_week"):
        generate_scenario(load_config(CONFIG), "S1", 42, weather_week=value)
