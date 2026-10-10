"""Equivalent-pump presolve preserves the original rolling LP problem."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dr_research.config import load_config
from dr_research.control import construct_reference, solve_control_window, _window
from dr_research.physics import validate_plan, transition
from dr_research.scenario import generate_scenario
from dr_research.types import Action, State

ROOT = Path(__file__).resolve().parents[1]


def case(seed=42):
    return generate_scenario(load_config(ROOT/"configs/experiment.toml"), "S2", seed, weather_week=8)


@pytest.mark.parametrize("seed", [42, 104])
def test_grouped_lp_matches_unaggregated_objective_and_original_physics(seed):
    s = case(seed)
    reference, state = construct_reference(s), s.initial_state()
    for _ in range(3):
        original = solve_control_window(s, state, reference, window_steps=12)
        grouped = solve_control_window(s, state, reference, window_steps=12, aggregate_pumps=True)
        assert original.status == grouped.status == "optimal", grouped.message
        assert grouped.electricity_yuan == pytest.approx(original.electricity_yuan, rel=1e-8, abs=1e-5)
        assert grouped.statistics["pump_aggregation_applied"]
        assert grouped.statistics["solver_pump_count"] < s.network.n_pumps
        assert grouped.statistics["physical_pump_count"] == s.network.n_pumps
        assert grouped.statistics["aggregation_max_roundoff_kw"] <= 1e-9
        local, end = _window(s, state, reference, 12)
        audit = validate_plan(local, replace(grouped.plan, start_t=0),
            state=State(0, state.energy_kwh, state.previous_power_kw))
        assert audit.feasible, audit.violations
        np.testing.assert_array_equal(grouped.plan.power_kw[-1], reference.power_kw[end-1])
        assert grouped.timings_s["assembly"] >= grouped.statistics["aggregation_time_s"]
        state = transition(s, state, grouped.plan.action(0))


def test_unequal_fixed_first_utilizations_split_groups_and_remain_exact():
    s = case()
    n = s.network
    reference = construct_reference(s)
    power = reference.power_kw[0].copy()
    pair = next((i, j) for i in range(n.n_pumps) for j in range(i+1, n.n_pumps)
                if n.pump_to_tank[i] == n.pump_to_tank[j] and n.pump_cop[i] == n.pump_cop[j])
    i, j = pair
    perturbation = .03*min(power[i], power[j])
    power[i] += perturbation
    power[j] -= perturbation
    first = Action(power, reference.delivery_kw[0].copy())
    original = solve_control_window(s, s.initial_state(), reference, window_steps=12, fixed_first_action=first)
    grouped = solve_control_window(s, s.initial_state(), reference, window_steps=12,
        fixed_first_action=first, aggregate_pumps=True)
    assert original.status == grouped.status == "optimal", grouped.message
    assert grouped.electricity_yuan == pytest.approx(original.electricity_yuan, rel=1e-8, abs=1e-5)
    np.testing.assert_array_equal(grouped.plan.power_kw[0], first.power_kw)
    np.testing.assert_array_equal(grouped.plan.delivery_kw[0], first.delivery_kw)


def test_no_grouping_when_every_cop_is_distinct():
    s = case()
    n = replace(s.network, pump_cop=s.network.pump_cop+np.arange(s.network.n_pumps)*1e-5)
    s = replace(s, network=n)
    reference = construct_reference(s)
    result = solve_control_window(s, s.initial_state(), reference, window_steps=12, aggregate_pumps=True)
    assert result.status == "optimal", result.message
    assert not result.statistics["pump_aggregation_applied"]
    assert result.statistics["solver_pump_count"] == s.network.n_pumps


@pytest.mark.parametrize("invalid", [True, 0, 65, 1.5])
def test_invalid_solver_thread_settings_rejected(invalid):
    s = case()
    result = solve_control_window(s, s.initial_state(), construct_reference(s), solver_threads=invalid)
    assert result.status == "error" and "solver_threads" in result.message
