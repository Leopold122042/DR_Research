"""Control tests check physical continuation and optimization meaning."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from dr_research.config import load_config
from dr_research.control import construct_reference, solve_control_window, stitch_reference_tail
from dr_research.optimization import solve_dispatch
from dr_research.physics import transition, validate_action, validate_plan
from dr_research.scenario import generate_scenario
from dr_research.types import Plan, State, ValidationReport
from helpers import scenario_prefix

CONFIG = Path(__file__).resolve().parents[1]/"configs"/"experiment.toml"


def case(scale="S1", seed=17, steps=48):
    return scenario_prefix(generate_scenario(load_config(CONFIG), scale, seed), steps)


def test_reference_is_constructive_and_feasible_at_maximum_scale():
    s = case("S5", steps=336)
    with patch("dr_research.control.linprog") as solver, patch("dr_research.optimization.linprog") as economic:
        ref = construct_reference(s)
    solver.assert_not_called()
    economic.assert_not_called()
    assert validate_plan(s, ref).feasible
    np.testing.assert_allclose(ref.energy_kwh, np.broadcast_to(s.initial_energy_kwh, ref.energy_kwh.shape))


def test_reference_rejects_unsupported_boundary_topology_and_infeasibility():
    s = case()
    with pytest.raises(ValueError, match="equal initial and terminal"):
        construct_reference(replace(s, terminal_energy_kwh=s.terminal_energy_kwh+1))
    missing = replace(s.network, delivery_edges=s.network.delivery_edges[:, :1],
                      edge_capacity_kw=s.network.edge_capacity_kw[:1])
    with pytest.raises(ValueError, match="backbone delivery"):
        construct_reference(replace(s, network=missing))
    with pytest.raises(ValueError, match="infeasible"):
        construct_reference(replace(s, heat_demand_kw=3*s.heat_demand_kw))


def test_full_remaining_mpc_matches_lp_on_identical_test_horizon():
    s = case()
    ref = construct_reference(s)
    full = solve_dispatch(s)
    mpc = solve_control_window(s, s.initial_state(), ref, window_steps=s.horizon)
    assert full.status == mpc.status == "optimal", mpc.message
    assert mpc.electricity_yuan == pytest.approx(full.electricity_yuan, abs=1e-6)
    assert validate_plan(s, mpc.plan).feasible
    assert not mpc.statistics["terminal_power_pinned"]
    assert mpc.statistics["lower_bound_scope"] == "window_only"
    assert set(mpc.timings_s) == {"assembly", "solve", "validation", "total"}


def test_twelve_step_closed_loop_and_reference_tail_are_physically_feasible():
    s = case()
    ref = construct_reference(s)
    state = s.initial_state()
    powers, deliveries, energy = [], [], [state.energy_kwh.copy()]
    for t in range(s.horizon):
        solved = solve_control_window(s, state, ref, window_steps=12)
        assert solved.status == "optimal", (t, solved.message)
        assert solved.plan.start_t == t
        assert solved.plan.power_kw.shape[0] == min(12, s.horizon-t)
        cache = stitch_reference_tail(s, state, solved.plan, ref)
        report = validate_plan(s, cache, state)
        assert report.feasible, (t, report.violations)
        if t+12 < s.horizon:
            np.testing.assert_allclose(solved.plan.power_kw[-1], ref.power_kw[t+11], atol=1e-7)
        action = solved.plan.action(0)
        assert validate_action(s, state, action).feasible
        powers.append(action.power_kw)
        deliveries.append(action.delivery_kw)
        state = transition(s, state, action)
        energy.append(state.energy_kwh.copy())
    executed = Plan(0, np.asarray(powers), np.asarray(deliveries), np.asarray(energy))
    report = validate_plan(s, executed)
    assert report.feasible, report.violations
    np.testing.assert_allclose(state.energy_kwh, s.terminal_energy_kwh, atol=1e-5)
    assert report.electricity_yuan >= solve_dispatch(s).electricity_yuan-1e-6








@pytest.mark.parametrize("kwargs", [{"window_steps": 1}, {"window_steps": True},
                                    {"window_steps": 2.5}, {"time_limit_s": 0},
                                    {"time_limit_s": np.nan}, {"solver_method": "invalid"}])
def test_invalid_control_parameters_rejected_before_solver(kwargs):
    s = case()
    ref = construct_reference(s)
    with patch("dr_research.control.linprog") as solver:
        solved = solve_control_window(s, s.initial_state(), ref, **kwargs)
    solver.assert_not_called()
    assert solved.status == "error"
    assert solved.plan is None


@pytest.mark.parametrize("raw, expected", [(1, "limit"), (2, "infeasible"), (3, "unbounded"), (4, "error")])
def test_failed_window_never_returns_fabricated_cost_or_plan(raw, expected):
    s = case()
    ref = construct_reference(s)
    fake = SimpleNamespace(status=raw, success=False, x=None, nit=0, message="controlled failure")
    with patch("dr_research.control.linprog", return_value=fake):
        solved = solve_control_window(s, s.initial_state(), ref)
    assert solved.status == expected
    assert solved.plan is None and solved.electricity_yuan is None and solved.lower_bound_yuan is None


def test_independent_validation_failure_cannot_be_optimal():
    s = case()
    ref = construct_reference(s)
    rejected = ValidationReport(False, ("test rejection",), {"test": 1.0})
    with patch("dr_research.control.validate_plan", return_value=rejected):
        solved = solve_control_window(s, s.initial_state(), ref)
    assert solved.status == "error"
    assert solved.plan is None and solved.electricity_yuan is None
    assert "test rejection" in solved.message


def test_tail_stitching_checks_state_inventory_and_power_boundaries():
    s = case()
    ref = construct_reference(s)
    state = s.initial_state()
    solved = solve_control_window(s, state, ref, window_steps=12)
    assert solved.status == "optimal", solved.message
    energy = solved.plan.energy_kwh.copy()
    energy[-1, 0] += 1
    with pytest.raises(ValueError, match="terminal inventory"):
        stitch_reference_tail(s, state, replace(solved.plan, energy_kwh=energy), ref)
    power = solved.plan.power_kw.copy()
    power[-1, 0] += 1
    with pytest.raises(ValueError, match="terminal power"):
        stitch_reference_tail(s, state, replace(solved.plan, power_kw=power), ref)
    with pytest.raises(ValueError, match="window start"):
        stitch_reference_tail(s, state, replace(solved.plan, start_t=1), ref)
    wrong = State(0, state.energy_kwh+1, state.previous_power_kw)
    with pytest.raises(ValueError, match="initial energy"):
        stitch_reference_tail(s, wrong, solved.plan, ref)


def test_reference_power_boundary_conflict_is_explicit_infeasibility():
    s = case()
    ref = construct_reference(s)
    forged = ref.power_kw.copy()
    forged[11] = s.network.pump_max_kw+1
    with patch("dr_research.control.linprog") as solver:
        solved = solve_control_window(s, s.initial_state(), replace(ref, power_kw=forged), window_steps=12)
    solver.assert_not_called()
    assert solved.status == "infeasible"
    assert solved.statistics["inconsistent_bound_count"] == s.network.n_pumps


def test_future_window_uses_absolute_time_and_last_interval_can_shorten_to_one():
    s = case()
    ref = construct_reference(s)
    final_t = s.horizon-1
    state = State(final_t, ref.energy_kwh[final_t].copy(), ref.power_kw[final_t-1].copy())
    solved = solve_control_window(s, state, ref, window_steps=12)
    assert solved.status == "optimal", solved.message
    assert solved.plan.start_t == final_t and solved.plan.power_kw.shape[0] == 1
    assert solved.electricity_yuan == pytest.approx(s.dt_hours*s.price_yuan_per_kwh[final_t]*solved.plan.power_kw.sum())
    full = stitch_reference_tail(s, state, solved.plan, ref)
    assert validate_plan(s, full, state).feasible
    ended = State(s.horizon, s.terminal_energy_kwh.copy(), solved.plan.power_kw[-1].copy())
    assert solve_control_window(s, ended, ref).status == "error"


def test_roundoff_in_actual_state_is_not_clipped_or_treated_as_configuration():
    s = case()
    ref = construct_reference(s)
    state = State(7, s.initial_energy_kwh.copy(), s.network.pump_max_kw+1e-10)
    before_power = state.previous_power_kw.copy()
    # Physical replay accepts roundoff in a measured successor. Keep the exact
    # state in ramp constraints; Scenario initialization remains valid input.
    from dr_research.optimization import build_linear_program
    with patch("dr_research.control.build_linear_program", wraps=build_linear_program) as builder:
        solved = solve_control_window(s, state, ref)
    assert solved.status == "optimal", solved.message
    passed_scenario = builder.call_args.args[0]
    passed_state = builder.call_args.kwargs["state"]
    np.testing.assert_array_equal(passed_scenario.initial_power_kw, s.initial_power_kw)
    np.testing.assert_array_equal(passed_state.previous_power_kw, before_power)
    np.testing.assert_array_equal(state.previous_power_kw, before_power)
    assert passed_state.t == 0
    assert validate_plan(s, stitch_reference_tail(s, state, solved.plan, ref), state).feasible


def test_s2_seed100_teacher_rollout_survives_solver_roundoff():
    s = case("S2", seed=100)
    ref = construct_reference(s)
    state = s.initial_state()
    powers, deliveries, energy = [], [], [state.energy_kwh.copy()]
    for t in range(s.horizon):
        solved = solve_control_window(s, state, ref, window_steps=12)
        assert solved.status == "optimal", (t, solved.message)
        action = solved.plan.action(0)
        assert validate_action(s, state, action).feasible
        powers.append(action.power_kw)
        deliveries.append(action.delivery_kw)
        state = transition(s, state, action)
        energy.append(state.energy_kwh.copy())
    report = validate_plan(s, Plan(0, np.asarray(powers), np.asarray(deliveries), np.asarray(energy)))
    assert report.feasible, report.violations
