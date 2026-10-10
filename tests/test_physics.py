from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dr_research.config import load_config
from dr_research.physics import transition, validate_action, validate_plan
from dr_research.scenario import generate_scenario
from dr_research.types import Action, Plan, State

CONFIG = Path(__file__).resolve().parents[1]/"configs"/"experiment.toml"


def constant_temperature_plan(s):
    """A hand-constructed witness: each backbone supplies its own demand,
    and pump heat replaces delivered heat plus losses at constant 80 C.
    This witness uses no optimizer or production transition routine.
    """
    n = s.network
    qmax = n.pump_max_kw*n.pump_cop
    tank_max = np.bincount(n.pump_to_tank, weights=qmax, minlength=n.n_tanks)
    tank_owner = np.arange(n.n_tanks)//2
    building_max = np.bincount(tank_owner, weights=tank_max, minlength=n.n_buildings)
    delivery = np.zeros((s.horizon, n.n_edges))
    tank_demand = s.heat_demand_kw[:, tank_owner]*tank_max[None, :]/building_max[tank_owner][None, :]
    for edge, (tank, building) in enumerate(n.delivery_edges.T):
        if tank_owner[tank] == building:
            delivery[:, edge] = tank_demand[:, tank]
    temperature = n.tank_reference_c+s.initial_energy_kwh/n.tank_heat_capacity_kwh_per_k
    heat_loss = (temperature[None, :]-s.tank_environment_c)*n.tank_ua_kw_per_k
    heat_needed = (tank_demand/n.eta_discharge+heat_loss)/n.eta_charge
    fractions = heat_needed/tank_max
    power = fractions[:, n.pump_to_tank]*n.pump_max_kw
    energies = np.repeat(s.initial_energy_kwh[None, :], s.horizon+1, axis=0)
    return Plan(0, power, delivery, energies)


def test_constructive_feasible_witness_and_cost_replay():
    s = generate_scenario(load_config(CONFIG), "S2", 28)
    plan = constant_temperature_plan(s)
    report = validate_plan(s, plan)
    assert report.feasible, report.violations
    expected = np.dot(s.price_yuan_per_kwh, plan.power_kw.sum(axis=1))*s.dt_hours
    assert report.electricity_yuan == pytest.approx(expected)
    next_state = transition(s, s.initial_state(), plan.action(0))
    np.testing.assert_allclose(next_state.energy_kwh, s.initial_energy_kwh, atol=1e-10)
    assert next_state.t == 1


def test_independent_replay_rejects_forged_inventory_and_building_transfer():
    s = generate_scenario(load_config(CONFIG), "S2", 12)
    plan = constant_temperature_plan(s)
    forged = plan.energy_kwh.copy()
    forged[10, 0] += 5
    report = validate_plan(s, replace(plan, energy_kwh=forged))
    assert not report.feasible and report.max_residuals["trajectory_dynamics"] >= 5-1e-8
    delivery = plan.delivery_kw.copy()
    e0 = int(np.flatnonzero(s.network.delivery_edges[1] == 0)[0])
    e1 = int(np.flatnonzero(s.network.delivery_edges[1] == 1)[0])
    delivery[0, e0] += 1
    delivery[0, e1] -= 1
    assert delivery[0].sum() == pytest.approx(plan.delivery_kw[0].sum())
    report = validate_plan(s, replace(plan, delivery_kw=delivery))
    assert not report.feasible and report.max_residuals["building_heat_balance"] == pytest.approx(1)


def test_no_clipping_and_operating_constraints():
    s = generate_scenario(load_config(CONFIG), "S1", 3)
    n = s.network
    action = Action(np.zeros(n.n_pumps), np.full(n.n_edges, 1e5))
    after = transition(s, s.initial_state(), action)
    assert np.all(after.energy_kwh < 0)  # pure transition exposes violations.
    report = validate_action(s, s.initial_state(), action)
    assert not report.feasible
    assert report.max_residuals["pump_lower"] > 0
    assert report.max_residuals["energy_lower"] > 0
    assert report.max_residuals["discharge_capacity"] > 0
    plan = constant_temperature_plan(s)
    full = Action(n.pump_max_kw.copy(), plan.delivery_kw[0].copy())
    report = validate_action(s, State(0, s.initial_energy_kwh.copy(), np.zeros(n.n_pumps)), full)
    assert report.max_residuals["ramp"] > 0
    assert report.max_residuals["source"] > 0
    assert report.max_residuals["power_band_upper"] > 0


def test_nonfinite_shape_and_time_boundaries():
    s = generate_scenario(load_config(CONFIG), "S1", 2)
    plan = constant_temperature_plan(s)
    bad_power = plan.power_kw.copy()
    bad_power[0, 0] = np.nan
    assert not validate_plan(s, replace(plan, power_kw=bad_power)).feasible
    assert not validate_plan(s, replace(plan, delivery_kw=plan.delivery_kw[:-1])).feasible
    assert not validate_plan(s, replace(plan, start_t=1)).feasible
    for invalid_t in (-1, s.horizon, 0.5, True):
        state = replace(s.initial_state(), t=invalid_t)
        assert not validate_action(s, state, plan.action(0)).feasible
        with pytest.raises(ValueError):
            transition(s, state, plan.action(0))
    bad = replace(plan.action(0), delivery_kw=np.zeros((s.network.n_edges, 1)))
    assert not validate_action(s, s.initial_state(), bad).feasible
    with pytest.raises(ValueError):
        validate_plan(s, plan, atol=-1)


def test_suffix_and_empty_remaining_horizon():
    s = generate_scenario(load_config(CONFIG), "S2", 25)
    plan = constant_temperature_plan(s)
    offset = 17
    state = State(offset, plan.energy_kwh[offset].copy(), plan.power_kw[offset-1].copy())
    suffix = Plan(offset, plan.power_kw[offset:].copy(), plan.delivery_kw[offset:].copy(), plan.energy_kwh[offset:].copy())
    assert validate_plan(s, suffix, state).feasible
    final = State(s.horizon, s.terminal_energy_kwh.copy(), plan.power_kw[-1].copy())
    empty = Plan(s.horizon, np.empty((0, s.network.n_pumps)), np.empty((0, s.network.n_edges)), s.terminal_energy_kwh[None, :].copy())
    report = validate_plan(s, empty, final)
    assert report.feasible and report.electricity_yuan == 0
    incorrect_final = replace(final, energy_kwh=final.energy_kwh-1)
    assert not validate_plan(s, replace(empty, energy_kwh=incorrect_final.energy_kwh[None, :]), incorrect_final).feasible
