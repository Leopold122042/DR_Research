"""Replay equivalence against the public one-step checker and transition.

The reference deliberately constructs separate per-step reports and transitions;
the production loop aggregates residuals directly and reuses computed states.
"""
from dataclasses import replace
import math

import numpy as np
import pytest

from dr_research.physics import transition, validate_action, validate_plan
from dr_research.types import State, ValidationReport
from helpers import small_case


def _public_step_replay(s, plan, state=None, atol=1e-5):
    """Simple reference for plans whose input array contracts are valid."""
    current = s.initial_state() if state is None else state
    n = s.network
    maxima = {}

    def add(name, values):
        a = np.asarray(values)
        value = max(0., float(np.max(a))) if a.size else 0.
        maxima[name] = max(maxima.get(name, 0.), value)

    add("trajectory_initial", abs(plan.energy_kwh[0]-current.energy_kwh))
    add("state_energy_lower", -current.energy_kwh)
    add("state_energy_upper", current.energy_kwh-n.tank_capacity_kwh)
    add("previous_power_lower", -current.previous_power_kw)
    add("previous_power_upper", current.previous_power_kw-n.pump_max_kw)
    errors, cost = [], 0.
    for offset in range(s.horizon-current.t):
        action = plan.action(offset)
        checked = validate_action(s, current, action, atol)
        for name, value in checked.max_residuals.items():
            add(name, value)
        numeric = {f"{name}: residual {value:.9g} exceeds {atol:g}"
                   for name, value in checked.max_residuals.items() if value > atol}
        errors.extend(f"step {current.t}: {message}"
                      for message in checked.violations if message not in numeric)
        if checked.electricity_yuan is None or not math.isfinite(checked.electricity_yuan):
            errors.append(f"step {current.t}: invalid cost or transition")
            break
        cost += checked.electricity_yuan
        current = transition(s, current, action)
        add("trajectory_dynamics", abs(plan.energy_kwh[offset+1]-current.energy_kwh))
    add("terminal_energy", abs(current.energy_kwh-s.terminal_energy_kwh))
    if not math.isfinite(cost):
        errors.append("total electricity cost is nonfinite")
    violations = tuple(errors)+tuple(
        f"{name}: residual {value:.9g} exceeds {atol:g}"
        for name, value in maxima.items() if value > atol)
    return ValidationReport(not violations, violations, maxima, cost)


@pytest.mark.parametrize("atol", [0., 1e-5, .25])
def test_replay_preserves_every_residual_and_cost(atol):
    s, anchor = small_case(12)
    rng = np.random.default_rng(42)
    plans = [anchor]
    # Include feasible and infeasible states/actions, with forged stored
    # trajectories. A later physically legal action never masks an earlier
    # constraint violation or changes the replay's physical initial state.
    for _ in range(15):
        plans.append(replace(anchor,
            power_kw=anchor.power_kw+rng.normal(0., 3., anchor.power_kw.shape),
            delivery_kw=anchor.delivery_kw+rng.normal(0., 4., anchor.delivery_kw.shape),
            energy_kwh=anchor.energy_kwh+rng.normal(0., .5, anchor.energy_kwh.shape)))
    for plan in plans:
        actual = validate_plan(s, plan, atol=atol)
        expected = _public_step_replay(s, plan, atol=atol)
        assert actual == expected


def test_readonly_plan_and_suffix_preserve_inputs():
    s, plan = small_case(12)
    start = 5
    state = State(start, plan.energy_kwh[start].copy(), plan.power_kw[start-1].copy())
    suffix = replace(plan, start_t=start, power_kw=plan.power_kw[start:],
                     delivery_kw=plan.delivery_kw[start:], energy_kwh=plan.energy_kwh[start:])
    arrays = [suffix.power_kw, suffix.delivery_kw, suffix.energy_kwh,
              state.energy_kwh, state.previous_power_kw]
    snapshots = [a.copy() for a in arrays]
    for a in arrays:
        a.flags.writeable = False
    assert validate_plan(s, suffix, state) == _public_step_replay(s, suffix, state)
    for a, original in zip(arrays, snapshots):
        np.testing.assert_array_equal(a, original)


@pytest.mark.parametrize("failure", ["energy", "cost"])
def test_nonfinite_derived_quantities_keep_error_reporting(failure):
    s, plan = small_case(4)
    if failure == "energy":
        s = replace(s, network=replace(s.network, pump_cop=np.full(4, 1e308)))
    else:
        s = replace(s, price_yuan_per_kwh=np.full(s.horizon, 1e308))
    with np.errstate(over="ignore", invalid="ignore"):
        actual = validate_plan(s, plan)
        expected = _public_step_replay(s, plan)
    assert not actual.feasible
    assert actual == expected


def test_empty_terminal_state_matches_step_reference():
    s, plan = small_case(3)
    final = State(s.horizon, s.terminal_energy_kwh.copy(), plan.power_kw[-1].copy())
    empty = replace(plan, start_t=s.horizon, power_kw=plan.power_kw[:0],
                    delivery_kw=plan.delivery_kw[:0], energy_kwh=plan.energy_kwh[-1:])
    assert validate_plan(s, empty, final) == _public_step_replay(s, empty, final)
