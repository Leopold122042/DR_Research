"""Small test fixtures derived from the supported weekly physical scenario."""
from dataclasses import replace
import numpy as np

from dr_research.physics import validate_scenario, validate_plan
from dr_research.types import Scenario, Network, Plan, State


def scenario_prefix(scenario: Scenario, steps: int = 48) -> Scenario:
    """Use a short prefix for solver tests, preserving physical end conditions.

    This is a test-only fixture. Published experiments retain all 336 steps.
    Every time-dependent field is sliced together; the original initial state
    and terminal inventory remain the optimization boundary conditions.
    """
    if type(steps) is not int or not 1 <= steps <= scenario.horizon:
        raise ValueError("steps must select a nonempty scenario prefix")
    time_fields = ("price_yuan_per_kwh", "heat_demand_kw", "ambient_c",
                   "tank_environment_c", "source_limit_kw", "power_lower_kw",
                   "power_upper_kw", "availability")
    result = replace(scenario,
                     **{name: getattr(scenario, name)[:steps].copy() for name in time_fields},
                     scenario_id=f"{scenario.scenario_id}-test-prefix{steps}")
    validate_scenario(result)
    return result


def small_case(horizon=8):
    """Two tanks and two buildings with a constant-temperature certificate."""
    net = Network(
        pump_to_tank=np.array([0, 0, 1, 1]),
        delivery_edges=np.array([[0, 0, 1, 1], [0, 1, 0, 1]]),
        pump_station=np.zeros(4, dtype=int), tank_station=np.zeros(2, dtype=int),
        building_station=np.zeros(2, dtype=int), pump_cop=np.array([2., 2.5, 2., 2.5]),
        pump_max_kw=np.full(4, 10.), pump_min_kw=np.zeros(4), pump_ramp_kw=np.full(4, 4.),
        tank_capacity_kwh=np.full(2, 20.), tank_charge_limit_kw=np.full(2, 30.),
        tank_discharge_limit_kw=np.full(2, 20.), tank_ua_kw_per_k=np.full(2, .04),
        tank_heat_capacity_kwh_per_k=np.ones(2), tank_reference_c=70.,
        eta_charge=np.ones(2), eta_discharge=np.ones(2), edge_capacity_kw=np.full(4, 12.))
    power = np.full(4, 8.4/4.5)
    scenario = Scenario(net, .5, np.linspace(.3, 1.2, horizon), np.full((horizon, 2), 8.),
                 np.full(horizon, 5.), np.full((horizon, 2), 70.), np.full((horizon, 1), 30.),
                 np.full(horizon, 2.), np.full(horizon, 30.), np.ones((horizon, 4), dtype=bool),
                 np.full(2, 10.), power.copy(), np.full(2, 10.), "safety-fixture", 42, "fixture")
    anchor = Plan(0, np.tile(power, (horizon, 1)), np.full((horizon, 4), 4.), np.full((horizon+1, 2), 10.))
    assert validate_plan(scenario, anchor).feasible
    return scenario, anchor


def suffix(scenario, anchor, t):
    state = State(t, anchor.energy_kwh[t].copy(),
                  anchor.power_kw[t-1].copy() if t else scenario.initial_power_kw.copy())
    return state, Plan(t, anchor.power_kw[t:], anchor.delivery_kw[t:], anchor.energy_kwh[t:])
