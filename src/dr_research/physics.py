"""Independent physical replay and constraint checks; never repair an action.

Energy is measured above the tank reference temperature. Charge power is heat
before charge efficiency, while discharge/edge powers are useful delivered heat.
Heat loss uses the beginning-of-step mixed-tank temperature (explicit Euler).
"""
from __future__ import annotations

import math
import numpy as np

from .types import Action, Plan, Scenario, State, ValidationReport


def _array(value, shape, name, errors, kind="float"):
    try:
        if not isinstance(value, np.ndarray):
            errors.append(f"{name}: expected a NumPy array")
            return None
        arr = np.asarray(value)
        if arr.shape != shape:
            errors.append(f"{name}: shape {arr.shape}, expected {shape}")
            return None
        if kind == "int" and (not np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)):
            errors.append(f"{name}: expected integer indices")
            return None
        if kind == "bool" and arr.dtype != np.bool_:
            errors.append(f"{name}: expected boolean array")
            return None
        if kind == "float" and (not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr) or np.issubdtype(arr.dtype, np.bool_)):
            errors.append(f"{name}: expected real numeric array")
            return None
        if not np.all(np.isfinite(arr)):
            errors.append(f"{name}: contains nonfinite values")
            return None
        return arr
    except (TypeError, ValueError):
        errors.append(f"{name}: invalid array")
        return None


def validate_scenario(s: Scenario) -> None:
    """Check shape, finiteness, units-compatible ranges and station topology."""
    errors = []
    n = s.network
    try:
        h, nt, b, e, horizon = n.n_pumps, n.n_tanks, n.n_buildings, n.n_edges, s.horizon
    except (TypeError, AttributeError, IndexError) as exc:
        raise ValueError("scenario: invalid array dimensions") from exc
    if min(h, nt, b, horizon) <= 0 or e < 0:
        raise ValueError("scenario: empty dimensions are not supported")
    index_specs = {"pump_to_tank": (h,), "delivery_edges": (2, e), "pump_station": (h,), "tank_station": (nt,), "building_station": (b,)}
    for name, shape in index_specs.items():
        _array(getattr(n, name), shape, f"network.{name}", errors, "int")
    for name in ("pump_cop", "pump_max_kw", "pump_min_kw", "pump_ramp_kw"):
        _array(getattr(n, name), (h,), f"network.{name}", errors)
    for name in ("tank_capacity_kwh", "tank_charge_limit_kw", "tank_discharge_limit_kw", "tank_ua_kw_per_k", "tank_heat_capacity_kwh_per_k", "eta_charge", "eta_discharge"):
        _array(getattr(n, name), (nt,), f"network.{name}", errors)
    _array(n.edge_capacity_kw, (e,), "network.edge_capacity_kw", errors)
    if errors:
        raise ValueError("; ".join(errors))
    if np.any(n.pump_to_tank < 0) or np.any(n.pump_to_tank >= nt):
        errors.append("pump_to_tank: out-of-range index")
    if np.any(n.delivery_edges < 0) or np.any(n.delivery_edges[0] >= nt) or np.any(n.delivery_edges[1] >= b):
        errors.append("delivery_edges: out-of-range index")
    if any(np.any(getattr(n, k) < 0) for k in ("pump_station", "tank_station", "building_station")):
        errors.append("station indices must be nonnegative")
    if errors:
        raise ValueError("; ".join(errors))
    stations = n.n_stations
    if not np.array_equal(np.unique(n.building_station), np.arange(stations)):
        errors.append("building_station: station indices must be contiguous")
    if np.any(n.pump_station >= stations) or np.any(n.tank_station >= stations):
        errors.append("station index exceeds the building station range")
    if not np.array_equal(n.pump_station, n.tank_station[n.pump_to_tank]):
        errors.append("pump heat edge crosses a station")
    if np.any(n.tank_station[n.delivery_edges[0]] != n.building_station[n.delivery_edges[1]]):
        errors.append("delivery heat edge crosses a station")
    if len(np.unique(n.delivery_edges, axis=1).T) != e:
        errors.append("delivery_edges contains duplicate edges")
    for name in ("pump_max_kw", "tank_capacity_kwh", "tank_heat_capacity_kwh_per_k"):
        if np.any(getattr(n, name) <= 0):
            errors.append(f"network.{name}: must be positive")
    for name in ("pump_ramp_kw", "tank_charge_limit_kw", "tank_discharge_limit_kw", "edge_capacity_kw"):
        if np.any(getattr(n, name) < 0):
            errors.append(f"network.{name}: must be nonnegative")
    if np.any(n.pump_cop < 1):
        errors.append("pump_cop must be at least one")
    if np.any(n.pump_min_kw < 0) or np.any(n.pump_min_kw > n.pump_max_kw):
        errors.append("pump_min_kw must lie between zero and pump_max_kw")
    if np.any(n.tank_ua_kw_per_k < 0):
        errors.append("tank_ua_kw_per_k must be nonnegative")
    if any(np.any((getattr(n, k) <= 0) | (getattr(n, k) > 1)) for k in ("eta_charge", "eta_discharge")):
        errors.append("tank efficiencies must lie in (0,1]")
    if not isinstance(n.tank_reference_c, (int, float)) or not math.isfinite(n.tank_reference_c):
        errors.append("tank_reference_c must be finite")
    if n.pump_models and (len(n.pump_models) != h or any(not isinstance(v, str) or not v for v in n.pump_models)):
        errors.append("pump_models must contain one name per pump")
    if isinstance(s.dt_hours, bool) or not isinstance(s.dt_hours, (int, float)) or not math.isfinite(s.dt_hours) or s.dt_hours <= 0:
        errors.append("dt_hours must be finite and positive")
    shapes = {"price_yuan_per_kwh": (horizon,), "heat_demand_kw": (horizon, b), "ambient_c": (horizon,), "tank_environment_c": (horizon, nt), "source_limit_kw": (horizon, stations), "power_lower_kw": (horizon,), "power_upper_kw": (horizon,), "initial_energy_kwh": (nt,), "initial_power_kw": (h,), "terminal_energy_kwh": (nt,)}
    for name, shape in shapes.items():
        _array(getattr(s, name), shape, name, errors)
    _array(s.availability, (horizon, h), "availability", errors, "bool")
    if errors:
        raise ValueError("; ".join(errors))
    for name in ("price_yuan_per_kwh", "heat_demand_kw", "source_limit_kw", "power_lower_kw", "power_upper_kw"):
        if np.any(getattr(s, name) < 0):
            errors.append(f"{name}: must be nonnegative")
    if np.any(s.power_lower_kw > s.power_upper_kw):
        errors.append("power service band lower bound exceeds upper bound")
    for name in ("initial_energy_kwh", "terminal_energy_kwh"):
        if np.any(getattr(s, name) < 0) or np.any(getattr(s, name) > n.tank_capacity_kwh):
            errors.append(f"{name}: outside the usable tank range")
    if np.any(s.initial_power_kw < 0) or np.any(s.initial_power_kw > n.pump_max_kw):
        errors.append("initial_power_kw outside physical capacity")
    if np.any(s.dt_hours*n.tank_ua_kw_per_k >= n.tank_heat_capacity_kwh_per_k):
        errors.append("time step is too large for the explicit heat-loss model")
    if errors:
        raise ValueError("; ".join(errors))


def _check_time(t, horizon, terminal=False):
    return not isinstance(t, (bool, np.bool_)) and isinstance(t, (int, np.integer)) and 0 <= t <= horizon-(not terminal)


def _state_errors(s, state, terminal=False):
    errors = []
    if not _check_time(state.t, s.horizon, terminal):
        errors.append(f"state.t: expected an integer in [0,{s.horizon if terminal else s.horizon-1}]")
    _array(state.energy_kwh, (s.network.n_tanks,), "state.energy_kwh", errors)
    _array(state.previous_power_kw, (s.network.n_pumps,), "state.previous_power_kw", errors)
    return errors


def _action_errors(s, action):
    errors = []
    _array(action.power_kw, (s.network.n_pumps,), "action.power_kw", errors)
    _array(action.delivery_kw, (s.network.n_edges,), "action.delivery_kw", errors)
    return errors


def _next_energy(s, state, action):
    n = s.network
    charge = np.bincount(n.pump_to_tank, weights=n.pump_cop*action.power_kw, minlength=n.n_tanks)
    delivered = np.bincount(n.delivery_edges[0], weights=action.delivery_kw, minlength=n.n_tanks)
    return _next_energy_from_flows(s, state, charge, delivered)


def _next_energy_from_flows(s, state, charge, delivered):
    """Apply the independent tank equations to already aggregated heat flows."""
    n = s.network
    temperature = n.tank_reference_c + state.energy_kwh/n.tank_heat_capacity_kwh_per_k
    loss = n.tank_ua_kw_per_k*(temperature-s.tank_environment_c[state.t])
    return state.energy_kwh+s.dt_hours*(n.eta_charge*charge-delivered/n.eta_discharge-loss)


def transition(s: Scenario, state: State, action: Action) -> State:
    """Apply physics exactly once; infeasible powers/energies are never clipped."""
    errors = _state_errors(s, state)+_action_errors(s, action)
    if errors:
        raise ValueError("; ".join(errors))
    energy = _next_energy(s, state, action)
    if not np.all(np.isfinite(energy)):
        raise ValueError("transition produced nonfinite energy")
    return State(int(state.t)+1, energy, np.asarray(action.power_kw, dtype=float).copy())


def _atol(atol):
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")


def _residual(residuals, name, value):
    arr = np.asarray(value)
    result = max(0.0, float(np.max(arr))) if arr.size else 0.0
    residuals[name] = max(residuals.get(name, 0.0), result)


def _report(errors, residuals, atol, cost=None):
    violations = tuple(errors)+tuple(f"{name}: residual {value:.9g} exceeds {atol:g}" for name, value in residuals.items() if value > atol)
    return ValidationReport(not violations, violations, residuals, cost)


def validate_action(s: Scenario, state: State, action: Action, atol: float = 1e-5) -> ValidationReport:
    """Check one physical action, including exact per-building heat balance."""
    _atol(atol)
    errors = _state_errors(s, state)+_action_errors(s, action)
    if errors:
        return _report(errors, {}, atol)
    residuals = {}
    errors, _, cost = _check_action_physics(s, state, action, residuals)
    return _report(errors, residuals, atol, cost)


def _check_action_physics(s, state, action, residuals):
    """Accumulate physical residuals and return the independently replayed state.

    The caller validates array shapes/finiteness and the time boundary. A plan
    does that once for its complete arrays; each recursively computed energy
    is checked here before the next step. No solver energy array is trusted.
    """
    n, t = s.network, state.t
    p, d, errors = action.power_kw, action.delivery_kw, []
    _residual(residuals, "state_energy_lower", -state.energy_kwh)
    _residual(residuals, "state_energy_upper", state.energy_kwh-n.tank_capacity_kwh)
    _residual(residuals, "previous_power_lower", -state.previous_power_kw)
    _residual(residuals, "previous_power_upper", state.previous_power_kw-n.pump_max_kw)
    _residual(residuals, "pump_lower", n.pump_min_kw*s.availability[t]-p)
    _residual(residuals, "pump_upper", p-n.pump_max_kw*s.availability[t])
    _residual(residuals, "ramp", np.abs(p-state.previous_power_kw)-n.pump_ramp_kw)
    _residual(residuals, "edge_lower", -d)
    _residual(residuals, "edge_upper", d-n.edge_capacity_kw)
    total_power = p.sum()
    _residual(residuals, "power_band_lower", s.power_lower_kw[t]-total_power)
    _residual(residuals, "power_band_upper", total_power-s.power_upper_kw[t])
    source = np.bincount(n.pump_station, weights=(n.pump_cop-1)*p, minlength=n.n_stations)
    _residual(residuals, "source", source-s.source_limit_kw[t])
    charge = np.bincount(n.pump_to_tank, weights=n.pump_cop*p, minlength=n.n_tanks)
    discharge = np.bincount(n.delivery_edges[0], weights=d, minlength=n.n_tanks)
    delivered = np.bincount(n.delivery_edges[1], weights=d, minlength=n.n_buildings)
    _residual(residuals, "charge_capacity", charge-n.tank_charge_limit_kw)
    _residual(residuals, "discharge_capacity", discharge-n.tank_discharge_limit_kw)
    _residual(residuals, "building_heat_balance", np.abs(delivered-s.heat_demand_kw[t]))
    next_energy = _next_energy_from_flows(s, state, charge, discharge)
    if not np.all(np.isfinite(next_energy)):
        return ["transition produced nonfinite energy"], next_energy, None
    _residual(residuals, "energy_lower", -next_energy)
    _residual(residuals, "energy_upper", next_energy-n.tank_capacity_kwh)
    if t == s.horizon-1:
        _residual(residuals, "terminal_energy", np.abs(next_energy-s.terminal_energy_kwh))
    cost = float(s.dt_hours*s.price_yuan_per_kwh[t]*total_power)
    if not math.isfinite(cost):
        errors.append("electricity cost is nonfinite")
    return errors, next_energy, cost


def validate_plan(s: Scenario, plan: Plan, state: State | None = None, atol: float = 1e-5) -> ValidationReport:
    """Replay an entire remaining horizon independently of solver state arrays."""
    _atol(atol)
    state = s.initial_state() if state is None else state
    errors = _state_errors(s, state, terminal=True)
    if not _check_time(plan.start_t, s.horizon, terminal=True):
        errors.append("plan.start_t: invalid horizon boundary")
    elif plan.start_t != state.t:
        errors.append("plan.start_t does not match state.t")
    if errors:
        return _report(errors, {}, atol)
    steps, n = s.horizon-state.t, s.network
    _array(plan.power_kw, (steps, n.n_pumps), "plan.power_kw", errors)
    _array(plan.delivery_kw, (steps, n.n_edges), "plan.delivery_kw", errors)
    _array(plan.energy_kwh, (steps+1, n.n_tanks), "plan.energy_kwh", errors)
    if errors:
        return _report(errors, {}, atol)
    residuals, total_cost = {}, 0.0
    _residual(residuals, "trajectory_initial", np.abs(plan.energy_kwh[0]-state.energy_kwh))
    _residual(residuals, "state_energy_lower", -state.energy_kwh)
    _residual(residuals, "state_energy_upper", state.energy_kwh-n.tank_capacity_kwh)
    _residual(residuals, "previous_power_lower", -state.previous_power_kw)
    _residual(residuals, "previous_power_upper", state.previous_power_kw-n.pump_max_kw)
    current = State(state.t, state.energy_kwh.copy(), state.previous_power_kw.copy())
    for offset in range(steps):
        # Views are safe: replay reads the plan and never repairs its actions.
        # Array contracts were checked above for the complete plan.
        action = Action(plan.power_kw[offset], plan.delivery_kw[offset])
        step_errors, next_energy, cost = _check_action_physics(s, current, action, residuals)
        errors.extend(f"step {current.t}: {message}" for message in step_errors)
        if cost is None or not math.isfinite(cost):
            errors.append(f"step {current.t}: invalid cost or transition")
            break
        total_cost += cost
        current = State(int(current.t)+1, next_energy, action.power_kw)
        _residual(residuals, "trajectory_dynamics", np.abs(plan.energy_kwh[offset+1]-current.energy_kwh))
    _residual(residuals, "terminal_energy", np.abs(current.energy_kwh-s.terminal_energy_kwh))
    if not math.isfinite(total_cost):
        errors.append("total electricity cost is nonfinite")
    return _report(errors, residuals, atol, total_cost)
