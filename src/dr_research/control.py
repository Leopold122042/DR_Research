"""Finite-horizon cost MPC and offline fixed-action continuation labels.

A constructive constant-temperature reference provides a feasible tail without
using the economic optimum. Window endpoints rejoin this reference in both
inventory and previous pump power, so shifting the horizon preserves a feasible
continuation under the current deterministic model.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning

from .optimization import build_linear_program
from .physics import validate_action, validate_plan, validate_scenario
from .types import Action, Plan, Scenario, SolveResult, State

_TOL = 1e-5


def _numeric(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != shape or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array) or np.issubdtype(array.dtype, np.bool_)
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    return array.astype(float, copy=False)


def _reference_arrays(s: Scenario, reference: Plan) -> None:
    if (isinstance(reference.start_t, (bool, np.bool_))
            or not isinstance(reference.start_t, (int, np.integer))
            or reference.start_t != 0):
        raise ValueError("reference must start at the beginning of the episode")
    n = s.network
    _numeric(reference.power_kw, (s.horizon, n.n_pumps), "reference.power_kw")
    _numeric(reference.delivery_kw, (s.horizon, n.n_edges), "reference.delivery_kw")
    _numeric(reference.energy_kwh, (s.horizon+1, n.n_tanks), "reference.energy_kwh")
    if not np.allclose(reference.energy_kwh[0], s.initial_energy_kwh, atol=_TOL, rtol=0):
        raise ValueError("reference initial energy does not match the scenario")
    if not np.allclose(reference.energy_kwh[-1], s.terminal_energy_kwh, atol=_TOL, rtol=0):
        raise ValueError("reference terminal energy does not match the scenario")


def construct_reference(s: Scenario) -> Plan:
    """Construct and independently verify a constant-inventory full-episode plan.

    This formula supports the generated two-tanks-per-building backbone. Each
    tank supplies its owner's demand in proportion to its connected pump heat
    capacity. Pump loading replenishes that delivery and tank loss at the initial
    temperature. Unsupported topologies, differing initial/final inventories or
    infeasible demand/equipment settings raise ValueError; no optimizer is used.
    """
    validate_scenario(s)
    n = s.network
    if n.n_tanks != 2*n.n_buildings:
        raise ValueError("constructive reference requires two backbone tanks per building")
    if not np.allclose(s.initial_energy_kwh, s.terminal_energy_kwh, atol=1e-9, rtol=0):
        raise ValueError("constant-inventory reference requires equal initial and terminal energy")
    owner = np.arange(n.n_tanks)//2
    if not np.array_equal(n.tank_station, n.building_station[owner]):
        raise ValueError("constructive reference backbone would cross a station")
    backbone = {(int(tank), int(building)): edge
                for edge, (tank, building) in enumerate(n.delivery_edges.T)}
    if any((tank, int(owner[tank])) not in backbone for tank in range(n.n_tanks)):
        raise ValueError("constructive reference requires every tank's backbone delivery edge")
    qmax = n.pump_max_kw*n.pump_cop
    tank_max = np.bincount(n.pump_to_tank, weights=qmax, minlength=n.n_tanks)
    if np.any(tank_max <= 0):
        raise ValueError("constructive reference requires positive pump capacity at every tank")
    building_max = np.bincount(owner, weights=tank_max, minlength=n.n_buildings)
    tank_demand = s.heat_demand_kw[:, owner]*tank_max[None, :]/building_max[owner][None, :]
    delivery = np.zeros((s.horizon, n.n_edges))
    for tank in range(n.n_tanks):
        delivery[:, backbone[(tank, int(owner[tank]))]] = tank_demand[:, tank]
    temperature = n.tank_reference_c+s.initial_energy_kwh/n.tank_heat_capacity_kwh_per_k
    heat_loss = (temperature[None, :]-s.tank_environment_c)*n.tank_ua_kw_per_k
    heat_needed = (tank_demand/n.eta_discharge+heat_loss)/n.eta_charge
    power = (heat_needed/tank_max)[:, n.pump_to_tank]*n.pump_max_kw
    energy = np.repeat(s.initial_energy_kwh[None, :], s.horizon+1, axis=0)
    reference = Plan(0, power, delivery, energy)
    report = validate_plan(s, reference)
    if not report.feasible:
        raise ValueError("constructive reference is infeasible: "+"; ".join(report.violations))
    return reference


def _window(s: Scenario, state: State, reference: Plan, window_steps: int) -> tuple[Scenario, int]:
    if (isinstance(window_steps, (bool, np.bool_))
            or not isinstance(window_steps, (int, np.integer)) or window_steps < 2):
        raise ValueError("window_steps must be an integer of at least two")
    if (isinstance(state.t, (bool, np.bool_))
            or not isinstance(state.t, (int, np.integer)) or not 0 <= state.t < s.horizon):
        raise ValueError("state.t must be an active integer time step")
    _reference_arrays(s, reference)
    start, end = int(state.t), min(int(state.t)+int(window_steps), s.horizon)
    _numeric(state.energy_kwh, (s.network.n_tanks,), "state.energy_kwh")
    _numeric(state.previous_power_kw, (s.network.n_pumps,), "state.previous_power_kw")
    arrays = {name: getattr(s, name)[start:end].copy() for name in (
        "price_yuan_per_kwh", "heat_demand_kw", "ambient_c", "tank_environment_c",
        "source_limit_kw", "power_lower_kw", "power_upper_kw", "availability")}
    terminal = s.terminal_energy_kwh if end == s.horizon else reference.energy_kwh[end]
    # Keep the scenario's exactly bounded initialization as a valid template.
    # Actual observed state is passed separately to the LP and replay below:
    # solver-scale roundoff in a physical state must not be clipped or promoted
    # into configuration data, whose bounds intentionally use strict checks.
    local = replace(s, **arrays, terminal_energy_kwh=terminal.copy(),
                    scenario_id=f"{s.scenario_id}:window:{start}:{end}")
    return local, end


def _equivalent_pump_groups(s, state, reference, end, first):
    """Aggregate only interchangeable per-unit pump trajectories.

    Members share receiver, COP, availability and minimum/ramp fractions. Their
    previous, fixed-first and pinned-terminal powers have equal utilization.
    Group power can therefore be split in proportion to rated electrical power:
    summation and proportional splitting preserve every LP constraint and cost.
    Floating keys use 12 decimals, followed by an absolute 1e-9-kW discrepancy
    check; independent replay on the ORIGINAL network remains mandatory.
    """
    n = s.network
    maximum = n.pump_max_kw
    fractions = [n.pump_min_kw/maximum, n.pump_ramp_kw/maximum,
                 state.previous_power_kw/maximum]
    if end < s.horizon:
        fractions.append(reference.power_kw[end-1]/maximum)
    if first is not None:
        fractions.append(np.asarray(first.power_kw)/maximum)
    catalogue = {}
    for i in range(n.n_pumps):
        key = (int(n.pump_to_tank[i]), float(n.pump_cop[i]),
               *(round(float(v[i]), 12) for v in fractions),
               s.availability[state.t:end, i].tobytes())
        catalogue.setdefault(key, []).append(i)
    if len(catalogue) == n.n_pumps:
        return None
    groups = [np.asarray(v, dtype=np.int64) for v in catalogue.values()]
    representatives = np.asarray([g[0] for g in groups])
    membership = np.empty(n.n_pumps, dtype=np.int64)
    for k, group in enumerate(groups):
        membership[group] = k
    totals = {name: np.asarray([getattr(n, name)[g].sum() for g in groups])
              for name in ("pump_max_kw", "pump_min_kw", "pump_ramp_kw")}
    weights = maximum/totals["pump_max_kw"][membership]
    discrepancy = 0.0
    for fraction in fractions:
        original = fraction*maximum
        combined = np.asarray([original[g].sum() for g in groups])
        discrepancy = max(discrepancy, float(np.max(np.abs(original-combined[membership]*weights))))
    if discrepancy > 1e-9:
        return None
    network = replace(n, **totals, pump_to_tank=n.pump_to_tank[representatives],
        pump_station=n.pump_station[representatives], pump_cop=n.pump_cop[representatives],
        pump_models=tuple("equivalent_pump_group" for _ in groups))
    def combine(values):
        return np.asarray([np.asarray(values)[g].sum() for g in groups])
    grouped_scenario = replace(s, network=network,
        availability=s.availability[:, representatives], initial_power_kw=combine(s.initial_power_kw))
    grouped_state = replace(state, previous_power_kw=combine(state.previous_power_kw))
    grouped_reference = replace(reference,
        power_kw=np.stack([reference.power_kw[:, g].sum(axis=1) for g in groups], axis=1))
    grouped_first = None if first is None else Action(combine(first.power_kw), first.delivery_kw)
    return grouped_scenario, grouped_state, grouped_reference, grouped_first, membership, weights, discrepancy


def solve_control_window(s: Scenario, state: State, reference: Plan,
                         window_steps: int = 48, fixed_first_action: Action | None = None,
                         time_limit_s: float = 60.0,
                         solver_method: str = "highs-ipm", solver_threads: int = 1,
                         aggregate_pumps: bool = False) -> SolveResult:
    """Cost MPC, optionally fixing the first action for OFFLINE value labels.

    Both paths share the same forecast window, physical constraints and terminal
    reference. Neither learned online controller calls this optimizer.
    Returned Plan has absolute start_t and covers only the prediction window.
    ``aggregate_pumps`` enables an optional exact pump-symmetry presolve. The
    original network is replayed after expanding any grouped solution. Cold
    unaggregated solves remain the numerical correctness oracle.
    """
    started = perf_counter()
    timings = {"assembly": 0.0, "solve": 0.0, "validation": 0.0, "total": 0.0}
    stats: dict[str, Any] = {}

    def result(status: str, message: str, plan: Plan | None = None,
               cost: float | None = None, lower: float | None = None) -> SolveResult:
        timings["total"] = perf_counter()-started
        return SolveResult(status, plan, cost, lower, dict(timings), message, dict(stats))

    try:
        if (isinstance(time_limit_s, (bool, np.bool_))
                or not np.isfinite(time_limit_s) or time_limit_s <= 0):
            raise ValueError("time_limit_s must be finite and positive")
        if not isinstance(solver_method, str) or solver_method not in {"highs", "highs-ipm", "highs-ds"}:
            raise ValueError("solver_method must be 'highs', 'highs-ipm' or 'highs-ds'")
        if isinstance(solver_threads, (bool, np.bool_)) or not isinstance(solver_threads, (int, np.integer)) or not 1 <= solver_threads <= 64:
            raise ValueError("solver_threads must be an integer in [1,64]")
        if not isinstance(aggregate_pumps, bool):
            raise ValueError("aggregate_pumps must be boolean")
        local, end = _window(s, state, reference, window_steps)
        local_state = State(0, state.energy_kwh.copy(), state.previous_power_kw.copy())
        stats.update(pump_aggregation_requested=aggregate_pumps, pump_aggregation_applied=False,
                     physical_pump_count=s.network.n_pumps, solver_pump_count=s.network.n_pumps,
                     aggregation_time_s=0.0)
        if aggregate_pumps:
            validate_scenario(s)
            # Check individual inputs before grouping can hide an invalid member.
            n = s.network
            if (np.any(state.energy_kwh < -1e-7) or np.any(state.energy_kwh > n.tank_capacity_kwh+1e-7)
                    or np.any(state.previous_power_kw < -1e-7) or np.any(state.previous_power_kw > n.pump_max_kw+1e-7)):
                raise ValueError("actual state is outside physical limits")
            first_check = None if fixed_first_action is None else validate_action(s, state, fixed_first_action, atol=_TOL)
            if first_check is not None and not first_check.feasible:
                return result("infeasible", "fixed first action failed physical checks: "+"; ".join(first_check.violations))
            grouping_started = perf_counter()
            grouped = _equivalent_pump_groups(s, state, reference, end, fixed_first_action)
            stats["aggregation_time_s"] = perf_counter()-grouping_started
            if grouped is not None:
                gs, gx, gr, gf, membership, weights, discrepancy = grouped
                preparation = perf_counter()-started
                remaining = time_limit_s-preparation
                if remaining <= 0:
                    timings["assembly"] = preparation
                    return result("limit", "pump aggregation exhausted the control-window budget")
                solved_group = solve_control_window(gs, gx, gr, window_steps=window_steps,
                    fixed_first_action=gf, time_limit_s=remaining, solver_method=solver_method,
                    solver_threads=solver_threads, aggregate_pumps=False)
                own_stats = dict(stats)
                stats.update(solved_group.statistics)
                stats.update(own_stats, pump_aggregation_applied=True,
                    solver_pump_count=gs.network.n_pumps, aggregation_max_roundoff_kw=discrepancy,
                    aggregation_algorithm="proportional_equal_utilization_same_receiver_and_cop")
                timings.update(solved_group.timings_s)
                timings["assembly"] += preparation
                if solved_group.plan is None:
                    return result(solved_group.status, solved_group.message)
                checking_started = perf_counter()
                power = solved_group.plan.power_kw[:, membership]*weights
                if fixed_first_action is not None:
                    power[0] = fixed_first_action.power_kw  # retain accepted ulps exactly
                if end < s.horizon:
                    power[-1] = reference.power_kw[end-1]
                expanded = replace(solved_group.plan, power_kw=power)
                audit = validate_plan(local, replace(expanded, start_t=0), state=local_state, atol=_TOL)
                stats["aggregation_original_network_residuals"] = audit.max_residuals
                stats["aggregation_original_network_feasible"] = audit.feasible
                if first_check is not None:
                    stats["fixed_action_residuals"] = first_check.max_residuals
                timings["validation"] += perf_counter()-checking_started
                if not audit.feasible:
                    return result("error", "expanded pump-group solution failed original-network replay: "+"; ".join(audit.violations))
                cost = float(local.dt_hours*np.dot(local.price_yuan_per_kwh, power.sum(axis=1)))
                return result(solved_group.status, solved_group.message+"; equivalent pump aggregation verified",
                    expanded, cost, cost if solved_group.status == "optimal" else None)
        lp = build_linear_program(local, state=local_state, fixed_first_action=fixed_first_action, objective="electricity")
        n, length, original_count = s.network, local.horizon, lp.c.size
        stats.update(lp.statistics)
        stats.update(window_start=int(state.t), window_end=end, window_steps=length,
                     terminal_power_pinned=end < s.horizon, solver_method=solver_method,
                     solver_threads=int(solver_threads),
                     objective="electricity", lower_bound_scope="window_only",
                     fixed_first_action=fixed_first_action is not None)
        bounds = lp.bounds.copy()
        if fixed_first_action is not None:
            first_check = validate_action(s, state, fixed_first_action, atol=_TOL)
            if not first_check.feasible:
                return result("infeasible", "fixed first action failed physical checks: "+"; ".join(first_check.violations))
            # Accepted solver actions can sit a few ulps outside a nominal
            # bound. Preserve that exact action, with the SAME acceptance
            # tolerance as all controllers, instead of clipping the label or
            # declaring a reversed 1e-14-wide bound infeasible.
            for block, values in ((slice(0, n.n_pumps), fixed_first_action.power_kw),
                    (slice(lp.variable_slices["delivery"].start,
                           lp.variable_slices["delivery"].start+n.n_edges), fixed_first_action.delivery_kw)):
                bounds[block, :] = np.asarray(values)[:, None]
            stats.update(fixed_action_acceptance_atol=_TOL,
                         fixed_action_residuals=first_check.max_residuals)
        if end < s.horizon:
            block = slice((length-1)*n.n_pumps, length*n.n_pumps)
            desired = reference.power_kw[end-1]
            bounds[block, 0] = np.maximum(bounds[block, 0], desired)
            bounds[block, 1] = np.minimum(bounds[block, 1], desired)
        c, a_eq, a_ub, b_eq, b_ub = lp.c.copy(), lp.A_eq, lp.A_ub, lp.b_eq, lp.b_ub
        stats["inconsistent_bound_count"] = int(np.count_nonzero(bounds[:, 0] > bounds[:, 1]))
    except (ValueError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        timings["assembly"] = perf_counter()-started
        return result("error", f"invalid control input: {exc}")
    timings["assembly"] = perf_counter()-started
    if stats["inconsistent_bound_count"]:
        return result("infeasible", "terminal reference power conflicts with pump operating bounds")
    clock = perf_counter()
    try:
        with warnings.catch_warnings():
            # SciPy 1.15.3 intentionally forwards this valid native HiGHS option.
            warnings.filterwarnings("ignore", message="Unrecognized options detected: .*'threads'.*",
                                    category=OptimizeWarning)
            solved = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                             bounds=bounds, method=solver_method,
                             options={"time_limit": float(time_limit_s), "threads": int(solver_threads)})
    except Exception as exc:
        timings["solve"] = perf_counter()-clock
        return result("error", f"HiGHS failed: {type(exc).__name__}: {exc}")
    timings["solve"] = perf_counter()-clock
    raw_status = int(solved.status)
    status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(raw_status, "error")
    stats.update(solver_status=raw_status, solver_iterations=int(getattr(solved, "nit", 0) or 0),
                 crossover_iterations=int(getattr(solved, "crossover_nit", 0) or 0))
    message = str(solved.message)
    plan, cost = None, None
    clock = perf_counter()
    if getattr(solved, "x", None) is not None:
        try:
            vector = np.asarray(solved.x)
            if vector.shape != c.shape or not np.isfinite(vector).all():
                raise ValueError("solver vector shape or finiteness is invalid")
            candidate = lp.plan_from_vector(vector[:original_count])
            report = validate_plan(local, candidate, state=lp.state, atol=_TOL)
            power_residual = (float(np.max(np.abs(candidate.power_kw[-1]-reference.power_kw[end-1])))
                              if end < s.horizon else 0.0)
            stats["validation_residuals"] = dict(report.max_residuals, terminal_power_match=power_residual)
            stats["validation_feasible"] = report.feasible and power_residual <= _TOL
            if stats["validation_feasible"]:
                plan = replace(candidate, start_t=int(state.t))
                cost = float(local.dt_hours*np.dot(local.price_yuan_per_kwh, candidate.power_kw.sum(axis=1)))
            else:
                message += "; independent validation failed: "+"; ".join(report.violations)
                if power_residual > _TOL:
                    message += "; terminal reference power mismatch"
        except Exception as exc:
            stats["validation_feasible"] = False
            message += f"; independent validation failed: {type(exc).__name__}: {exc}"
    timings["validation"] = perf_counter()-clock
    if status == "optimal" and (not solved.success or plan is None):
        status = "error"
        message += "; an unvalidated solver result is not an optimum"
    if plan is not None and status in {"infeasible", "unbounded"}:
        status = "error"
        message += "; solver status contradicts the validated primal plan"
    lower = cost if status == "optimal" else None
    return result(status, message, plan, cost, lower)


def stitch_reference_tail(s: Scenario, state: State, window_plan: Plan, reference: Plan) -> Plan:
    """Attach a validated control window to a validated reference tail.

    Only interface conditions are checked here; call validate_plan on the final
    executed trajectory for independent whole-episode replay. Full replay on every
    cache update is deliberately avoided. Inputs must come from the validated
    constructors above, and a failed interface raises ValueError.
    """
    _reference_arrays(s, reference)
    n = s.network
    if (isinstance(state.t, (bool, np.bool_)) or not isinstance(state.t, (int, np.integer))
            or not 0 <= state.t < s.horizon):
        raise ValueError("state.t must be an active integer time step")
    if (isinstance(window_plan.start_t, (bool, np.bool_))
            or not isinstance(window_plan.start_t, (int, np.integer))
            or window_plan.start_t != state.t):
        raise ValueError("window start must match the current state")
    if np.asarray(window_plan.power_kw).ndim != 2:
        raise ValueError("window power must be a two-dimensional array")
    length = window_plan.power_kw.shape[0]
    if not 1 <= length <= s.horizon-state.t:
        raise ValueError("window length is outside the remaining episode")
    _numeric(window_plan.power_kw, (length, n.n_pumps), "window power")
    _numeric(window_plan.delivery_kw, (length, n.n_edges), "window delivery")
    _numeric(window_plan.energy_kwh, (length+1, n.n_tanks), "window energy")
    _numeric(state.energy_kwh, (n.n_tanks,), "state.energy_kwh")
    _numeric(state.previous_power_kw, (n.n_pumps,), "state.previous_power_kw")
    if not np.allclose(window_plan.energy_kwh[0], state.energy_kwh, atol=_TOL, rtol=0):
        raise ValueError("window initial energy does not match current state")
    first = validate_action(s, state, window_plan.action(0), atol=_TOL)
    if not first.feasible:
        raise ValueError("window first action fails current-state constraints: "+"; ".join(first.violations))
    end = state.t+length
    target = s.terminal_energy_kwh if end == s.horizon else reference.energy_kwh[end]
    if not np.allclose(window_plan.energy_kwh[-1], target, atol=_TOL, rtol=0):
        raise ValueError("window terminal inventory does not match reference continuation")
    if end < s.horizon:
        if not np.allclose(window_plan.power_kw[-1], reference.power_kw[end-1], atol=_TOL, rtol=0):
            raise ValueError("window terminal power does not match reference continuation")
        if np.any(np.abs(reference.power_kw[end]-window_plan.power_kw[-1]) > n.pump_ramp_kw+_TOL):
            raise ValueError("reference continuation violates pump ramp limits")
    return Plan(int(state.t),
                np.concatenate([window_plan.power_kw, reference.power_kw[end:]], axis=0),
                np.concatenate([window_plan.delivery_kw, reference.delivery_kw[end:]], axis=0),
                np.concatenate([window_plan.energy_kwh, reference.energy_kwh[end+1:]], axis=0))
