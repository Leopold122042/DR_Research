"""Sparse remaining-horizon dispatch for the declared linear reference plant.

All power variables are in kW, inventories in kWh and prices in yuan/kWh.
The public matrix representation supports audits without running the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
import warnings

import numpy as np
from scipy.optimize import linprog, OptimizeWarning
from scipy.sparse import coo_matrix, csr_matrix

from .types import Action, Plan, Scenario, SolveResult, State


@dataclass(frozen=True)
class LinearProgram:
    """Variables are time-major P, time-major edge F, then boundary-state S.

    ``bounds[:, 0]`` and ``bounds[:, 1]`` are lower and upper bounds. A fixed
    first action is intersected with the original equipment bounds, so a lower
    bound above its upper bound is an explicit certificate of incompatibility.
    ``constraint_blocks`` stores half-open row ranges for auditing each block.
    """

    c: np.ndarray
    A_eq: csr_matrix
    b_eq: np.ndarray
    A_ub: csr_matrix
    b_ub: np.ndarray
    bounds: np.ndarray
    state: State
    remaining_horizon: int
    variable_slices: dict[str, slice]
    variable_shapes: dict[str, tuple[int, int]]
    constraint_blocks: dict[str, dict[str, tuple[int, int]]]
    objective: str
    statistics: dict[str, Any]

    def plan_from_vector(self, vector: np.ndarray) -> Plan:
        values = np.asarray(vector, dtype=float)
        if values.shape != self.c.shape or not np.isfinite(values).all():
            raise ValueError("solver vector has an invalid shape or nonfinite values")
        arrays = {
            name: values[part].reshape(self.variable_shapes[name]).copy()
            for name, part in self.variable_slices.items()
        }
        return Plan(self.state.t, arrays["power"], arrays["delivery"], arrays["energy"])


class _Rows:
    """Accumulate sparse coordinate chunks; never allocate a dense matrix."""

    def __init__(self, n_variables: int) -> None:
        self.n_variables = n_variables
        self.n_rows = 0
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.data: list[np.ndarray] = []
        self.rhs: list[np.ndarray] = []
        self.blocks: dict[str, tuple[int, int]] = {}

    def block(self, name: str, rhs: np.ndarray) -> int:
        values = np.asarray(rhs, dtype=float).reshape(-1)
        start = self.n_rows
        self.n_rows += values.size
        self.rhs.append(values)
        self.blocks[name] = (start, self.n_rows)
        return start

    def add(self, rows: np.ndarray, cols: np.ndarray, values: Any) -> None:
        rr, cc, vv = np.broadcast_arrays(rows, cols, values)
        rr, cc, vv = rr.reshape(-1), cc.reshape(-1), vv.reshape(-1)
        nonzero = vv != 0
        if nonzero.any():
            self.rows.append(rr[nonzero].astype(np.int64, copy=False))
            self.cols.append(cc[nonzero].astype(np.int64, copy=False))
            self.data.append(vv[nonzero].astype(float, copy=False))

    def finish(self) -> tuple[csr_matrix, np.ndarray]:
        if self.data:
            matrix = coo_matrix(
                (np.concatenate(self.data),
                 (np.concatenate(self.rows), np.concatenate(self.cols))),
                shape=(self.n_rows, self.n_variables),
            ).tocsr()
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
        else:
            matrix = csr_matrix((self.n_rows, self.n_variables), dtype=float)
        return matrix, np.concatenate(self.rhs) if self.rhs else np.empty(0)


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != (size,) or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array) or np.issubdtype(array.dtype, np.bool_)
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return array.astype(float, copy=False)


def _inputs(scenario: Scenario, state: State | None,
            first: Action | None, objective: str) -> tuple[State, Action | None]:
    from .physics import validate_scenario

    validate_scenario(scenario)
    if not isinstance(objective, str) or objective != "electricity":
        raise ValueError("objective must be 'electricity'")
    net = scenario.network
    initial = scenario.initial_state() if state is None else state
    if (isinstance(initial.t, (bool, np.bool_))
            or not isinstance(initial.t, (int, np.integer))
            or not 0 <= initial.t <= scenario.horizon):
        raise ValueError("state.t must be an integer boundary in [0, horizon]")
    energy = _vector(initial.energy_kwh, net.n_tanks, "state.energy_kwh")
    previous = _vector(initial.previous_power_kw, net.n_pumps, "state.previous_power_kw")
    tol = 1e-7
    if np.any(energy < -tol) or np.any(energy > net.tank_capacity_kwh + tol):
        raise ValueError("state energy is outside tank limits")
    if np.any(previous < -tol) or np.any(previous > net.pump_max_kw + tol):
        raise ValueError("state previous power is outside pump limits")
    initial = State(int(initial.t), energy.copy(), previous.copy())
    if first is not None:
        if initial.t == scenario.horizon:
            raise ValueError("a terminal state has no first action to fix")
        first = Action(_vector(first.power_kw, net.n_pumps, "fixed_first_action.power_kw").copy(),
                       _vector(first.delivery_kw, net.n_edges, "fixed_first_action.delivery_kw").copy())
    return initial, first


def build_linear_program(scenario: Scenario, state: State | None = None,
                         fixed_first_action: Action | None = None,
                         objective: str = "electricity") -> LinearProgram:
    """Build CSR constraints for all remaining intervals without solving them.

    Availability is exogenous: unavailable pumps have zero power, while an
    available pump uses its declared minimum and maximum power bounds. Charging
    capacity applies before eta_charge; all delivery capacities use useful heat.
    """
    state, first = _inputs(scenario, state, fixed_first_action, objective)
    n = scenario.network
    start, length = state.t, scenario.horizon - state.t
    H, S, B, E, G = n.n_pumps, n.n_tanks, n.n_buildings, n.n_edges, n.n_stations
    p_count, f_count = length * H, length * E
    s_offset = p_count + f_count
    count = s_offset + (length + 1) * S
    slices = {"power": slice(0, p_count), "delivery": slice(p_count, s_offset),
              "energy": slice(s_offset, count)}
    shapes = {"power": (length, H), "delivery": (length, E), "energy": (length + 1, S)}
    c = np.zeros(count)
    c[:p_count] = np.repeat(scenario.price_yuan_per_kwh[start:] * scenario.dt_hours, H)
    bounds = np.zeros((count, 2))
    available = scenario.availability[start:]
    bounds[:p_count, 0] = (available * n.pump_min_kw).reshape(-1)
    bounds[:p_count, 1] = (available * n.pump_max_kw).reshape(-1)
    bounds[p_count:s_offset, 1] = np.tile(n.edge_capacity_kw, length)
    bounds[s_offset:, 1] = np.tile(n.tank_capacity_kwh, length + 1)
    bounds[s_offset:s_offset + S, :] = state.energy_kwh[:, None]
    if first is not None:
        for block, values in ((slice(0, H), first.power_kw),
                              (slice(p_count, p_count + E), first.delivery_kw)):
            bounds[block, 0] = np.maximum(bounds[block, 0], values)
            bounds[block, 1] = np.minimum(bounds[block, 1], values)

    eq, ub = _Rows(count), _Rows(count)
    pumps, tanks, edges = np.arange(H), np.arange(S), np.arange(E)
    edge_tank, edge_building = n.delivery_edges
    dt = scenario.dt_hours
    retention = 1.0 - dt * n.tank_ua_kw_per_k / n.tank_heat_capacity_kwh_per_k
    for offset in range(length):
        t = start + offset
        pp = offset * H + pumps
        ff = p_count + offset * E + edges
        ss = s_offset + offset * S + tanks
        next_ss = ss + S
        loss_constant = -dt * n.tank_ua_kw_per_k * (n.tank_reference_c - scenario.tank_environment_c[t])
        row = eq.block(f"storage[{t}]", loss_constant)
        eq.add(row + tanks, next_ss, 1.0)
        eq.add(row + tanks, ss, -retention)
        eq.add(row + n.pump_to_tank, pp, -dt * n.eta_charge[n.pump_to_tank] * n.pump_cop)
        eq.add(row + edge_tank, ff, dt / n.eta_discharge[edge_tank])
        row = eq.block(f"building_demand[{t}]", scenario.heat_demand_kw[t])
        eq.add(row + edge_building, ff, 1.0)

        up_rhs = n.pump_ramp_kw + (state.previous_power_kw if offset == 0 else 0.0)
        row = ub.block(f"ramp_up[{t}]", up_rhs)
        ub.add(row + pumps, pp, 1.0)
        if offset:
            ub.add(row + pumps, pp - H, -1.0)
        down_rhs = n.pump_ramp_kw - (state.previous_power_kw if offset == 0 else 0.0)
        row = ub.block(f"ramp_down[{t}]", down_rhs)
        ub.add(row + pumps, pp, -1.0)
        if offset:
            ub.add(row + pumps, pp - H, 1.0)
        row = ub.block(f"tank_charge[{t}]", n.tank_charge_limit_kw)
        ub.add(row + n.pump_to_tank, pp, n.pump_cop)
        row = ub.block(f"tank_delivery[{t}]", n.tank_discharge_limit_kw)
        ub.add(row + edge_tank, ff, 1.0)
        row = ub.block(f"source[{t}]", scenario.source_limit_kw[t])
        ub.add(row + n.pump_station, pp, n.pump_cop - 1.0)
        row = ub.block(f"power_upper[{t}]", np.array([scenario.power_upper_kw[t]]))
        ub.add(np.full(H, row), pp, 1.0)
        row = ub.block(f"power_lower[{t}]", np.array([-scenario.power_lower_kw[t]]))
        ub.add(np.full(H, row), pp, -1.0)

    row = eq.block("terminal_energy", scenario.terminal_energy_kwh)
    eq.add(row + tanks, s_offset + length * S + tanks, 1.0)
    A_eq, b_eq = eq.finish()
    A_ub, b_ub = ub.finish()
    stats = {"n_variables": count, "n_equalities": A_eq.shape[0],
             "n_inequalities": A_ub.shape[0],
             "n_constraints": A_eq.shape[0] + A_ub.shape[0],
             "nnz_eq": A_eq.nnz, "nnz_ub": A_ub.nnz,
             "nnz": A_eq.nnz + A_ub.nnz, "remaining_horizon": length,
             "n_pumps": H, "n_tanks": S, "n_buildings": B, "n_stations": G,
             "objective": objective, "fixed_first_action": first is not None,
             "inconsistent_bound_count": int(np.count_nonzero(bounds[:, 0] > bounds[:, 1]))}
    return LinearProgram(c, A_eq, b_eq, A_ub, b_ub, bounds, state, length,
                         slices, shapes, {"eq": eq.blocks, "ub": ub.blocks}, objective, stats)


def solve_dispatch(scenario: Scenario, state: State | None = None,
                   fixed_first_action: Action | None = None,
                   objective: str = "electricity",
                   time_limit_s: float = 60.0,
                   solver_method: str = "highs-ipm", solver_threads: int = 1) -> SolveResult:
    """Solve with HiGHS and independently validate any returned primal plan.

    ``time_limit_s`` is the solver limit, not an end-to-end deadline. Timings
    separately include assembly, solver execution and physical validation.
    The objective is total electricity cost over the remaining intervals.
    The default explicitly selects HiGHS IPM, including its crossover phase;
    ``highs`` and ``highs-ds`` are available for reproducible comparisons.
    """
    started = perf_counter()
    timing = {"assembly": 0.0, "solve": 0.0, "validation": 0.0, "total": 0.0}
    stats: dict[str, Any] = {}

    def result(status: str, message: str, plan: Plan | None = None,
               cost: float | None = None, lower: float | None = None) -> SolveResult:
        timing["total"] = perf_counter() - started
        return SolveResult(status, plan, cost, lower, dict(timing), message, dict(stats))

    try:
        if isinstance(time_limit_s, (bool, np.bool_)) or not np.isfinite(time_limit_s) or time_limit_s <= 0:
            raise ValueError("time_limit_s must be finite and positive")
        if not isinstance(solver_method, str) or solver_method not in {"highs", "highs-ipm", "highs-ds"}:
            raise ValueError("solver_method must be 'highs', 'highs-ipm' or 'highs-ds'")
        if isinstance(solver_threads, (bool, np.bool_)) or not isinstance(solver_threads, (int, np.integer)) or not 1 <= solver_threads <= 64:
            raise ValueError("solver_threads must be an integer in [1,64]")
        stats["solver_method"] = solver_method
        stats["solver_threads"] = int(solver_threads)
        lp = build_linear_program(scenario, state, fixed_first_action, objective)
        stats.update(lp.statistics)
    except (ValueError, TypeError, IndexError, AttributeError) as exc:
        timing["assembly"] = perf_counter() - started
        return result("error", f"invalid dispatch input: {exc}")
    timing["assembly"] = perf_counter() - started
    if stats["inconsistent_bound_count"]:
        return result("infeasible", "fixed first action conflicts with equipment or edge bounds")
    clock = perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected: .*'threads'.*",
                                    category=OptimizeWarning)
            solved = linprog(lp.c, A_ub=lp.A_ub if lp.A_ub.shape[0] else None,
                             b_ub=lp.b_ub if lp.b_ub.size else None,
                             A_eq=lp.A_eq if lp.A_eq.shape[0] else None,
                             b_eq=lp.b_eq if lp.b_eq.size else None,
                             bounds=lp.bounds, method=solver_method,
                             options={"time_limit": float(time_limit_s), "threads": int(solver_threads)})
    except Exception as exc:
        timing["solve"] = perf_counter() - clock
        return result("error", f"HiGHS failed: {type(exc).__name__}: {exc}")
    timing["solve"] = perf_counter() - clock
    raw_status = int(solved.status)
    stats["solver_status"] = raw_status
    stats["solver_iterations"] = int(getattr(solved, "nit", 0) or 0)
    stats["crossover_iterations"] = int(getattr(solved, "crossover_nit", 0) or 0)
    status = {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(raw_status, "error")
    message = str(solved.message)
    plan, cost = None, None
    clock = perf_counter()
    if getattr(solved, "x", None) is not None:
        try:
            from .physics import validate_plan

            candidate = lp.plan_from_vector(solved.x)
            report = validate_plan(scenario, candidate, state=lp.state, atol=1e-5)
            stats["validation_feasible"] = report.feasible
            stats["validation_residuals"] = report.max_residuals
            if report.feasible:
                plan = candidate
                cost = float(scenario.dt_hours * np.dot(
                    scenario.price_yuan_per_kwh[lp.state.t:], candidate.power_kw.sum(axis=1)))
            else:
                message += "; independent validation failed: " + "; ".join(report.violations)
        except Exception as exc:
            stats["validation_feasible"] = False
            message += f"; independent validation failed: {type(exc).__name__}: {exc}"
    timing["validation"] = perf_counter() - clock
    if status == "optimal" and (not solved.success or plan is None):
        status = "error"
        message += "; an unvalidated solver result is not an optimum"
    if plan is not None and status in {"infeasible", "unbounded"}:
        status = "error"
        message += "; solver status contradicts the validated primal plan"
    lower = cost if status == "optimal" else None
    return result(status, message, plan, cost, lower)
