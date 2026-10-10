"""Nested Sobol proposals containing only the current physical action.

Total power is allocated conditionally across pump intervals. An anchor
delivery witnesses a conservative initial box; exact constraint-preserving
pair moves explore beyond it. No future trajectory or optimizer is used.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.stats import qmc
from .physics import validate_action
from .types import Action, Plan, Scenario, State

_POINT_CACHE: OrderedDict = OrderedDict()
_CACHE_LIMIT_BYTES = 128*1024*1024


@dataclass(frozen=True)
class Candidate:
    action: Action
    candidate_id: int
    next_energy_kwh: np.ndarray


@dataclass(frozen=True)
class CandidatePool:
    candidates: list[Candidate]
    stats: dict

    @property
    def actions(self):
        return [candidate.action for candidate in self.candidates]

    @property
    def ids(self):
        return np.asarray([candidate.candidate_id for candidate in self.candidates], dtype=np.int64)

    def __len__(self):
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index]


def _sobol_prefix(dimension, count, seed):
    key = (dimension, seed)
    cached = _POINT_CACHE.get(key)
    if cached is not None and len(cached) >= count:
        _POINT_CACHE.move_to_end(key)
        return cached[:count], True
    points = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(count.bit_length()-1)
    points.setflags(write=False)
    _POINT_CACHE[key] = points
    _POINT_CACHE.move_to_end(key)
    while len(_POINT_CACHE) > 1 and sum(value.nbytes for value in _POINT_CACHE.values()) > _CACHE_LIMIT_BYTES:
        _POINT_CACHE.popitem(last=False)
    return points, False


def _group_sum(values, groups, size):
    if values.ndim == 1:
        return np.bincount(groups, weights=values, minlength=size)
    result = np.zeros((len(values), size), dtype=np.float64)
    np.add.at(result, (np.arange(len(values))[:, None], groups[None, :]), values)
    return result


def _mix_to_witness(values, witness, groups, weights, bound, *, lower):
    """Tighten one side of an independent box towards a feasible witness."""
    baseline = _group_sum(values*weights, groups, len(bound))
    known = _group_sum(witness*weights, groups, len(bound))
    numerator = bound-baseline if lower else baseline-bound
    denominator = known-baseline if lower else baseline-known
    fraction = np.divide(np.maximum(numerator, 0), denominator,
                         out=np.zeros_like(numerator), where=denominator > 1e-14)
    return values+np.clip(fraction, 0, 1)[groups]*(witness-values)


def _restrict(lo, hi, value, coefficient, lower, upper):
    if coefficient > 0:
        lo = np.maximum(lo, (lower-value)/coefficient)
        hi = np.minimum(hi, (upper-value)/coefficient)
    elif coefficient < 0:
        lo = np.maximum(lo, (upper-value)/coefficient)
        hi = np.minimum(hi, (lower-value)/coefficient)
    return lo, hi


def _conditional_power(points, lower, upper, total, order):
    """Equation 14e: reserve the remaining devices' lower/upper sums."""
    power = np.empty((len(points), len(lower)), dtype=np.float64)
    remaining_lower = np.cumsum(lower[order][::-1])[::-1]
    remaining_upper = np.cumsum(upper[order][::-1])[::-1]
    remaining = total.copy()
    for offset, pump in enumerate(order[:-1]):
        lo = np.maximum(lower[pump], remaining-remaining_upper[offset+1])
        hi = np.minimum(upper[pump], remaining-remaining_lower[offset+1])
        power[:, pump] = lo+np.maximum(hi-lo, 0)*points[:, offset]
        remaining -= power[:, pump]
    power[:, order[-1]] = remaining
    return power


def _key(s, action):
    n = s.network
    normalized = np.concatenate((action.power_kw/n.pump_max_kw,
                                 action.delivery_kw/np.maximum(n.edge_capacity_kw, 1)))
    return np.round(normalized, 12).tobytes()


def generate_candidates(s: Scenario, state: State, anchor: Plan,
                        raw_count: int, seed: int) -> CandidatePool:
    """N power-of-two Sobol actions plus unchanged backup; no future plans.

    IDs are zero for backup and raw Sobol row + 1 thereafter. Numerical
    rejection and dedup preserve IDs/prefixes; no refill. Reuse seed across
    time steps to reuse scrambled points, with fresh state-dependent mapping.
    The witness box certifies a subset of feasible total powers. Pair moves
    use original current constraints, not this box; this is feasible
    exploration, not a uniform sampler of the entire feasible polytope.
    """
    started = perf_counter()
    if isinstance(raw_count, (bool, np.bool_)) or not isinstance(raw_count, (int, np.integer)) or raw_count < 1 or raw_count & (raw_count-1):
        raise ValueError("raw_count must be a positive power of two")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if anchor.start_t != state.t:
        raise ValueError("anchor must start at the actual state")
    first = anchor.action(0)
    checked = validate_action(s, state, first)
    if not checked.feasible:
        raise ValueError("anchor current action is infeasible: "+"; ".join(checked.violations))
    n, count = s.network, int(raw_count)
    points, hit = _sobol_prefix(1+2*n.n_pumps+n.n_edges, count, int(seed))
    lower = np.maximum(s.availability[state.t]*n.pump_min_kw, state.previous_power_kw-n.pump_ramp_kw)
    upper = np.minimum(s.availability[state.t]*n.pump_max_kw, state.previous_power_kw+n.pump_ramp_kw)
    if np.any(lower-upper > 1e-7):
        raise ValueError("current pump bounds have an empty intersection")
    lower = np.minimum(lower, upper)
    temperature = n.tank_reference_c+state.energy_kwh/n.tank_heat_capacity_kwh_per_k
    loss = n.tank_ua_kw_per_k*(temperature-s.tank_environment_c[state.t])
    passive = state.energy_kwh-s.dt_hours*loss
    final = state.t == s.horizon-1
    energy_lower = s.terminal_energy_kwh if final else np.zeros(n.n_tanks)
    energy_upper = s.terminal_energy_kwh if final else n.tank_capacity_kwh
    delivery0 = _group_sum(first.delivery_kw, n.delivery_edges[0], n.n_tanks)
    heat_lower = np.maximum(0, (energy_lower-passive+s.dt_hours*delivery0/n.eta_discharge)/(s.dt_hours*n.eta_charge))
    heat_upper = np.minimum(n.tank_charge_limit_kw,
                           (energy_upper-passive+s.dt_hours*delivery0/n.eta_discharge)/(s.dt_hours*n.eta_charge))
    box_lower = _mix_to_witness(lower, first.power_kw, n.pump_to_tank, n.pump_cop, heat_lower, lower=True)
    box_upper = _mix_to_witness(upper, first.power_kw, n.pump_to_tank, n.pump_cop, heat_upper, lower=False)
    box_upper = _mix_to_witness(box_upper, first.power_kw, n.pump_station, n.pump_cop-1,
                                s.source_limit_kw[state.t], lower=False)
    total_lower = max(float(box_lower.sum()), float(s.power_lower_kw[state.t]))
    total_upper = min(float(box_upper.sum()), float(s.power_upper_kw[state.t]))
    if total_lower-total_upper > 1e-6:
        raise ValueError("witness total-power interval is empty")
    total_upper = max(total_lower, total_upper)
    total = total_lower+(total_upper-total_lower)*points[:, 0]
    order = np.random.default_rng(int(seed)).permutation(n.n_pumps)
    power = _conditional_power(points[:, 1:1+n.n_pumps], box_lower, box_upper, total, order)
    charge = _group_sum(power*n.pump_cop, n.pump_to_tank, n.n_tanks)
    source = _group_sum(power*(n.pump_cop-1), n.pump_station, n.n_stations)
    # Constraint-preserving pairs keep total power fixed and leave the initial
    # box. A fixed global power band still permits device redistribution.
    for k, i in enumerate(order):
        j = int(order[(k+1) % n.n_pumps])
        if i == j:
            continue
        lo = np.maximum(lower[i]-power[:, i], power[:, j]-upper[j])
        hi = np.minimum(upper[i]-power[:, i], power[:, j]-lower[j])
        ti, tj = int(n.pump_to_tank[i]), int(n.pump_to_tank[j])
        gi, gj = int(n.pump_station[i]), int(n.pump_station[j])
        if ti == tj:
            lo, hi = _restrict(lo, hi, charge[:, ti], n.pump_cop[i]-n.pump_cop[j], heat_lower[ti], heat_upper[ti])
        else:
            lo, hi = _restrict(lo, hi, charge[:, ti], n.pump_cop[i], heat_lower[ti], heat_upper[ti])
            lo, hi = _restrict(lo, hi, charge[:, tj], -n.pump_cop[j], heat_lower[tj], heat_upper[tj])
        if gi == gj:
            lo, hi = _restrict(lo, hi, source[:, gi], n.pump_cop[i]-n.pump_cop[j], 0, s.source_limit_kw[state.t, gi])
        else:
            lo, hi = _restrict(lo, hi, source[:, gi], n.pump_cop[i]-1, 0, s.source_limit_kw[state.t, gi])
            lo, hi = _restrict(lo, hi, source[:, gj], -(n.pump_cop[j]-1), 0, s.source_limit_kw[state.t, gj])
        delta = lo+np.maximum(hi-lo, 0)*points[:, 1+n.n_pumps+k]
        power[:, i] += delta
        power[:, j] -= delta
        charge[:, ti] += n.pump_cop[i]*delta
        charge[:, tj] -= n.pump_cop[j]*delta
        source[:, gi] += (n.pump_cop[i]-1)*delta
        source[:, gj] -= (n.pump_cop[j]-1)*delta
    flow = np.broadcast_to(first.delivery_kw, (count, n.n_edges)).copy()
    discharge = np.broadcast_to(delivery0, (count, n.n_tanks)).copy()
    pre_delivery = passive[None, :]+s.dt_hours*n.eta_charge[None, :]*charge
    discharge_lower = np.maximum(0, (pre_delivery-energy_upper)*n.eta_discharge/s.dt_hours)
    discharge_upper = np.minimum(n.tank_discharge_limit_kw, (pre_delivery-energy_lower)*n.eta_discharge/s.dt_hours)
    for building in range(n.n_buildings):
        edges = np.flatnonzero(n.delivery_edges[1] == building)
        if len(edges) < 2:
            continue
        pivot = int(edges[np.argmax(first.delivery_kw[edges])])
        tp = int(n.delivery_edges[0, pivot])
        for edge in edges:
            if edge == pivot:
                continue
            te = int(n.delivery_edges[0, edge])
            lo = np.maximum(-flow[:, edge], flow[:, pivot]-n.edge_capacity_kw[pivot])
            hi = np.minimum(n.edge_capacity_kw[edge]-flow[:, edge], flow[:, pivot])
            if te != tp:
                lo, hi = _restrict(lo, hi, discharge[:, te], 1, discharge_lower[:, te], discharge_upper[:, te])
                lo, hi = _restrict(lo, hi, discharge[:, tp], -1, discharge_lower[:, tp], discharge_upper[:, tp])
            delta = lo+np.maximum(hi-lo, 0)*points[:, 1+2*n.n_pumps+edge]
            flow[:, edge] += delta
            flow[:, pivot] -= delta
            discharge[:, te] += delta
            discharge[:, tp] -= delta
    # Recompute final aggregates rather than trusting incremental bookkeeping.
    charge = _group_sum(power*n.pump_cop, n.pump_to_tank, n.n_tanks)
    source = _group_sum(power*(n.pump_cop-1), n.pump_station, n.n_stations)
    discharge = _group_sum(flow, n.delivery_edges[0], n.n_tanks)
    delivered = _group_sum(flow, n.delivery_edges[1], n.n_buildings)
    next_energy = passive+s.dt_hours*(n.eta_charge*charge-discharge/n.eta_discharge)
    tol = 1e-6
    checks = {
        "nonfinite": np.isfinite(power).all(axis=1)&np.isfinite(flow).all(axis=1)&np.isfinite(next_energy).all(axis=1),
        "pump_bounds": ((power >= lower-tol)&(power <= upper+tol)).all(axis=1),
        "power_band": (power.sum(axis=1) >= s.power_lower_kw[state.t]-tol)&(power.sum(axis=1) <= s.power_upper_kw[state.t]+tol),
        "source": (source <= s.source_limit_kw[state.t]+tol).all(axis=1),
        "charge": (charge <= n.tank_charge_limit_kw+tol).all(axis=1),
        "edge_bounds": ((flow >= -tol)&(flow <= n.edge_capacity_kw+tol)).all(axis=1),
        "discharge": (discharge <= n.tank_discharge_limit_kw+tol).all(axis=1),
        "building_balance": (np.abs(delivered-s.heat_demand_kw[state.t]) <= tol).all(axis=1),
        "inventory": ((next_energy >= energy_lower-tol)&(next_energy <= energy_upper+tol)).all(axis=1),
    }
    legal = np.logical_and.reduce(list(checks.values()))
    first_energy = passive+s.dt_hours*(n.eta_charge*_group_sum(first.power_kw*n.pump_cop, n.pump_to_tank, n.n_tanks)-delivery0/n.eta_discharge)
    candidates = [Candidate(first, 0, first_energy)]
    seen, duplicates = {_key(s, first)}, 0
    for row in np.flatnonzero(legal):
        action = Action(power[row], flow[row])
        key = _key(s, action)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        candidates.append(Candidate(action, int(row)+1, next_energy[row]))
    for candidate in candidates:
        for array in (candidate.action.power_kw, candidate.action.delivery_kw, candidate.next_energy_kwh):
            array.setflags(write=False)
    stats = dict(raw_count=count, requested_count=count+1, retained_count=len(candidates),
                 accepted_raw_count=int(legal.sum()), rejected_count=int((~legal).sum()), duplicate_count=duplicates,
                 rejection_reasons={name: int((~mask).sum()) for name, mask in checks.items() if not mask.all()},
                 sobol_dimension=points.shape[1], sobol_cache_hit=hit,
                 total_power_interval_kw=[total_lower, total_upper],
                 mapping="total_first_witness_box_then_constraint_preserving_pairs",
                 generation_time_s=perf_counter()-started)
    return CandidatePool(candidates, stats)
