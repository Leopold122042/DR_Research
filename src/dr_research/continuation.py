"""Post-ranking continuation certificates for the exact scored action.

Each bounded attempt constructs only a local window. The unchanged backup
tail is materialized once a window passes independent physical replay.
An unsuccessful construction means UNKNOWN, not physical infeasibility.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from .physics import validate_action, validate_plan
from .types import Action, Plan, Scenario, State

_TIME_FIELDS = ("price_yuan_per_kwh", "heat_demand_kw", "ambient_c",
                "tank_environment_c", "source_limit_kw", "power_lower_kw",
                "power_upper_kw", "availability")


@dataclass(frozen=True)
class ContinuationContext:
    scenario: Scenario
    state: State
    anchor: Plan
    window_steps: int
    length: int
    remaining: int
    initial_energy: np.ndarray
    previous_power: np.ndarray
    decay: np.ndarray


@dataclass(frozen=True)
class ContinuationResult:
    status: str
    plan: Plan | None
    reason: str
    attempts: int = 0
    profile: str | None = None
    replay_count: int = 0

    @property
    def accepted(self):
        return self.status == "accepted"


def prepare_continuation_context(s: Scenario, state: State, anchor: Plan,
                                 window_steps: int) -> ContinuationContext:
    """Check a certified anchor's window once and bind it to this actual state.

    The caller supplies a previously audited immutable continuation. Its tail
    is shared, and its window constraints are checked without another dynamics
    replay. Every newly accepted window receives independent physical replay.
    """
    if isinstance(window_steps, (bool, np.bool_)) or not isinstance(window_steps, (int, np.integer)) or window_steps < 2:
        raise ValueError("window_steps must be an integer of at least two")
    if isinstance(state.t, (bool, np.bool_)) or not isinstance(state.t, (int, np.integer)) or not 0 <= state.t < s.horizon or anchor.start_t != state.t:
        raise ValueError("anchor and state must start at the current active step")
    n, remaining = s.network, s.horizon-state.t
    expected = ((anchor.power_kw, (remaining, n.n_pumps)),
                (anchor.delivery_kw, (remaining, n.n_edges)),
                (anchor.energy_kwh, (remaining+1, n.n_tanks)))
    if any(not isinstance(array, np.ndarray) or array.shape != shape for array, shape in expected):
        raise ValueError("anchor must contain the complete remaining trajectory")
    if not np.allclose(anchor.energy_kwh[0], state.energy_kwh, atol=1e-5, rtol=0):
        raise ValueError("anchor initial inventory does not match actual state")
    if not np.allclose(anchor.energy_kwh[-1], s.terminal_energy_kwh, atol=1e-5, rtol=0):
        raise ValueError("anchor terminal inventory does not match task")
    length = min(int(window_steps), remaining)
    if any(not np.isfinite(array[:length+1]).all() for array, _ in expected):
        raise ValueError("anchor window contains nonfinite values")
    first = validate_action(s, state, anchor.action(0))
    if not first.feasible:
        raise ValueError("anchor first action is infeasible: "+"; ".join(first.violations))
    power, flow, energy = anchor.power_kw[:length], anchor.delivery_kw[:length], anchor.energy_kwh[:length+1]
    available = s.availability[state.t:state.t+length]
    previous = np.vstack([state.previous_power_kw, power[:-1]])
    charge = _sum(power*n.pump_cop, n.pump_to_tank, n.n_tanks)
    discharge = _sum(flow, n.delivery_edges[0], n.n_tanks)
    source = _sum(power*(n.pump_cop-1), n.pump_station, n.n_stations)
    constraints = (
        ("pump_upper", power, available*n.pump_max_kw),
        ("pump_lower", -power, -(available*n.pump_min_kw)),
        ("edge_upper", flow, n.edge_capacity_kw), ("edge_lower", -flow, 0),
        ("ramp_up", power-previous, n.pump_ramp_kw),
        ("ramp_down", previous-power, n.pump_ramp_kw),
        ("energy_upper", energy[1:], n.tank_capacity_kwh), ("energy_lower", -energy[1:], 0),
        ("tank_charge", charge, n.tank_charge_limit_kw),
        ("tank_discharge", discharge, n.tank_discharge_limit_kw),
        ("source", source, s.source_limit_kw[state.t:state.t+length]),
        ("power_upper", power.sum(axis=1), s.power_upper_kw[state.t:state.t+length]),
        ("power_lower", -power.sum(axis=1), -s.power_lower_kw[state.t:state.t+length]),
    )
    for name, value, bound in constraints:
        if not np.isfinite(value).all() or not np.isfinite(bound).all():
            raise ValueError("anchor constraint is nonfinite: "+name)
        if np.any(value-bound > 1e-5):
            raise ValueError("anchor violates "+name)
    if length < remaining and np.any(np.abs(anchor.power_kw[length]-power[-1]) > n.pump_ramp_kw+1e-5):
        raise ValueError("anchor violates tail ramp")
    decay = 1-s.dt_hours*n.tank_ua_kw_per_k/n.tank_heat_capacity_kwh_per_k
    if np.any(decay <= 0) or np.any(decay > 1):
        raise ValueError("explicit tank-loss model requires decay in (0,1]")
    return ContinuationContext(s, state, anchor, int(window_steps), length, remaining,
                               state.energy_kwh.copy(), state.previous_power_kw.copy(), decay)


def _sum(values, groups, size):
    return np.stack([np.bincount(groups, weights=row, minlength=size) for row in values])


def _window_energy(s, state, power, flow):
    n = s.network
    charge = _sum(power*n.pump_cop, n.pump_to_tank, n.n_tanks)
    discharge = _sum(flow, n.delivery_edges[0], n.n_tanks)
    decay = 1-s.dt_hours*n.tank_ua_kw_per_k/n.tank_heat_capacity_kwh_per_k
    energy = np.empty((len(power)+1, n.n_tanks))
    energy[0] = state.energy_kwh
    for k in range(len(power)):
        environment = s.tank_environment_c[state.t+k]
        constant_loss = n.tank_ua_kw_per_k*(n.tank_reference_c-environment)
        energy[k+1] = decay*energy[k]+s.dt_hours*(n.eta_charge*charge[k]-discharge[k]/n.eta_discharge-constant_loss)
    return energy, charge


def try_continuation(s: Scenario, state: State, anchor: Plan, candidate,
                     window_steps: int, *, context: ContinuationContext | None = None,
                     max_profiles: int = 3) -> ContinuationResult:
    """Accept the identical candidate action or return UNKNOWN; never shrink it.

    Future pump powers bridge back toward backup under ramp limits. Independent
    tank corrections use future device, charge, source and ramp headroom. Up to
    three deterministic time profiles are attempted, with no LP or objective
    optimization. A proven backup remains available if every attempt fails.
    """
    if isinstance(max_profiles, bool) or not isinstance(max_profiles, int) or not 1 <= max_profiles <= 3:
        raise ValueError("max_profiles must be an integer from 1 through 3")
    action = candidate if isinstance(candidate, Action) else candidate.action
    checked = validate_action(s, state, action)
    if not checked.feasible:
        return ContinuationResult("unknown", None, "current_action_invalid: "+"; ".join(checked.violations))
    if context is None:
        context = prepare_continuation_context(s, state, anchor, window_steps)
    elif (context.scenario is not s or context.state is not state or context.anchor is not anchor
          or context.window_steps != window_steps
          or not np.array_equal(context.initial_energy, state.energy_kwh)
          or not np.array_equal(context.previous_power, state.previous_power_kw)):
        raise ValueError("continuation context differs from scenario/state/anchor/window")
    if (np.array_equal(action.power_kw, anchor.power_kw[0])
            and np.array_equal(action.delivery_kw, anchor.delivery_kw[0])):
        return ContinuationResult("accepted", anchor, "certified_backup", 0, "backup")
    n, length = s.network, context.length
    power = anchor.power_kw[:length].copy()
    flow = anchor.delivery_kw[:length].copy()
    power[0], flow[0] = action.power_kw, action.delivery_kw
    # A ramp bridge changes only future actions, leaving the scored action exact.
    for k in range(1, length):
        power[k] = np.clip(power[k], power[k-1]-n.pump_ramp_kw, power[k-1]+n.pump_ramp_kw)
    if length < context.remaining and not np.allclose(power[-1], anchor.power_kw[length-1], atol=1e-8, rtol=0):
        return ContinuationResult("unknown", None, "ramp_bridge_cannot_rejoin_window")
    base_energy, base_charge = _window_energy(s, state, power, flow)
    target = anchor.energy_kwh[length]
    correction = target-base_energy[-1]
    free = np.arange(1, length if length == context.remaining else length-1)
    if not len(free) and np.max(np.abs(correction)) > 1e-7:
        return ContinuationResult("unknown", None, "no_free_future_step_for_exact_action")
    previous = np.vstack([state.previous_power_kw, power[:-1]])
    following = np.vstack([power[1:], anchor.power_kw[length:length+1]
                            if length < context.remaining else power[-1:]])
    available = s.availability[state.t:state.t+length]
    lower = np.maximum(available*n.pump_min_kw, np.maximum(previous-n.pump_ramp_kw, following-n.pump_ramp_kw))
    upper = np.minimum(available*n.pump_max_kw, np.minimum(previous+n.pump_ramp_kw, following+n.pump_ramp_kw))
    increase = correction >= 0
    headroom = np.where(increase[n.pump_to_tank], np.maximum(upper-power, 0), np.maximum(power-lower, 0))
    headroom[0] = 0
    if length < context.remaining:
        headroom[-1] = 0
    # Positive changes reserve shared source/charge margins. Negative changes
    # provide no speculative extra capacity to other tanks.
    room_heat = _sum(headroom*n.pump_cop, n.pump_to_tank, n.n_tanks)
    heat_limit = np.where(increase, np.maximum(n.tank_charge_limit_kw-base_charge, 0), base_charge)
    heat_fraction = np.minimum(1, np.divide(heat_limit, room_heat, out=np.ones_like(room_heat), where=room_heat > 1e-14))
    headroom *= heat_fraction[:, n.pump_to_tank]
    positive_room = headroom*increase[n.pump_to_tank]
    source_room = _sum(positive_room*(n.pump_cop-1), n.pump_station, n.n_stations)
    source_base = _sum(power*(n.pump_cop-1), n.pump_station, n.n_stations)
    source_limit = np.maximum(s.source_limit_kw[state.t:state.t+length]-source_base, 0)
    source_fraction = np.minimum(1, np.divide(source_limit, source_room, out=np.ones_like(source_room), where=source_room > 1e-14))
    headroom *= np.where(increase[n.pump_to_tank], source_fraction[:, n.pump_station], 1)
    room_heat = _sum(headroom*n.pump_cop, n.pump_to_tank, n.n_tanks)
    decay_weights = context.decay[None, :]**(length-1-np.arange(length)[:, None])
    profiles = ("headroom", "early_headroom", "late_headroom")[:max_profiles]
    failures, replay_count = [], 0
    for attempt, profile in enumerate(profiles, 1):
        time_weight = np.ones(length)
        if profile == "early_headroom":
            time_weight = 1/(1+np.arange(length)/max(1, length-1)*3)
        elif profile == "late_headroom":
            time_weight = 1+np.arange(length)/max(1, length-1)*3
        response = s.dt_hours*n.eta_charge*np.sum(decay_weights*room_heat*time_weight[:, None], axis=0)
        if np.any((response <= 1e-14)&(np.abs(correction) > 1e-7)):
            failures.append(profile+": insufficient_future_headroom")
            continue
        fraction = np.divide(correction, response, out=np.zeros_like(correction), where=response > 1e-14)
        trial_power = power+headroom*time_weight[:, None]*fraction[n.pump_to_tank]
        # Preserve exact first-action identity including its floating values.
        trial_power[0] = action.power_kw
        energy, _ = _window_energy(s, state, trial_power, flow)
        if np.max(np.abs(energy[-1]-target)) > 1e-7:
            failures.append(profile+": terminal_roundoff")
            continue
        energy[-1] = target
        local = replace(s, **{name: getattr(s, name)[state.t:state.t+length] for name in _TIME_FIELDS},
                        initial_energy_kwh=state.energy_kwh, initial_power_kw=state.previous_power_kw,
                        terminal_energy_kwh=target)
        local_plan = Plan(0, trial_power, flow, energy)
        replay_count += 1
        replay = validate_plan(local, local_plan, State(0, state.energy_kwh, state.previous_power_kw))
        if not replay.feasible:
            failures.append(profile+": "+"; ".join(replay.violations))
            continue
        plan = Plan(state.t, np.concatenate([trial_power, anchor.power_kw[length:]]),
                    np.concatenate([flow, anchor.delivery_kw[length:]]),
                    np.concatenate([energy, anchor.energy_kwh[length+1:]]))
        for array in (plan.power_kw, plan.delivery_kw, plan.energy_kwh):
            array.setflags(write=False)
        return ContinuationResult("accepted", plan, "independent_window_replay_passed", attempt, profile, replay_count)
    return ContinuationResult("unknown", None, " | ".join(failures), len(profiles), None, replay_count)
