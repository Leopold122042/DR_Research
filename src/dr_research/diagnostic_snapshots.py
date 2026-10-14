"""Frozen-state reconstruction and archived message-mask replay for diagnosis.

No optimization, training, or new closed-loop experiment is performed here.
Recorded future executed actions are observations, never a backup certificate.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from .continuation import prepare_continuation_context, try_continuation
from .control import construct_reference
from .physics import transition, validate_plan
from .types import Action, Plan, State


def select_steps(scenario, seed=42):
    """Select one time per prespecified stratum using time/price alone.

    Scenarios begin Monday at 00:00, as in the weekly experiment. Identical
    time/price arrays and seed therefore select identical times at every scale.
    Selection never observes a method's costs, failures, or candidate scores.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    dt = float(scenario.dt_hours)
    if not np.isfinite(dt) or dt <= 0 or not np.isclose(scenario.horizon*dt, 168):
        raise ValueError("snapshot strata require a full Monday-start seven-day scenario")
    price = np.asarray(scenario.price_yuan_per_kwh, dtype=float)
    if price.shape != (scenario.horizon,) or not np.isfinite(price).all():
        raise ValueError("price must contain one finite value per control step")
    hours = np.arange(scenario.horizon)*dt
    day, hour = np.floor(hours/24).astype(int), hours % 24
    weekday = day < 5
    # A valley action within the final two hours before a price rise.
    near_rise = np.zeros(scenario.horizon, dtype=bool)
    lookahead = max(1, int(np.ceil(2/dt)))
    for t in range(scenario.horizon):
        following = price[t+1:min(scenario.horizon, t+lookahead+1)]
        near_rise[t] = bool(len(following) and np.any(following > price[t]+1e-12))
    strata = (
        ("weekday_valley_before_rise", weekday & np.isclose(price, price.min()) & near_rise),
        ("weekday_high_price", weekday & np.isclose(price, price.max())),
        ("weekend_daytime", (day >= 5) & (hour >= 10) & (hour < 16)),
        ("sunday_late", (day == 6) & (hour >= 20) & (np.arange(scenario.horizon) < scenario.horizon-1)),
    )
    rng = np.random.default_rng(seed)
    result = []
    for category, mask in strata:
        choices = np.flatnonzero(mask)
        if not len(choices):
            raise ValueError(f"No eligible time in prespecified stratum {category}")
        t = int(rng.choice(choices))
        result.append(dict(category=category, t=t, day_index=int(day[t]),
                           hour=float(hour[t]), price_yuan_per_kwh=float(price[t]),
                           eligible_steps=int(len(choices))))
    return result


def _finite_array(arrays, key, shape):
    value = np.asarray(arrays[key])
    if (value.shape != shape or not np.issubdtype(value.dtype, np.number)
            or np.iscomplexobj(value) or not np.isfinite(value).all()):
        raise ValueError(f"Archived {key} must be a finite real array of shape {shape}")
    return value


def _indexed_rows(report, key, horizon):
    rows = report.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"Archive requires {key}")
    indexed = {}
    for row in rows:
        t = row.get("t")
        if type(t) is not int or not 0 <= t < horizon or t in indexed:
            raise ValueError(f"Malformed or duplicate {key} time")
        indexed[t] = row
    return indexed


def _check_boundary(scenario, state, anchor, reference, window):
    end = min(state.t+window, scenario.horizon)
    offset = end-state.t
    if not np.allclose(anchor.energy_kwh[offset], reference.energy_kwh[end], atol=1e-7, rtol=0):
        raise ValueError(f"Backup and static LP reference terminal inventories differ at t={state.t}")
    # The full-episode terminal LP pins inventory only; intermediate windows
    # also pin their final pump power to permit the static reference tail.
    if end < scenario.horizon and not np.allclose(
            anchor.power_kw[offset-1], reference.power_kw[end-1], atol=1e-7, rtol=0):
        raise ValueError(f"Backup and static LP reference terminal powers differ at t={state.t}")


def reconstruct_snapshots(scenario, arrays, report, settings, steps):
    """Rebuild historical certificates using only previously executed actions.

    Every archived state is physically audited. Before each requested action,
    the backup is reconstructed by deterministic continuation of past actions,
    beginning with the original constructive reference. All returned arrays
    refer to immutable-by-convention state/plan objects, not a future trace.
    """
    if not steps or any(type(t) is not int or not 0 <= t < scenario.horizon for t in steps):
        raise ValueError("steps must be a nonempty list of active integer times")
    if len(set(steps)) != len(steps):
        raise ValueError("snapshot times must be unique")
    n, horizon = scenario.network, scenario.horizon
    if (report.get("method") != "gnn_value" or report.get("accepted") is not True
            or report.get("executed_steps") != horizon
            or report.get("scenario_id") != scenario.scenario_id):
        raise ValueError("Archive must be an accepted complete GNN rollout for this scenario")
    ct = settings["control"]
    window = ct["window_steps"]
    safety = ct.get("safety_window_steps", window)
    if safety != window or report.get("window_steps") != window or report.get("safety_window_steps") != safety:
        raise ValueError("Diagnosis requires matching LP, model, and safety window lengths")
    if report.get("candidate_count_requested") != ct["candidate_count"]:
        raise ValueError("Archived candidate count differs from reconstruction settings")
    power = _finite_array(arrays, "power_kw", (horizon, n.n_pumps))
    flow = _finite_array(arrays, "delivery_kw", (horizon, n.n_edges))
    energy = _finite_array(arrays, "energy_kwh", (horizon+1, n.n_tanks))
    times = _finite_array(arrays, "decision_t", (horizon,))
    selected = _finite_array(arrays, "selected_candidate", (horizon,))
    preferred = _finite_array(arrays, "score_preferred_candidate", (horizon,))
    reasons = np.asarray(arrays["execution_reason"])
    if not np.array_equal(times, np.arange(horizon)) or reasons.shape != (horizon,):
        raise ValueError("Archived decisions must cover every control step in order")
    if any(np.any(a != a.astype(np.int64)) for a in (selected, preferred)) or np.any(selected < 0):
        raise ValueError("Archived candidate identifiers must be integers")
    audit = validate_plan(scenario, Plan(0, power, flow, energy))
    if not audit.feasible:
        raise ValueError("Archived executed trajectory failed physical audit: "+"; ".join(audit.violations))
    scoring = _indexed_rows(report, "scoring_steps", horizon)
    sampling = _indexed_rows(report, "sampling_steps", horizon)
    reference = construct_reference(scenario)
    state, anchor = scenario.initial_state(), reference
    snapshots, wanted = {}, set(steps)
    for t in range(max(steps)+1):
        if not np.allclose(state.energy_kwh, energy[t], atol=1e-8, rtol=0):
            raise ValueError(f"Reconstructed inventory differs from archive at t={t}")
        expected_previous = scenario.initial_power_kw if t == 0 else power[t-1]
        if not np.array_equal(state.previous_power_kw, expected_previous):
            raise ValueError(f"Reconstructed previous power differs from archive at t={t}")
        _check_boundary(scenario, state, anchor, reference, window)
        action = Action(power[t].copy(), flow[t].copy())
        if t in wanted:
            if t not in scoring or t not in sampling:
                raise ValueError(f"Requested snapshot lacks archived scoring/sampling at t={t}")
            snapshots[t] = dict(state=State(t, state.energy_kwh.copy(), state.previous_power_kw.copy()),
                                anchor=anchor, reference=reference,
                                archival=dict(selected_candidate_id=int(selected[t]),
                                    preferred_candidate_id=int(preferred[t]),
                                    action=action, execution_reason=str(reasons[t]),
                                    scoring_row=scoring[t], sampling_row=sampling[t],
                                    common_boundary_verified=True,
                                    reconstruction="deterministic_past_action_continuation"))
        context = prepare_continuation_context(scenario, state, anchor, safety)
        continuation = try_continuation(scenario, state, anchor, action, safety, context=context)
        if not continuation.accepted or continuation.plan is None:
            raise ValueError(f"Cannot reconstruct executed continuation at t={t}: {continuation.reason}")
        if int(selected[t]) == 0 and continuation.reason != "certified_backup":
            raise ValueError(f"Archived backup selection differs from reconstructed backup at t={t}")
        if str(reasons[t]).startswith("fallback:") and int(selected[t]) != 0:
            raise ValueError(f"Archived fallback executed a nonbackup identifier at t={t}")
        if not str(reasons[t]).startswith("fallback:") and str(reasons[t]) != continuation.reason:
            raise ValueError(f"Continuation reason differs from archive at t={t}")
        plan = continuation.plan
        if not np.array_equal(plan.power_kw[0], action.power_kw) or not np.array_equal(plan.delivery_kw[0], action.delivery_kw):
            raise ValueError(f"Continuation changed archived current action at t={t}")
        state = transition(scenario, state, action)
        if not np.allclose(state.energy_kwh, energy[t+1], atol=1e-8, rtol=0):
            raise ValueError(f"Reconstructed next inventory differs from archive at t={t}")
        anchor = Plan(t+1, plan.power_kw[1:], plan.delivery_kw[1:], plan.energy_kwh[1:])
    return snapshots


def replay_scores(model, scenario, state, pool, scoring_row):
    """Replay archived adaptive mask and independently evaluate the full GNN.

    Groups are applied in their original order, preserving cache roundoff and
    the historical scheduling decisions. This does not rerun a wall-clock
    scheduler; returned timings are diagnostic replay timings only.
    """
    import torch
    if scoring_row.get("t") != state.t or scoring_row.get("mode") not in {"adaptive", "random", "coarse"}:
        raise ValueError("Archived score row must match time and a cache-based scoring mode")
    actions = pool.actions if hasattr(pool, "actions") else list(pool)
    total = len(actions)*scenario.network.n_pumps
    if scoring_row.get("total_messages") != total:
        raise ValueError("Archived score row has different pool dimensions")
    groups = scoring_row.get("selected_groups")
    if not isinstance(groups, list) or scoring_row.get("rounds") != len(groups):
        raise ValueError("Archived refinement rounds do not match selected groups")
    seen = set()
    for i, group in enumerate(groups):
        positions = group.get("positions")
        if (group.get("round") != i or not isinstance(positions, list) or not positions
                or any(type(p) is not int or not 0 <= p < total for p in positions)
                or len(set(positions)) != len(positions) or seen.intersection(positions)
                or group.get("new_messages") != len(positions)):
            raise ValueError("Archived refinement positions/rounds contain invalid or repeated entries")
        seen.update(positions)
    if scoring_row.get("refined_messages") != len(seen):
        raise ValueError("Archived refined-message count differs from its mask")
    clock = perf_counter()
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prepared = model.prepare(scenario, state, actions)
            prepare_s = perf_counter()-clock
            clock = perf_counter()
            cache = model.start_cache(prepared)
            coarse_s = perf_counter()-clock
            clock = perf_counter()
            for group in groups:
                stats = model.refine(cache, group["positions"])
                if stats["new_messages"] != group["new_messages"]:
                    raise ValueError("Replayed refinement changed archived message count")
                # Historical scheduling reads scores after every selected group.
                model.score(cache)
            adaptive = (model.score(cache)+prepared.current_yuan).detach().cpu().numpy().astype(float, copy=True)
            refinement_s = perf_counter()-clock
            clock = perf_counter()
            full = (model.full(prepared)+prepared.current_yuan).detach().cpu().numpy().astype(float, copy=True)
            full_s = perf_counter()-clock
    finally:
        model.train(was_training)
    if any(x.shape != (len(actions),) or not np.isfinite(x).all() for x in (full, adaptive)):
        raise ValueError("Replayed GNN scores are malformed or nonfinite")
    return full, adaptive, dict(prepare_s=prepare_s, coarse_s=coarse_s,
        archived_mask_refinement_s=refinement_s, full_scoring_s=full_s,
        refined_messages=len(seen), rounds=len(groups),
        timing_scope="diagnostic_replay_only_shared_prepare_and_projection_cache",
        archived_scoring_time_s=scoring_row.get("scoring_time_s"))
