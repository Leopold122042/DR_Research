"""Paired LP-MPC and budgeted candidate-value GNN rolling control."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from .candidates import generate_candidates
from .continuation import prepare_continuation_context, try_continuation
from .control import construct_reference, solve_control_window, stitch_reference_tail
from .experiment_config import seed_for
from .physics import transition, validate_action, validate_plan
from .types import Plan, Scenario

METHODS = ("lp_mpc", "gnn_value")
_CONTROL_ERRORS = (ValueError, TypeError, RuntimeError, FloatingPointError,
                   AttributeError, IndexError, OverflowError)
_SCORE_TIMES = ("prepare_s", "coarse_scoring_s", "error_prediction_s", "sensitivity_s",
                "scheduling_s", "refinement_s", "full_scoring_s")


def evaluate_candidates(*args, **kwargs):
    """Keep the pure LP-MPC import path independent of the learning runtime."""
    from .scheduler import evaluate_candidates as evaluate
    return evaluate(*args, **kwargs)


def run_controller(s: Scenario, settings: dict, method: str, model=None, *,
                   model_load_s: float = 0., error_predictor=None, latency_model=None,
                   gnn_mode=None, progress=None) -> tuple[dict, dict]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if (method == "gnn_value") != (model is not None):
        raise ValueError("Supply exactly one trained model for gnn_value")
    if not np.isfinite(model_load_s) or model_load_s < 0:
        raise ValueError("model_load_s must be finite and nonnegative")
    ct, rf = settings["control"], settings.get("refinement", {})
    safety_window = ct.get("safety_window_steps", ct["window_steps"])
    if type(safety_window) is not int or not 2 <= safety_window <= ct["window_steps"]:
        raise ValueError("safety_window_steps must be between 2 and window_steps")
    if model is not None and getattr(model, "window_steps", ct["window_steps"]) != ct["window_steps"]:
        raise ValueError("Model forecast window must match control.window_steps")
    max_attempts = ct.get("max_continuation_attempts", 8)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_continuation_attempts must be positive")
    mode = gnn_mode or rf.get("mode", "full")
    if method == "gnn_value" and mode not in ("coarse", "full", "random", "adaptive"):
        raise ValueError("Unknown GNN refinement mode")
    if error_predictor is None:
        error_predictor = getattr(model, "error_predictor", None)
    if latency_model is None:
        latency_model = getattr(model, "latency_model", None)
    clock = perf_counter()
    reference = construct_reference(s)
    anchor, state = reference, s.initial_state()
    prepare_s = perf_counter()-clock+model_load_s
    report = dict(method=method, gnn_mode=mode if model is not None else None,
                  scenario_id=s.scenario_id, accepted=False, status="running",
                  total_electricity_yuan=None, prepare_time_s=prepare_s,
                  candidate_generation_time_s=0., inference_time_s=0.,
                  candidate_verification_time_s=0., optimization_time_s=0.,
                  execution_check_time_s=0., solver_calls=0, fallback_count=0,
                  fallback_reasons=[], deadline_misses=0, anchor_action_count=0,
                  anchor_selected_count=0, continuation_attempt_count=0,
                  continuation_unknown_count=0, continuation_unknowns=[],
                  window_replay_count=0, current_candidate_rejection_count=0,
                  candidate_duplicate_count=0, candidate_raw_count=0,
                  raw_feasible_count=0, decision_attempts=0, executed_steps=0,
                  window_steps=ct["window_steps"],
                  safety_window_steps=safety_window if model is not None else None,
                  candidate_count_requested=ct["candidate_count"] if model is not None else 0,
                  candidate_budget_includes_backup=False,
                  refined_messages=0, total_messages=0, refinement_rounds=0,
                  scoring_budget_exhausted_count=0, scoring_overrun_s=0.,
                  recalibration_required_count=0, sampling_steps=[], scoring_steps=[], solver_steps=[])
    for name in _SCORE_TIMES:
        report["score_"+name] = 0.
    powers, flows, energies = [], [], [state.energy_kwh.copy()]
    costs, latencies, pool_sizes, selected_ids, preferred_ids = [], [], [], [], []
    decision_steps, reasons, continuation_counts, replay_counts = [], [], [], []
    # State-dependent maps change every step; scrambled coordinates do not.
    sobol_seed = seed_for(settings["master_seed"], "sobol", s.seed)
    for step in range(s.horizon):
        started = perf_counter()
        next_plan, selected, preferred, pool_size = anchor, -1, -1, 0
        attempts, replays, failure = 0, 0, None
        continuation_reason = "not_applicable"
        first_step_preparation = prepare_s if step == 0 else 0.

        def remaining_time():
            return ct["time_limit_s"]-(perf_counter()-started)-first_step_preparation

        def prepare_action(plan):
            clock = perf_counter()
            try:
                action = plan.action(0)
                checked = validate_action(s, state, action)
                if not checked.feasible:
                    raise ValueError("action rejected: "+"; ".join(checked.violations))
                if plan.start_t != state.t or not np.allclose(plan.energy_kwh[0], state.energy_kwh, atol=1e-5, rtol=0):
                    raise ValueError("continuation does not match actual state")
                identical = (np.array_equal(action.power_kw, anchor.power_kw[0])
                             and np.array_equal(action.delivery_kw, anchor.delivery_kw[0]))
                shifted = Plan(state.t+1, plan.power_kw[1:], plan.delivery_kw[1:], plan.energy_kwh[1:])
                return action, checked, identical, shifted
            finally:
                report["execution_check_time_s"] += perf_counter()-clock

        try:
            if remaining_time() <= 0:
                raise RuntimeError("preparation exhausted decision budget")
            if method == "lp_mpc":
                clock = perf_counter()
                report["solver_calls"] += 1
                try:
                    result = solve_control_window(s, state, reference, window_steps=ct["window_steps"],
                                                  time_limit_s=remaining_time(), solver_method=ct["solver_method"],
                                                  solver_threads=ct.get("solver_threads", 1), aggregate_pumps=True)
                    report["solver_steps"].append(dict(t=step, status=result.status,
                        statistics=getattr(result, "statistics", {}),
                        timings_s=getattr(result, "timings_s", {}),
                        electricity_yuan=getattr(result, "electricity_yuan", None)))
                    if result.plan is None or result.status not in {"optimal", "limit"}:
                        raise RuntimeError(f"MPC {result.status}: {result.message}")
                    next_plan = stitch_reference_tail(s, state, result.plan, reference)
                finally:
                    report["optimization_time_s"] += perf_counter()-clock
            else:
                clock = perf_counter()
                try:
                    pool = generate_candidates(s, state, anchor, ct["candidate_count"], sobol_seed)
                finally:
                    report["candidate_generation_time_s"] += perf_counter()-clock
                pool_size = len(pool)
                report["sampling_steps"].append(dict(t=step, **pool.stats))
                report["candidate_raw_count"] += pool.stats["raw_count"]
                report["current_candidate_rejection_count"] += pool.stats["rejected_count"]
                report["candidate_duplicate_count"] += pool.stats["duplicate_count"]
                report["raw_feasible_count"] += pool.stats["accepted_raw_count"]
                clock = perf_counter()
                try:
                    acceptance_reserve = max(.01, float(rf.get("reserve_s", .001)))
                    scoring_budget = min(float(rf.get("time_budget_s", 1.)), remaining_time()-acceptance_reserve)
                    if scoring_budget <= 0:
                        raise RuntimeError("candidate generation exhausted scoring budget")
                    scores, scoring = evaluate_candidates(model, s, state, pool.actions, mode=mode,
                        error_predictor=error_predictor, latency_model=latency_model,
                        time_budget_s=scoring_budget, seed=seed_for(settings["master_seed"], "refinement", s.seed, step),
                        shortlist_size=rf.get("shortlist_size", 8), coverage_count=rf.get("coverage_count", 2),
                        group_size=rf.get("group_size", 64), max_rounds=rf.get("max_rounds", 16),
                        reserve_s=rf.get("reserve_s", .001))
                    scores = np.asarray(scores, dtype=float)
                    if scores.shape != (pool_size,) or not np.isfinite(scores).all():
                        raise ValueError("invalid candidate values")
                    report["scoring_steps"].append(dict(t=step, **scoring))
                    for name in _SCORE_TIMES:
                        report["score_"+name] += float(scoring.get(name, 0.))
                    for name in ("refined_messages", "total_messages"):
                        report[name] += int(scoring.get(name, 0))
                    report["refinement_rounds"] += int(scoring.get("rounds", 0))
                    report["scoring_budget_exhausted_count"] += int(scoring.get("budget_exhausted", False))
                    report["scoring_overrun_s"] += float(scoring.get("overrun_s", 0.))
                    report["recalibration_required_count"] += int(scoring.get("recalibration_required", False))
                    order = np.argsort(scores, kind="stable")
                    preferred = pool[int(order[0])].candidate_id
                finally:
                    report["inference_time_s"] += perf_counter()-clock
                clock = perf_counter()
                try:
                    context = prepare_continuation_context(s, state, anchor, safety_window)
                    accepted = None
                    for index in order[:max_attempts]:
                        if remaining_time() <= 0:
                            raise RuntimeError("continuation exhausted decision budget")
                        candidate = pool[int(index)]
                        attempts += 1
                        result = try_continuation(s, state, anchor, candidate, safety_window, context=context)
                        replays += result.replay_count
                        if not result.accepted:
                            report["continuation_unknown_count"] += 1
                            report["continuation_unknowns"].append(dict(t=step, candidate_id=candidate.candidate_id, reason=result.reason))
                            continue
                        if (result.plan is None or not np.array_equal(result.plan.power_kw[0], candidate.action.power_kw)
                                or not np.array_equal(result.plan.delivery_kw[0], candidate.action.delivery_kw)):
                            raise ValueError("continuation changed the scored current action")
                        accepted, selected = result, candidate.candidate_id
                        break
                    if accepted is None:
                        raise RuntimeError("bounded continuation search returned unknown for every attempted candidate")
                    next_plan, continuation_reason = accepted.plan, accepted.reason
                    report["anchor_selected_count"] += int(selected == 0)
                finally:
                    report["candidate_verification_time_s"] += perf_counter()-clock
                    report["continuation_attempt_count"] += attempts
                    report["window_replay_count"] += replays
            prepared = prepare_action(next_plan)
            if remaining_time() < 0:
                raise RuntimeError("end-to-end decision deadline exceeded")
        except _CONTROL_ERRORS as exc:
            failure = f"{type(exc).__name__}: {exc}"
            next_plan, selected = anchor, 0 if method == "gnn_value" else -1
            report["fallback_count"] += 1
            report["fallback_reasons"].append(dict(t=step, reason=failure))
            try:
                prepared = prepare_action(anchor)
            except _CONTROL_ERRORS as anchor_exc:
                report.update(status="invalid_continuation", message=str(anchor_exc))
                prepared = None
        decision_steps.append(step)
        reasons.append("fallback:"+failure if failure else continuation_reason)
        report["decision_attempts"] += 1
        latency = perf_counter()-started
        latencies.append(latency)
        report["deadline_misses"] += int(latency+first_step_preparation > ct["time_limit_s"])
        if prepared is None:
            break
        action, checked, identical, shifted = prepared
        report["anchor_action_count"] += int(identical)
        anchor = shifted
        state = transition(s, state, action)
        powers.append(action.power_kw)
        flows.append(action.delivery_kw)
        energies.append(state.energy_kwh.copy())
        costs.append(float(checked.electricity_yuan))
        pool_sizes.append(pool_size)
        selected_ids.append(selected)
        preferred_ids.append(preferred)
        continuation_counts.append(attempts)
        replay_counts.append(replays)
        report["executed_steps"] = step+1
        if progress and ((step+1) % 6 == 0 or step+1 == s.horizon):
            progress(dict(phase="rollout", method=method, gnn_mode=report["gnn_mode"],
                          step=step+1, total_steps=s.horizon,
                          day=(step+1)/48, days=s.horizon/48, scenario=s.scenario_id,
                          electricity_yuan_so_far=float(sum(costs)),
                          control_time_s_so_far=prepare_s+sum(latencies),
                          fallback_count=report["fallback_count"]))
    public_latency = np.asarray(latencies)
    if len(public_latency):
        public_latency[0] += prepare_s
    report.update(control_time_s=prepare_s+sum(latencies),
                  latency_mean_s=float(np.mean(public_latency)) if len(public_latency) else None,
                  latency_p95_s=float(np.quantile(public_latency, .95)) if len(public_latency) else None,
                  latency_max_s=float(np.max(public_latency)) if len(public_latency) else None,
                  candidate_pool_mean=float(np.mean(pool_sizes)) if pool_sizes else None,
                  refinement_fraction=report["refined_messages"]/max(1, report["total_messages"]))
    if len(powers) == s.horizon:
        clock = perf_counter()
        audit = validate_plan(s, Plan(0, np.asarray(powers), np.asarray(flows), np.asarray(energies)))
        report.update(audit_time_s=perf_counter()-clock, maximum_residuals=audit.max_residuals,
                      accepted=audit.feasible,
                      status=("completed_with_fallback" if report["fallback_count"] else "completed") if audit.feasible else "audit_failed",
                      total_electricity_yuan=float(sum(costs)) if audit.feasible else None)
    traces = dict(power_kw=np.asarray(powers), delivery_kw=np.asarray(flows), energy_kwh=np.asarray(energies),
                  electricity_yuan=np.asarray(costs), control_latency_s=public_latency,
                  candidate_pool_size=np.asarray(pool_sizes), selected_candidate=np.asarray(selected_ids),
                  score_preferred_candidate=np.asarray(preferred_ids),
                  continuation_attempts=np.asarray(continuation_counts), decision_t=np.asarray(decision_steps),
                  candidate_window_replays=np.asarray(replay_counts),
                  execution_reason=np.asarray(reasons))
    return report, traces
