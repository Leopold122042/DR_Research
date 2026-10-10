"""Controller timing, exact-action acceptance, bounded search, and failures."""
import json
from dataclasses import replace
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from dr_research import dispatch
from dr_research.continuation import ContinuationResult
from dr_research.types import ValidationReport
from helpers import small_case


def settings(time_limit=100.):
    return {"master_seed": 42, "control": {"window_steps": 4,
            "candidate_count": 16, "time_limit_s": time_limit,
            "max_continuation_attempts": 8, "solver_method": "highs-ipm"},
            "refinement": {"mode": "full", "time_budget_s": 1.}}


def install_case(monkeypatch, horizon=4):
    s, anchor = small_case(horizon)
    monkeypatch.setattr(dispatch, "construct_reference", lambda _: anchor)
    def score(model, scenario, state, actions, **kwargs):
        return model.predict_cost(scenario, state, actions), {}
    monkeypatch.setattr(dispatch, "evaluate_candidates", score)
    return s, anchor


def forbid_lp(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A learned online controller invoked an optimizer")
    monkeypatch.setattr(dispatch, "solve_control_window", forbidden)
    monkeypatch.setattr("dr_research.control.linprog", forbidden)
    monkeypatch.setattr("scipy.optimize.linprog", forbidden)


class ValueModel:
    def __init__(self, invalid=None, prefer_nonbackup=False):
        self.invalid, self.prefer_nonbackup = invalid, prefer_nonbackup

    def predict_cost(self, scenario, state, actions):
        values = np.arange(len(actions), dtype=float)
        if self.prefer_nonbackup:
            values[0] = 1e6
        if self.invalid == "nan":
            values[:] = np.nan
        elif self.invalid == "shape":
            values = values[:-1]
        return values


def assert_cost_and_time(s, report, traces):
    assert report["accepted"] and report["executed_steps"] == s.horizon
    expected = s.dt_hours*np.dot(s.price_yuan_per_kwh, traces["power_kw"].sum(axis=1))
    assert report["total_electricity_yuan"] == pytest.approx(expected)
    assert report["control_time_s"] == pytest.approx(traces["control_latency_s"].sum())
    assert report["latency_max_s"] == pytest.approx(traces["control_latency_s"].max())
    json.dumps(report, allow_nan=False)


def test_comparison_contains_only_lp_mpc_and_candidate_value():
    assert dispatch.METHODS == ("lp_mpc", "gnn_value")


@pytest.mark.parametrize("invalid", ["nan", "shape"])
def test_invalid_scores_use_backup_and_record_its_executed_id(monkeypatch, invalid):
    s, anchor = install_case(monkeypatch)
    forbid_lp(monkeypatch)
    report, traces = dispatch.run_controller(s, settings(), "gnn_value", ValueModel(invalid))
    assert_cost_and_time(s, report, traces)
    np.testing.assert_array_equal(traces["power_kw"], anchor.power_kw)
    assert report["fallback_count"] == s.horizon
    assert (traces["selected_candidate"] == 0).all()
    assert all(x.startswith("fallback:") for x in traces["execution_reason"])


def test_backup_choice_is_not_failure_and_raw_budget_excludes_backup(monkeypatch):
    s, anchor = install_case(monkeypatch)
    forbid_lp(monkeypatch)
    report, traces = dispatch.run_controller(s, settings(), "gnn_value", ValueModel())
    assert_cost_and_time(s, report, traces)
    assert report["solver_calls"] == report["fallback_count"] == 0
    assert report["anchor_selected_count"] == report["anchor_action_count"] == s.horizon
    assert (traces["selected_candidate"] == 0).all()
    assert (traces["execution_reason"] == "certified_backup").all()
    assert report["candidate_raw_count"] == 16*s.horizon
    assert not report["candidate_budget_includes_backup"]


def test_score_before_continuation_and_unknown_winner_is_skipped(monkeypatch):
    s, _ = install_case(monkeypatch, horizon=8)
    forbid_lp(monkeypatch)
    scored, rejected = set(), []
    original = dispatch.try_continuation
    class Model(ValueModel):
        def predict_cost(self, scenario, state, actions):
            scored.add(state.t)
            return super().predict_cost(scenario, state, actions)
    def verify(scenario, state, anchor, candidate, window, **kwargs):
        assert state.t in scored
        if state.t == 0 and not rejected:
            rejected.append(candidate.candidate_id)
            return ContinuationResult("unknown", None, "injected unknown")
        return original(scenario, state, anchor, candidate, window, **kwargs)
    monkeypatch.setattr(dispatch, "try_continuation", verify)
    report, trace = dispatch.run_controller(s, settings(), "gnn_value", Model(prefer_nonbackup=True))
    assert_cost_and_time(s, report, trace)
    assert rejected and report["continuation_unknown_count"] >= 1
    assert trace["score_preferred_candidate"][0] == rejected[0]
    assert trace["selected_candidate"][0] != rejected[0]
    assert report["window_replay_count"] == int(trace["candidate_window_replays"].sum())


def test_continuation_cannot_silently_execute_an_unscored_shrunk_action(monkeypatch):
    s, anchor = install_case(monkeypatch)
    forbid_lp(monkeypatch)
    def corrupt(scenario, state, current_anchor, candidate, window, **kwargs):
        return ContinuationResult("accepted", current_anchor, "fabricated certificate")
    monkeypatch.setattr(dispatch, "try_continuation", corrupt)
    report, trace = dispatch.run_controller(s, settings(), "gnn_value", ValueModel(prefer_nonbackup=True))
    assert_cost_and_time(s, report, trace)
    assert any("changed the scored" in item["reason"] for item in report["fallback_reasons"])
    np.testing.assert_array_equal(trace["power_kw"], anchor.power_kw)


def test_bounded_unknown_search_executes_backup_and_never_claims_infeasible(monkeypatch):
    s, anchor = install_case(monkeypatch)
    monkeypatch.setattr(dispatch, "try_continuation", lambda *a, **k: ContinuationResult("unknown", None, "bounded repair exhausted"))
    cfg = settings()
    cfg["control"]["max_continuation_attempts"] = 2
    report, trace = dispatch.run_controller(s, cfg, "gnn_value", ValueModel(prefer_nonbackup=True))
    assert_cost_and_time(s, report, trace)
    assert report["fallback_count"] == s.horizon
    assert np.max(trace["continuation_attempts"]) <= 2
    assert report["current_candidate_rejection_count"] == 0
    assert report["continuation_unknown_count"] == int(trace["continuation_attempts"].sum())
    np.testing.assert_array_equal(trace["power_kw"], anchor.power_kw)


def test_deadline_includes_final_validation_and_fallback_checks(monkeypatch):
    s, anchor = install_case(monkeypatch)
    forbid_lp(monkeypatch)
    now = [0.]
    monkeypatch.setattr(dispatch, "perf_counter", lambda: now[0])
    original = dispatch.validate_action
    def slow_validation(*args):
        now[0] += 1.
        return original(*args)
    monkeypatch.setattr(dispatch, "validate_action", slow_validation)
    report, traces = dispatch.run_controller(s, settings(.75), "gnn_value", ValueModel(), model_load_s=.25)
    assert_cost_and_time(s, report, traces)
    np.testing.assert_allclose(traces["control_latency_s"], [2.25, 2., 2., 2.])
    assert report["fallback_count"] == report["deadline_misses"] == s.horizon
    assert report["control_time_s"] == 8.25


def test_failed_backup_attempt_contributes_to_latency(monkeypatch):
    s, _ = install_case(monkeypatch)
    now = [0.]
    monkeypatch.setattr(dispatch, "perf_counter", lambda: now[0])
    class FailingModel:
        def predict_cost(self, scenario, state, actions):
            now[0] += 2.
            raise RuntimeError("model failed")
    def reject(*args):
        now[0] += 3.
        return ValidationReport(False, ("invalid test anchor",), {})
    monkeypatch.setattr(dispatch, "validate_action", reject)
    report, traces = dispatch.run_controller(s, settings(1.), "gnn_value", FailingModel(), model_load_s=.25)
    assert not report["accepted"] and report["status"] == "invalid_continuation"
    assert report["executed_steps"] == 0 and report["decision_attempts"] == 1
    assert report["inference_time_s"] == 2. and report["execution_check_time_s"] == 3.
    assert report["control_time_s"] == report["latency_max_s"] == 5.25
    np.testing.assert_array_equal(traces["control_latency_s"], [5.25])


def test_preparation_exhaustion_does_not_count_a_solver_call(monkeypatch):
    s, _ = install_case(monkeypatch, horizon=1)
    forbid_lp(monkeypatch)
    report, traces = dispatch.run_controller(s, settings(1.), "lp_mpc", model_load_s=2.)
    assert_cost_and_time(s, report, traces)
    assert report["solver_calls"] == 0 and report["fallback_count"] == 1


def test_real_full_and_coarse_model_modes_complete_a_short_rollout(monkeypatch):
    torch = pytest.importorskip("torch")
    from dr_research.gnn import CandidateValueGNN
    torch.set_num_threads(1)
    s, anchor = small_case(8)
    monkeypatch.setattr(dispatch, "construct_reference", lambda _: anchor)
    forbid_lp(monkeypatch)
    model = CandidateValueGNN(window_steps=4, hidden_dim=8, latent_dim=4, coarse_width=4)
    for mode in ("coarse", "full"):
        report, traces = dispatch.run_controller(s, settings(), "gnn_value", model, gnn_mode=mode)
        assert_cost_and_time(s, report, traces)
        assert report["gnn_mode"] == mode
        assert report["inference_time_s"] > 0 and report["score_prepare_s"] > 0


def test_prediction_and_safety_windows_are_separate(monkeypatch):
    s, _ = install_case(monkeypatch, horizon=8)
    cfg = settings()
    cfg["control"].update(window_steps=8, safety_window_steps=4)
    report, traces = dispatch.run_controller(s, cfg, "gnn_value", ValueModel())
    assert_cost_and_time(s, report, traces)
    assert report["window_steps"] == 8 and report["safety_window_steps"] == 4


def test_runtime_prediction_window_must_match_the_trained_model(monkeypatch):
    s, _ = install_case(monkeypatch)
    model = ValueModel()
    model.window_steps = 12
    with pytest.raises(ValueError, match="forecast window"):
        dispatch.run_controller(s, settings(), "gnn_value", model)


@pytest.mark.parametrize("bad", [True, 0, 1, 49, 12.5])
def test_invalid_safety_window_rejected(bad, monkeypatch):
    s, _ = install_case(monkeypatch)
    cfg = settings()
    cfg["control"]["safety_window_steps"] = bad
    with pytest.raises(ValueError, match="safety_window"):
        dispatch.run_controller(s, cfg, "gnn_value", ValueModel())
