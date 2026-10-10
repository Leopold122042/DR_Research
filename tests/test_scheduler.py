"""Only selected high-fidelity messages are computed within measured budgets."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dr_research import scheduler


class ConstantLatency:
    def __init__(self, seconds=.001, inside=True):
        self.seconds, self.inside = seconds, inside

    def predict(self, features):
        return self.seconds

    def in_profile(self, features):
        return self.inside


class ToyModel:
    def __init__(self, clock=None, *, gradient_s=0., refinement_s=0.):
        self.computed = []
        self.clock, self.gradient_s, self.refinement_s = clock, gradient_s, refinement_s

    def prepare(self, scenario, state, actions):
        return SimpleNamespace(candidate_count=len(actions), n_pumps=4, n_tanks=2,
                               current_yuan=torch.zeros(len(actions)),
                               receiver_index=torch.tensor([0, 0, 1, 1]),
                               full_projection_cache_ready=False)

    def start_cache(self, prepared):
        return SimpleNamespace(prepared=prepared,
                               messages=torch.zeros(prepared.candidate_count, 4, 1),
                               refined=torch.zeros(prepared.candidate_count, 4, dtype=torch.bool))

    def score(self, cache):
        return 10.-cache.messages.sum((1, 2))

    def full(self, prepared):
        self.computed.extend(range(prepared.candidate_count*4))
        return torch.full((prepared.candidate_count,), 6.)

    def error_features(self, cache):
        return torch.ones(cache.prepared.candidate_count, 4, 2)

    def sensitivity(self, cache, candidate_ids):
        if self.clock is not None:
            self.clock[0] += self.gradient_s
        result = torch.zeros_like(cache.refined, dtype=torch.float32)
        result[candidate_ids] = 1.
        return result

    def refine(self, cache, positions):
        if self.clock is not None:
            self.clock[0] += self.refinement_s
        fresh = [int(p) for p in positions if not cache.refined.reshape(-1)[int(p)]]
        self.computed.extend(fresh)
        cache.messages.reshape(-1)[fresh] = 1.
        cache.refined.reshape(-1)[fresh] = True
        cache.prepared.full_projection_cache_ready = True
        return dict(new_messages=len(fresh))


def unit_errors(features):
    return torch.ones(features.shape[:-1])


def test_coarse_evaluation_computes_no_complete_messages():
    model = ToyModel()
    scores, report = scheduler.evaluate_candidates(model, None, None, [0, 1], mode="coarse")
    np.testing.assert_array_equal(scores, [10., 10.])
    assert model.computed == [] and report["refinement_fraction"] == 0


def test_adaptive_only_evaluates_selected_unique_positions():
    model = ToyModel()
    scores, report = scheduler.evaluate_candidates(model, None, None, [0, 1], mode="adaptive",
        error_predictor=unit_errors, latency_model=ConstantLatency(), time_budget_s=10.,
        max_rounds=2, group_size=2, reserve_s=0.)
    selected = [p for row in report["selected_groups"] for p in row["positions"]]
    assert model.computed == selected and len(set(selected)) == len(selected) == 4
    assert report["refined_messages"] == 4 and report["refinement_fraction"] == .5
    assert np.min(scores) < 10.


def test_random_and_full_are_explicit_ablations_of_same_score():
    full, full_report = scheduler.evaluate_candidates(ToyModel(), None, None, [0], mode="full")
    random, random_report = scheduler.evaluate_candidates(ToyModel(), None, None, [0], mode="random",
        latency_model=ConstantLatency(), time_budget_s=10., max_rounds=8, group_size=2)
    np.testing.assert_array_equal(full, random)
    assert full_report["refinement_fraction"] == random_report["refinement_fraction"] == 1.


def test_gradient_overhead_can_exhaust_budget_before_any_refinement(monkeypatch):
    now = [0.]
    monkeypatch.setattr(scheduler, "perf_counter", lambda: now[0])
    model = ToyModel(now, gradient_s=.95)
    _, report = scheduler.evaluate_candidates(model, None, None, [0], mode="adaptive",
        error_predictor=unit_errors, latency_model=ConstantLatency(.1),
        time_budget_s=1., reserve_s=.01)
    assert report["sensitivity_s"] == .95 and report["refined_messages"] == 0
    assert model.computed == []


def test_underpredicted_batch_overrun_is_reported_and_no_next_group_runs(monkeypatch):
    now = [0.]
    monkeypatch.setattr(scheduler, "perf_counter", lambda: now[0])
    model = ToyModel(now, refinement_s=2.)
    _, report = scheduler.evaluate_candidates(model, None, None, [0], mode="adaptive",
        error_predictor=unit_errors, latency_model=ConstantLatency(.1),
        time_budget_s=1., reserve_s=0., group_size=1)
    assert report["rounds"] == 1 and report["overrun_s"] == 1.
    assert report["scoring_time_s"] == 2. and report["budget_exhausted"]


def test_unprofiled_workloads_request_recalibration_without_hidden_full_work():
    model = ToyModel()
    _, report = scheduler.evaluate_candidates(model, None, None, [0], mode="random",
        latency_model=ConstantLatency(inside=False), time_budget_s=10.)
    assert model.computed == [] and report["recalibration_required"]
    assert report["out_of_profile_groups"] > 0


def test_shortlist_includes_leader_and_independent_coverage():
    picked = scheduler.shortlist([5, 1, 3, 8, 4], leader_count=2, coverage_count=2,
                                 rng=np.random.default_rng(42))
    np.testing.assert_array_equal(picked[:2], [1, 2])
    assert len(set(picked)) == 4


def test_error_predictor_roundtrip_and_heldout_state_report():
    x = torch.linspace(0, 1, 20).reshape(10, 2)
    y = torch.linspace(.1, 1, 10)
    model, report = scheduler.fit_error_predictor(x, y, validation_features=x[:3],
        validation_norms=y[:3], epochs=3, hidden_dim=4)
    restored = scheduler.MessageErrorPredictor(**model.architecture)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x), restored(x))
    assert torch.isfinite(model(x)).all() and (model(x) >= 0).all()
    assert report["split"] == "caller_supplied_held_out_states"


def test_real_message_model_calibration_and_full_refinement_match(monkeypatch):
    from dr_research.gnn import CandidateValueGNN
    from helpers import small_case
    scenario, anchor = small_case()
    model = CandidateValueGNN(window_steps=4, hidden_dim=8, latent_dim=4).eval()
    actions = [anchor.action(0), anchor.action(0)]
    with torch.no_grad():
        prepared = model.prepare(scenario, scenario.initial_state(), actions)
    features, targets = scheduler.paired_error_examples(model, [prepared])
    assert features.shape == (8, model.error_feature_dim) and targets.shape == (8,)
    profile, report = scheduler.calibrate_latency(model, [prepared],
        group_sizes=(1, 2, 4), repeats=1, warmup=0)
    assert report["device"] == "cpu" and report["validation_batches"] > 0
    assert {row["features"][1] for row in report["measured_batches"]} == {0., 10.}
    with torch.no_grad():
        exact = model.full(prepared).numpy()+prepared.current_yuan.numpy()
    calls = []
    original = model.complete_messages

    def trace_selected(pool, positions=None):
        assert positions is not None, "online adaptive schedule requested unselected complete messages"
        calls.extend(np.asarray(positions).reshape(-1).tolist())
        return original(pool, positions)

    monkeypatch.setattr(model, "complete_messages", trace_selected)
    scores, actual = scheduler.evaluate_candidates(model, scenario, scenario.initial_state(), actions,
        mode="adaptive", error_predictor=unit_errors, latency_model=profile,
        time_budget_s=10., group_size=2, max_rounds=8)
    assert actual["refined_messages"] == 8 and len(calls) == len(set(calls)) == 8
    np.testing.assert_allclose(scores, exact, atol=1e-4, rtol=1e-6)
    too_many = model.prepare(scenario, scenario.initial_state(), actions+[actions[0]])
    assert "candidate_pool_exceeds_profile" in scheduler.profile_compatibility(model, too_many, profile)
    profile.calibration["torch_threads"] += 1
    _, blocked = scheduler.evaluate_candidates(model, scenario, scenario.initial_state(), actions,
        mode="adaptive", error_predictor=unit_errors, latency_model=profile, time_budget_s=10.)
    assert "changed_torch_threads" in blocked["profile_mismatches"]
    assert blocked["refined_messages"] == 0 and blocked["recalibration_required"]
