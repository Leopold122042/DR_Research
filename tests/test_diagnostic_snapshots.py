"""Historical snapshots must retain the certificate known at decision time."""
from dataclasses import replace
import copy
import numpy as np
import pytest
import torch

from dr_research.candidates import generate_candidates
from dr_research.continuation import try_continuation
from dr_research.diagnostic_snapshots import reconstruct_snapshots, replay_scores, select_steps
from dr_research.gnn import CandidateValueGNN
from dr_research.physics import transition
from dr_research.types import Action, Plan
from helpers import small_case


def _historical_rollout(monkeypatch):
    s, reference = small_case(16)
    monkeypatch.setattr("dr_research.diagnostic_snapshots.construct_reference", lambda s: reference)
    state, anchor = s.initial_state(), reference
    power, flow, energy = [], [], [state.energy_kwh.copy()]
    selected, reasons, original_snapshots = [], [], {}
    for t in range(s.horizon):
        original_snapshots[t] = (state, anchor)
        # Deliberately make later executed actions differ from the certificate
        # retained at t=0: future recorded execution must not become that anchor.
        action = anchor.action(0)
        candidate_id = 0
        if t in (0, 2):
            action = Action(action.power_kw+.2, action.delivery_kw.copy())
            candidate_id = 1
        accepted = try_continuation(s, state, anchor, action, 6)
        assert accepted.accepted
        selected.append(candidate_id)
        reasons.append(accepted.reason)
        power.append(action.power_kw)
        flow.append(action.delivery_kw)
        state = transition(s, state, action)
        energy.append(state.energy_kwh.copy())
        plan = accepted.plan
        anchor = Plan(t+1, plan.power_kw[1:], plan.delivery_kw[1:], plan.energy_kwh[1:])
    arrays = dict(power_kw=np.asarray(power), delivery_kw=np.asarray(flow),
        energy_kwh=np.asarray(energy), decision_t=np.arange(s.horizon),
        selected_candidate=np.asarray(selected), score_preferred_candidate=np.asarray(selected),
        execution_reason=np.asarray(reasons))
    report = dict(method="gnn_value", accepted=True, executed_steps=s.horizon,
        scenario_id=s.scenario_id, window_steps=6, safety_window_steps=6,
        candidate_count_requested=8, scoring_steps=[dict(t=t) for t in range(s.horizon)],
        sampling_steps=[dict(t=t) for t in range(s.horizon)])
    settings = dict(control=dict(window_steps=6, safety_window_steps=6, candidate_count=8))
    return s, reference, arrays, report, settings, original_snapshots


def test_reconstruction_matches_historical_certificate_not_future_trace(monkeypatch):
    s, reference, arrays, report, settings, original = _historical_rollout(monkeypatch)
    snapshots = reconstruct_snapshots(s, arrays, report, settings, [0, 1, 3, 14])
    assert snapshots[0]["anchor"] is reference
    assert not np.array_equal(snapshots[0]["anchor"].power_kw, arrays["power_kw"])
    for t, snapshot in snapshots.items():
        state, anchor = original[t]
        np.testing.assert_array_equal(snapshot["state"].energy_kwh, state.energy_kwh)
        np.testing.assert_array_equal(snapshot["state"].previous_power_kw, state.previous_power_kw)
        for name in ("power_kw", "delivery_kw", "energy_kwh"):
            np.testing.assert_array_equal(getattr(snapshot["anchor"], name), getattr(anchor, name))
        assert snapshot["archival"]["common_boundary_verified"]


@pytest.mark.parametrize("corruption", ["energy", "previous_action_id", "candidate_count", "safety_window"])
def test_reconstruction_rejects_changed_archive_contract(monkeypatch, corruption):
    s, _, arrays, report, settings, _ = _historical_rollout(monkeypatch)
    if corruption == "energy":
        arrays["energy_kwh"][2, 0] += 1
    elif corruption == "previous_action_id":
        arrays["selected_candidate"][0] = 0
    elif corruption == "candidate_count":
        report["candidate_count_requested"] = 16
    else:
        settings["control"]["safety_window_steps"] = 4
    with pytest.raises(ValueError):
        reconstruct_snapshots(s, arrays, report, settings, [3])


def test_time_strata_are_shared_across_scales_and_do_not_use_outcomes():
    s, _ = small_case(336)
    hour = np.arange(336)*.5 % 24
    price = np.where(hour < 8, .3, np.where(hour < 17, .7, 1.2))
    s = replace(s, price_yuan_per_kwh=price)
    first = select_steps(s)
    other = select_steps(replace(s, seed=100, scenario_id="different-scale"))
    assert first == other
    assert len({row["t"] for row in first}) == 4
    assert first[0]["day_index"] < 5 and 6 <= first[0]["hour"] < 8
    assert first[1]["day_index"] < 5 and first[1]["price_yuan_per_kwh"] == 1.2
    assert first[2]["day_index"] >= 5 and 10 <= first[2]["hour"] < 16
    assert first[3]["day_index"] == 6 and 20 <= first[3]["hour"] < 23.5


def _score_case():
    s, anchor = small_case(12)
    model = CandidateValueGNN(window_steps=6, hidden_dim=8, latent_dim=4, coarse_width=2)
    pool = generate_candidates(s, s.initial_state(), anchor, 8, 42)
    h = s.network.n_pumps
    row = dict(t=0, mode="adaptive", total_messages=len(pool)*h, refined_messages=4,
        rounds=2, selected_groups=[dict(round=0, positions=[0, 1], new_messages=2),
                                  dict(round=1, positions=[h+2, h+3], new_messages=2)])
    return s, model, pool, row


def test_archived_mask_replay_matches_manual_original_order_and_full_scores():
    s, model, pool, row = _score_case()
    full, adaptive, timing = replay_scores(model, s, s.initial_state(), pool, row)
    assert model.training  # Helper restores caller mode.
    with torch.no_grad():
        prepared = model.prepare(s, s.initial_state(), pool.actions)
        cache = model.start_cache(prepared)
        for group in row["selected_groups"]:
            model.refine(cache, group["positions"])
        np.testing.assert_array_equal(adaptive, (model.score(cache)+prepared.current_yuan).numpy())
        np.testing.assert_array_equal(full, (model.full(prepared)+prepared.current_yuan).numpy())
    assert timing["refined_messages"] == 4
    assert timing["timing_scope"].startswith("diagnostic_replay_only")


@pytest.mark.parametrize("corruption", ["duplicate", "bounds", "round", "count", "pool"])
def test_mask_replay_rejects_archive_or_pool_mismatch(corruption):
    s, model, pool, row = _score_case()
    row = copy.deepcopy(row)
    if corruption == "duplicate":
        row["selected_groups"][1]["positions"] = [0, 1]
    elif corruption == "bounds":
        row["selected_groups"][0]["positions"][0] = row["total_messages"]
    elif corruption == "round":
        row["selected_groups"][0]["round"] = 1
    elif corruption == "count":
        row["refined_messages"] += 1
    else:
        row["total_messages"] -= 4
    with pytest.raises(ValueError):
        replay_scores(model, s, s.initial_state(), pool, row)
