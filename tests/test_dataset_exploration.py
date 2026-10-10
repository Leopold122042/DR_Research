"""Independent exact-action exploration and grouped offline LP labels."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from dr_research import dataset
from dr_research.config import load_config
from dr_research.control import construct_reference, solve_control_window
from dr_research.experiment_config import load_settings, seed_for
from dr_research.physics import transition, validate_plan
from dr_research.scenario import generate_scenario
from dr_research.types import Plan
from helpers import scenario_prefix

ROOT = Path(__file__).resolve().parents[1]


def short_case(seed=42, week=0):
    physical = load_config(ROOT/"configs/experiment.toml")
    return physical, scenario_prefix(generate_scenario(physical, "S1", seed, weather_week=week), 8)


def settings():
    cfg = load_settings(ROOT/"configs/comparison_smoke.toml")
    cfg["control"].update(window_steps=4, safety_window_steps=4)
    cfg["data"].update(sample_stride=3)
    return cfg


def explore(s, cfg):
    state = s.initial_state()
    anchor = construct_reference(s)
    records, powers, flows, energies = [], [], [], [state.energy_kwh.copy()]
    for t in range(s.horizon):
        pool, chosen, certificate, info = dataset.exploration_step(s, state, anchor, cfg, "train")
        candidate = pool[chosen]
        assert not hasattr(candidate, "plan")
        np.testing.assert_array_equal(certificate.power_kw[0], candidate.action.power_kw)
        np.testing.assert_array_equal(certificate.delivery_kw[0], candidate.action.delivery_kw)
        assert validate_plan(s, certificate, state).feasible
        records.append(dict(t=t, chosen_candidate_id=candidate.candidate_id, pool_count=len(pool), **info))
        powers.append(candidate.action.power_kw)
        flows.append(candidate.action.delivery_kw)
        state = transition(s, state, candidate.action)
        energies.append(state.energy_kwh.copy())
        anchor = Plan(state.t, certificate.power_kw[1:], certificate.delivery_kw[1:], certificate.energy_kwh[1:])
    plan = Plan(0, np.asarray(powers), np.asarray(flows), np.asarray(energies))
    assert validate_plan(s, plan).feasible
    return records, plan


def test_exploration_is_reproducible_and_has_no_lp_selection(monkeypatch):
    _, s = short_case()
    def forbidden(*a, **k):
        pytest.fail("Exploration called an economic optimizer")
    monkeypatch.setattr(dataset, "solve_control_window", forbidden)
    monkeypatch.setattr("dr_research.control.linprog", forbidden)
    first, a = explore(s, settings())
    second, b = explore(s, settings())
    assert first == second
    assert any(row["chosen_candidate_id"] > 0 for row in first)
    assert all(row["pool_count"] > 1 for row in first[:-1])
    np.testing.assert_array_equal(a.power_kw, b.power_kw)
    with pytest.raises(ValueError, match="training and validation"):
        dataset.exploration_step(s, s.initial_state(), construct_reference(s), settings(), "test")


def test_unavailable_label_summary_distinguishes_proven_infeasibility():
    records = [dict(status="optimal", available=True), dict(status="infeasible", available=False),
               dict(status="limit", available=False), dict(status="error", available=False)]
    summary = dataset.label_summary(records, 12)
    assert summary["candidate_labels_attempted"] == 4
    assert summary["candidate_labels_unavailable"] == 3
    assert summary["candidate_labels_infeasible"] == 1
    assert summary["candidate_labels_unknown"] == 2
    assert summary["teacher_window_steps"] == 12


def test_group_labels_and_unknowns_follow_independent_exploration(tmp_path, monkeypatch):
    physical, _ = short_case()
    cfg = settings()
    monkeypatch.setattr(dataset, "generate_scenario", lambda physical, scale, seed, weather_week: short_case(seed, weather_week)[1])
    calls = []
    def solve(s, state, reference, **kwargs):
        assert kwargs.get("fixed_first_action") is not None
        calls.append((s.scenario_id, state.t))
        # Current legality alone is no certificate of future LP feasibility.
        if len(calls) % 3 == 2:
            return SimpleNamespace(status="infeasible", plan=None, electricity_yuan=None)
        return solve_control_window(s, state, reference, **kwargs)
    monkeypatch.setattr(dataset, "solve_control_window", solve)
    output = tmp_path/"dataset"
    manifest = dataset.prepare_dataset(physical, cfg, output)
    payload, _ = dataset.load_dataset(output, physical, cfg)
    trace = json.loads((output/"exploration.json").read_text())["records"]
    labels = json.loads((output/"label_status.json").read_text())["records"]
    assert set(payload) == {"train", "validation"}
    assert len(trace) == 16 and len(labels) == len(calls)
    assert manifest["candidate_labels_unavailable"] == sum(not row["available"] for row in labels) > 0
    assert manifest["candidate_labels_infeasible"] == sum(row["status"] == "infeasible" for row in labels)
    assert manifest["candidate_labels_unknown"] == sum(not row["available"] and row["status"] != "infeasible" for row in labels)
    assert manifest["teacher_window_steps"] == 4 and manifest["teacher"].startswith("4-step")
    for row in labels:
        if row["available"]:
            assert row["status"] == "optimal"
            assert row["total_yuan"] == pytest.approx(row["current_yuan"]+row["future_yuan"])
        else:
            assert "future_yuan" not in row
    for split in ("train", "validation"):
        rows = [r for r in trace if r["split"] == split]
        assert [r["t"] for r in rows] == list(range(8))
        assert rows[0]["sampled"] and rows[-1]["sampled"]
        assert any(not r["sampled"] for r in rows)
        for group in payload[split]:
            assert group["future"].ndim == 1
            assert len(group["future"]) == len(group["candidate_ids"]) == len(group["power"])
            assert torch.isfinite(group["future"]).all()
            permitted = {row["candidate_id"] for row in labels if row["split"] == split and row["case"] == group["case"] and row["t"] == group["t"] and row["available"]}
            assert set(group["candidate_ids"].tolist()) == permitted
    (output/"exploration.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        dataset.load_dataset(output, physical, cfg)
