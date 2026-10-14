"""Resumption, unknown-label handling and exact band restriction tests."""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest

from dr_research.diagnostic_runner import (
    atomic_json, band_scenario, load_config, output_lock, read_json,
    solve_label, summarize_output, array_hash, run_labels, file_hash, label_identity,
)
from dr_research.control import solve_control_window
from dr_research.types import SolveResult
from helpers import small_case


def settings():
    return {"control": {"window_steps": 4, "solver_method": "highs-ipm"}}


def config():
    return {"solver_time_limit_s": 10., "solver_threads": 1,
            "aggregate_pumps": True, "counts": [2, 4], "policy_count": 4}


def test_label_resume_does_not_solve_twice_and_rejects_different_identity(tmp_path, monkeypatch):
    import dr_research.diagnostic_runner as runner
    s, reference = small_case(8)
    original = runner.solve_control_window
    calls = []
    def wrapped(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)
    monkeypatch.setattr(runner, "solve_control_window", wrapped)
    path = tmp_path / "label.json"
    one, fresh = solve_label(path, s, s.initial_state(), reference, settings(), config(), key="same")
    assert fresh and one["lp_status"] == "optimal"
    two, fresh = solve_label(path, s, s.initial_state(), reference, settings(), config(), key="same")
    assert not fresh and two == one and len(calls) == 1
    with pytest.raises(ValueError, match="identity"):
        solve_label(path, s, s.initial_state(), reference, settings(), config(), key="different")


def test_unknown_is_not_infeasible_and_retried_only_explicitly(tmp_path, monkeypatch):
    import dr_research.diagnostic_runner as runner
    s, reference = small_case(8)
    calls = []
    def limited(*a, **kw):
        calls.append(1)
        return SolveResult("limit", None, None, None, {"total": .01}, "test limit", {})
    monkeypatch.setattr(runner, "solve_control_window", limited)
    path = tmp_path / "label.json"
    label, _ = solve_label(path, s, s.initial_state(), reference, settings(), config(), key="k")
    assert label["lp_status"] == "unknown" and label["solver_status"] == "limit"
    assert label["q_yuan"] is None
    solve_label(path, s, s.initial_state(), reference, settings(), config(), key="k")
    assert len(calls) == 1
    solve_label(path, s, s.initial_state(), reference, settings(), config(), key="k", retry_unknown=True)
    assert len(calls) == 2
    assert len(read_json(path)["previous_attempts"]) == 1


def test_power_band_is_same_problem_with_only_first_total_restricted():
    s, reference = small_case(8)
    value = float(reference.power_kw[0].sum())
    band = band_scenario(s, 0, [value, value])
    np.testing.assert_array_equal(band.power_upper_kw[1:], s.power_upper_kw[1:])
    np.testing.assert_array_equal(band.power_lower_kw[1:], s.power_lower_kw[1:])
    assert s.power_upper_kw[0] != value  # Original frozen scenario remains unchanged.
    base = solve_control_window(s, s.initial_state(), reference, window_steps=4)
    limited = solve_control_window(band, s.initial_state(), reference, window_steps=4)
    fixed = solve_control_window(s, s.initial_state(), reference, window_steps=4,
                                 fixed_first_action=reference.action(0))
    assert base.status == limited.status == fixed.status == "optimal"
    assert base.electricity_yuan <= limited.electricity_yuan + 1e-5
    assert limited.electricity_yuan <= fixed.electricity_yuan + 1e-5


def test_output_lock_releases_on_error(tmp_path):
    with pytest.raises(RuntimeError, match="test failure"):
        with output_lock(tmp_path):
            raise RuntimeError("test failure")
    with output_lock(tmp_path):
        atomic_json(tmp_path / "saved.json", {"complete": True})
    assert read_json(tmp_path / "saved.json") == {"complete": True}


def test_s2_s4_default_contract_and_invalid_counts(tmp_path):
    default = Path(__file__).resolve().parents[1] / "configs/cause_diagnosis.toml"
    cfg = load_config(default)
    assert cfg["scales"] == ["S2", "S3", "S4"] and cfg["counts"] == [64, 256]
    modified = default.read_text().replace("[64, 256]", "[63, 256]")
    target = tmp_path / "bad.toml"
    target.write_text(modified)
    with pytest.raises(ValueError, match="powers of two"):
        load_config(target)


def test_partial_summaries_do_not_claim_a_pool_oracle(tmp_path):
    atomic_json(tmp_path / "snapshot_index.json", {"snapshots": [dict(id="S2_t001", scale="S2", t=1, category="test")]})
    folder = tmp_path / "snapshots/S2_t001"
    folder.mkdir(parents=True)
    arrays = dict(energy=np.ones(2), previous_power=np.ones(4),
                  candidate_power=np.arange(20.).reshape(5, 4), candidate_delivery=np.ones((5, 4)))
    np.savez(folder / "arrays.npz", **arrays)
    rows = [dict(candidate_id=i, accepted=True, full_score=float(i), adaptive_score=float(i),
                 continuation_s=0., continuation_reason="test", action_sha256=array_hash(arrays["candidate_power"][i], arrays["candidate_delivery"][i])) for i in range(5)]
    meta = dict(id="S2_t001", protocol_id="fixture", scenario_fingerprint="fixture",
                candidates=rows, raw_count=4, selected_id=0, n_tanks=2,
                arrays_file="arrays.npz", arrays_sha256=file_hash(folder / "arrays.npz"), total_power_interval_kw=[0., 50.])
    atomic_json(folder / "snapshot.json", meta)
    atomic_json(folder / "labels/continuous.json", dict(identity=label_identity(meta, arrays, "continuous", "continuous"), lp_status="optimal", q_yuan=10.))
    atomic_json(folder / "labels/candidate_00000.json", dict(identity=label_identity(meta, arrays, "candidate_00000", rows[0]["action_sha256"]), lp_status="optimal", q_yuan=11.))
    summary = summarize_output(tmp_path, config())
    assert summary["completed_snapshot_count"] == 0
    state = summary["snapshots"][0]
    assert state["counts"]["4"]["pool_oracle"] is None
    assert state["counts"]["4"]["decomposition"] is None
    assert (tmp_path / "candidate_labels.csv").exists()
    assert (tmp_path / "snapshot_summary.csv").exists()
    wrong = read_json(folder / "labels/candidate_00000.json")
    wrong["identity"] = "copied-from-another-action"
    atomic_json(folder / "labels/candidate_00000.json", wrong)
    with pytest.raises(ValueError, match="identity differs during summary"):
        summarize_output(tmp_path, config())


def test_tiny_label_pipeline_pauses_resumes_and_produces_exact_sum(tmp_path):
    from dr_research.candidates import generate_candidates
    from dr_research.control import construct_reference
    from dr_research.continuation import try_continuation
    from dr_research.io import save_scenario, scenario_fingerprint
    from dr_research.config import load_config as physical_config
    from dr_research.scenario import generate_scenario
    from helpers import scenario_prefix
    s = scenario_prefix(generate_scenario(physical_config(Path(__file__).resolve().parents[1] / "configs/experiment.toml"), "S1", 17), 8)
    reference = construct_reference(s)
    state = s.initial_state()
    pool = generate_candidates(s, state, reference, 4, 42)
    evaluation = tmp_path / "source"
    evaluation.mkdir()
    save_scenario(evaluation / "scenario.npz", s)
    out = tmp_path / "out"
    folder = out / "snapshots/S2_t000"
    folder.mkdir(parents=True)
    rows = []
    for c in pool:
        result = try_continuation(s, state, reference, c, 4)
        rows.append(dict(candidate_id=c.candidate_id, accepted=result.accepted,
            full_score=float(c.candidate_id), adaptive_score=float(c.candidate_id),
            action_sha256=array_hash(c.action.power_kw, c.action.delivery_kw)))
    np.savez(folder / "arrays.npz", energy=state.energy_kwh, previous_power=state.previous_power_kw,
             candidate_power=np.stack([c.action.power_kw for c in pool]),
             candidate_delivery=np.stack([c.action.delivery_kw for c in pool]))
    meta = dict(id="S2_t000", scale="S2", t=0, n_tanks=2, raw_count=4, protocol_id="fixture",
        scenario_rel="scenario.npz", scenario_fingerprint=scenario_fingerprint(s),
        arrays_file="arrays.npz", arrays_sha256=file_hash(folder / "arrays.npz"),
        candidates=rows, selected_id=0, max_attempts=8,
        total_power_interval_kw=pool.stats["total_power_interval_kw"])
    atomic_json(folder / "snapshot.json", meta)
    items = [dict(id="S2_t000", scale="S2", t=0, category="fixture")]
    atomic_json(out / "snapshot_index.json", dict(snapshots=items))
    archived = dict(protocol_id="fixture", protocol=dict(settings=settings()))
    run_labels(out, evaluation, archived, config(), items, max_solves=2)
    assert len(list((folder / "labels").glob("*.json"))) == 2
    first = file_hash(folder / "labels/continuous.json")
    run_labels(out, evaluation, archived, config(), items)
    assert file_hash(folder / "labels/continuous.json") == first
    report = read_json(out / "summary.json")
    assert report["completed_snapshot_count"] == 1
    parts = report["snapshots"][0]["counts"]["4"]["decomposition"]
    assert abs(parts["identity_residual_yuan"]) < 1e-8
    # Extending the pool reuses every old label, while new rows have no
    # independently evaluated adaptive policy score.
    cached = {p.name: file_hash(p) for p in (folder / "labels").glob("*.json")}
    extended = generate_candidates(s, state, reference, 8, 42)
    prefix_hashes = {r["candidate_id"]: r["action_sha256"] for r in rows}
    extra_rows = []
    for c in extended:
        h = array_hash(c.action.power_kw, c.action.delivery_kw)
        if c.candidate_id in prefix_hashes:
            assert prefix_hashes[c.candidate_id] == h
        extra_rows.append(dict(candidate_id=c.candidate_id,
            accepted=try_continuation(s, state, reference, c, 4).accepted,
            full_score=float(c.candidate_id) if c.candidate_id <= 4 else None,
            adaptive_score=float(c.candidate_id) if c.candidate_id <= 4 else None,
            action_sha256=h))
    np.savez(folder / "arrays8.npz", energy=state.energy_kwh, previous_power=state.previous_power_kw,
             candidate_power=np.stack([c.action.power_kw for c in extended]),
             candidate_delivery=np.stack([c.action.delivery_kw for c in extended]))
    meta.update(raw_count=8, candidates=extra_rows, arrays_file="arrays8.npz",
                arrays_sha256=file_hash(folder / "arrays8.npz"))
    atomic_json(folder / "snapshot.json", meta)
    run_labels(out, evaluation, archived, config(), items)
    assert all(file_hash(folder / "labels" / name) == h for name, h in cached.items())
    extended_report = read_json(out / "summary.json")["snapshots"][0]
    assert extended_report["status"] == "complete"
    assert extended_report["counts"]["8"]["score_scope"] == "coverage_only"
    assert extended_report["counts"]["8"]["coverage"]["yuan"] <= extended_report["counts"]["4"]["coverage"]["yuan"] + 1e-5
