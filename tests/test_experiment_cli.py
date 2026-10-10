"""Comparison protocol, ablations, verified resume and shell failure status."""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest
from dr_research import experiment


@pytest.mark.parametrize('command', ['compare', 'run'])
@pytest.mark.parametrize('accepted,fallback,deadline,expected', [
    (True, 0, 0, 0), (False, 0, 0, 2), (True, 1, 0, 0), (True, 0, 1, 2)])
def test_failed_control_results_propagate_to_exit_code(monkeypatch, tmp_path,
                                                      command, accepted, fallback, deadline, expected):
    monkeypatch.setattr(experiment, 'load_config', lambda _: {})
    monkeypatch.setattr(experiment, 'load_settings', lambda _: {'control': {'torch_threads': 1}})
    monkeypatch.setattr(experiment, 'prepare_dataset', lambda *a, **kw: {})
    monkeypatch.setattr(experiment, 'tune_models', lambda *a, **kw: {
        'models': {kind: {'selected_checkpoint': 'policy.pt'} for kind in ('value',)}})
    calls = []
    def comparison(*a, **kw):
        calls.append(True)
        return {'records': [dict(accepted=accepted, fallback_count=fallback, deadline_misses=deadline)]}
    monkeypatch.setattr(experiment, 'compare', comparison)
    args = [command, '--out', str(tmp_path/'results')]
    if command == 'compare':
        args += ['--hpo', 'unused']
    assert experiment.main(args) == expected
    assert calls == [True]


def comparison_case(monkeypatch, tmp_path):
    from helpers import small_case
    scenario, _ = small_case()
    hpo = tmp_path/"hpo"
    hpo.mkdir()
    checkpoint = hpo/"selected.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    (hpo/"hpo_report.json").write_text(json.dumps({"models": {"value": {
        "selected_checkpoint": "selected.pt", "selected_trial": 0,
        "trials": [{"trial": 0, "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest()}]
    }}}), encoding="utf-8")
    settings = dict(master_seed=42, control=dict(torch_threads=1),
                    data=dict(test_scales=["S1", "S2"], test_weeks=[10]),
                    refinement=dict(mode="adaptive", time_budget_s=1.), evaluation=dict(repeats=1))
    checkpoint_settings, calls = [], []

    def load(path, physical, cfg, kind):
        checkpoint_settings.append(deepcopy(cfg))
        assert kind == "value"
        return SimpleNamespace(error_predictor=object(), latency_model=object()), {}

    def control(s, cfg, method, model, **kwargs):
        calls.append((method, deepcopy(cfg), kwargs))
        return dict(method=method, accepted=True, fallback_count=0, deadline_misses=0,
                    solver_calls=int(method == "lp_mpc"), total_electricity_yuan=10.,
                    control_time_s=.1, status="completed"), {"power_kw": np.zeros((1, 4))}

    monkeypatch.setattr(experiment, "generate_scenario", lambda *a, **kw: scenario)
    monkeypatch.setattr(experiment, "load_checkpoint", load)
    monkeypatch.setattr(experiment, "run_controller", control)
    return hpo, settings, checkpoint_settings, calls


def test_mode_and_scale_overrides_do_not_change_training_contract(monkeypatch, tmp_path):
    hpo, settings, loaded, calls = comparison_case(monkeypatch, tmp_path)
    result = experiment.compare({}, settings, hpo, tmp_path/"out", None,
                                gnn_mode="full", scales=["S1"], candidate_count=1024)
    assert len(calls) == 2 and {row[0] for row in calls} == {"lp_mpc", "gnn_value"}
    assert all(row[1]["refinement"]["mode"] == "full" for row in calls)
    assert all(row[1]["control"]["candidate_count"] == 1024 for row in calls)
    assert all(cfg == settings for cfg in loaded)
    assert settings["refinement"]["mode"] == "adaptive"
    assert result["ablation"]["evaluated_scales"] == ["S1"]
    assert result["ablation"]["gnn_mode"] == "full"
    assert result["ablation"]["candidate_count"] == 1024


def test_resume_skips_only_hashed_completed_runs(monkeypatch, tmp_path):
    hpo, settings, _, calls = comparison_case(monkeypatch, tmp_path)
    output = tmp_path/"out"
    first = experiment.compare({}, settings, hpo, output, None, scales=["S1"])
    resumed = experiment.compare({}, settings, hpo, output, None, scales=["S1"], resume=True)
    assert len(calls) == 2 and resumed["records"] == first["records"]
    trace = output/"S1_week10"/"gnn_value_r0.npz"
    trace.write_bytes(trace.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="checksum"):
        experiment.compare({}, settings, hpo, output, None, scales=["S1"], resume=True)


def test_resume_rejects_ablation_protocol_changes(monkeypatch, tmp_path):
    hpo, settings, _, _ = comparison_case(monkeypatch, tmp_path)
    output = tmp_path/"out"
    experiment.compare({}, settings, hpo, output, None, scales=["S1"])
    with pytest.raises(ValueError, match="Resume protocol differs"):
        experiment.compare({}, settings, hpo, output, None, scales=["S1"],
                           gnn_mode="coarse", resume=True)


def test_compare_scales_must_stay_inside_declared_test_scales(monkeypatch, tmp_path):
    hpo, settings, _, _ = comparison_case(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="subset"):
        experiment.compare({}, settings, hpo, tmp_path/"out", None, scales=["S5"])
    assert not (tmp_path/"out").exists()


def test_tune_resume_is_explicitly_forwarded_to_training(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "load_config", lambda _: {})
    monkeypatch.setattr(experiment, "load_settings", lambda _: {"control": {"torch_threads": 1}})
    calls = []

    def tune(*args, **kwargs):
        calls.append(kwargs)
        return {"models": {"value": {"selected_checkpoint": "selected.pt"}}}

    monkeypatch.setattr(experiment, "tune_models", tune)
    assert experiment.main(["tune", "--dataset", "labels", "--out", str(tmp_path/"hpo"), "--resume"]) == 0
    assert calls == [{"resume": True}]

