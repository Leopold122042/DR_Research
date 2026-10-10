"""Grouped teacher supervision, Eq.24 decision training and artifact contracts."""
from copy import deepcopy
from pathlib import Path
from shutil import copyfile, copytree
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from dr_research import dataset, training
from dr_research.candidates import generate_candidates
from dr_research.config import load_config
from dr_research.control import construct_reference, solve_control_window
from dr_research.experiment_config import load_settings, search_space, seed_for, validate_settings
from dr_research.gnn import CandidateValueGNN
from dr_research.training import decision_loss, fit_model, fit_coarse, teacher_loss, load_checkpoint, tune_models
from dr_research.types import Action, State
from dr_research.scenario import generate_scenario
from dataclasses import replace
from test_dataset_exploration import short_case, settings as short_settings

ROOT = Path(__file__).resolve().parents[1]


def test_search_is_declared_and_weather_splits_stay_disjoint():
    smoke = load_settings(ROOT/"configs/comparison_smoke.toml")
    full = load_settings(ROOT/"configs/comparison.toml")
    assert len(search_space(smoke)) == 1
    expected = len(full["training"]["hidden_dims"])*len(full["training"]["latent_dims"])*len(full["training"]["learning_rates"])
    assert len(search_space(full)) == expected
    assert len({tuple(sorted(hp.items())) for hp in search_space(full)}) == expected
    assert seed_for(42, "train", "S1", 0) != seed_for(42, "validation", "S1", 0)
    smoke["data"]["test_weeks"] = smoke["data"]["train_weeks"].copy()
    with pytest.raises(ValueError, match="overlap"):
        validate_settings(smoke)


def test_eq24_detaches_teacher_and_rewards_its_preferred_action():
    teacher = torch.tensor([1., 4., 9.], requires_grad=True)
    wrong_scores = torch.tensor([5., 3., 0.], requires_grad=True)
    loss = decision_loss(wrong_scores, teacher, .2, 2.)
    loss.backward()
    assert teacher.grad is None
    assert wrong_scores.grad is not None and torch.count_nonzero(wrong_scores.grad) > 0
    correct = decision_loss(torch.tensor([0., 3., 8.]), teacher, .2, 2.)
    assert correct < loss.detach()
    shifted = decision_loss(torch.tensor([0., 3., 8.])+100, teacher+100, .2, 2.)
    torch.testing.assert_close(correct, shifted)


def test_forecast_ends_at_sunday_and_future_baseline_excludes_current_bill():
    physical = load_config(ROOT/"configs/experiment.toml")
    s = generate_scenario(physical, "S1", 42, weather_week=0)
    assert s.horizon == 336
    reference = construct_reference(s)
    t = s.horizon-2
    state = State(t, reference.energy_kwh[t].copy(), reference.power_kw[t-1].copy())
    action = reference.action(t)
    model = CandidateValueGNN(window_steps=4, hidden_dim=16, latent_dim=4, coarse_width=1)
    inputs = []
    hook = model.global_encoder.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    with torch.no_grad():
        prepared = model.prepare(s, state, [action])
    hook.remove()
    torch.testing.assert_close(inputs[0][:4], torch.tensor([s.price_yuan_per_kwh[t]/1.5, s.price_yuan_per_kwh[t+1]/1.5, 0., 0.], dtype=torch.float32))
    torch.testing.assert_close(inputs[0][20:24], torch.tensor([1., 1., 0., 0.]))
    prices = s.price_yuan_per_kwh.copy()
    prices[t] *= 2
    with torch.no_grad():
        changed = model.prepare(replace(s, price_yuan_per_kwh=prices), state, [action])
    torch.testing.assert_close(prepared.baseline_yuan, changed.baseline_yuan)
    torch.testing.assert_close(changed.current_yuan, 2*prepared.current_yuan)
    terminal = State(t+1, reference.energy_kwh[t+1].copy(), reference.power_kw[t].copy())
    with torch.no_grad():
        end = model.prepare(s, terminal, [reference.action(t+1)])
        assert end.baseline_yuan.item() == 0 and model.full(end).item() == 0


def test_mixed_decision_loss_updates_only_coarse_parameters():
    torch.set_num_threads(1)
    _, s = short_case()
    state, anchor = s.initial_state(), construct_reference(s)
    actions = generate_candidates(s, state, anchor, 8, 42).actions
    model = CandidateValueGNN(window_steps=4, hidden_dim=16, latent_dim=4, coarse_width=1).freeze_teacher()
    with torch.no_grad():
        p = model.prepare(s, state, actions)
        target = model.full(p)+p.current_yuan
    score = model.mixed(p, torch.tensor([0, 1]))+p.current_yuan
    loss = decision_loss(score, target, .2, max(float(target.std()), .1))
    loss.backward()
    assert all(parameter.grad is None for parameter in model.teacher_parameters())
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in model.coarse_parameters())


def test_complete_small_training_and_checkpoint_contracts(tmp_path, monkeypatch):
    physical, _ = short_case()
    cfg = short_settings()
    monkeypatch.setattr(dataset, "generate_scenario", lambda physical, scale, seed, weather_week: short_case(seed, weather_week)[1])
    data_dir, hpo_dir = tmp_path/"data", tmp_path/"hpo"
    manifest = dataset.prepare_dataset(physical, cfg, data_dir)
    result = tune_models(physical, cfg, data_dir, hpo_dir)
    assert len(result["models"]["value"]["trials"]) == 1
    path = hpo_dir/result["models"]["value"]["selected_checkpoint"]
    model, meta = load_checkpoint(path, physical, cfg, "value")
    assert meta["fit_report"]["trained_complete"] and meta["fit_report"]["trained_coarse"]
    assert meta["dataset_key"] == manifest["key"]
    assert model.error_predictor is not None and model.latency_model is not None
    assert all(not p.requires_grad for p in model.teacher_parameters())
    _, s = short_case()
    actions = generate_candidates(s, s.initial_state(), construct_reference(s), 8, 42).actions
    with torch.no_grad():
        p = model.prepare(s, s.initial_state(), actions)
        values = model.full(p)
    assert torch.isfinite(values).all()
    changed = deepcopy(cfg)
    changed["control"]["window_steps"] += 1
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path, physical, changed)
    changed = deepcopy(cfg)
    changed["training"]["epochs"] += 1
    changed["evaluation"]["repeats"] += 1
    loaded, _ = load_checkpoint(path, physical, changed)
    with torch.no_grad():
        torch.testing.assert_close(values, loaded.full(loaded.prepare(s, s.initial_state(), actions)))
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(path, physical, cfg, "unsupported_policy")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for name, mutate in (
        ("wrong_kind", lambda m: m.update(kind="unsupported_policy")),
        ("stale_data", lambda m: m.update(dataset_key="stale")),
        ("missing_data", lambda m: m.pop("dataset_key")),
        ("wrong_architecture_window", lambda m: m["architecture"].update(window_steps=12)),
        ("missing_architecture", lambda m: m.pop("architecture")),
        ("bool_window", lambda m: m["architecture"].update(window_steps=True)),
    ):
        altered = deepcopy(payload)
        mutate(altered["metadata"])
        altered_path = tmp_path/(name+".pt")
        torch.save(altered, altered_path)
        with pytest.raises(ValueError):
            load_checkpoint(altered_path, physical, cfg)
    with monkeypatch.context() as patch:
        patch.setattr(dataset, "DATASET_VERSION", "changed-label-protocol")
        with pytest.raises(ValueError, match="contract"):
            load_checkpoint(path, physical, cfg)
    # A resumed teacher trial skips fit_model, which used to be the only place
    # that reset the CPU thread count before coarse training and calibration.
    resumed_dir = tmp_path/"resumed_hpo"
    resumed_dir.mkdir()
    copyfile(hpo_dir/"settings.json", resumed_dir/"settings.json")
    copytree(hpo_dir/"trial_00", resumed_dir/"trial_00")
    coarse_fit, calibration = training.fit_coarse, training.calibrate_latency

    def no_teacher_fit(*args, **kwargs):
        raise AssertionError("Completed teacher trial must be reused")

    def check_runtime(function):
        def checked(*args, **kwargs):
            assert torch.get_num_threads() == cfg["control"]["torch_threads"] == 1
            assert torch.are_deterministic_algorithms_enabled()
            return function(*args, **kwargs)
        return checked

    with monkeypatch.context() as patch:
        patch.setattr(training, "fit_model", no_teacher_fit)
        patch.setattr(training, "fit_coarse", check_runtime(coarse_fit))
        patch.setattr(training, "calibrate_latency", check_runtime(calibration))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(False)
        resumed = tune_models(physical, cfg, data_dir, resumed_dir, resume=True)
        assert resumed["hpo_session_resumed"]
        assert torch.get_num_threads() == 1


def test_fixed_action_teacher_preserves_numerically_accepted_action():
    _, s = short_case()
    ref, state = construct_reference(s), s.initial_state()
    free = solve_control_window(s, state, ref, window_steps=4)
    assert free.status == "optimal"
    action = free.plan.action(0)
    fixed = solve_control_window(s, state, ref, window_steps=4, fixed_first_action=action)
    assert fixed.status == "optimal", fixed.message
    np.testing.assert_allclose(fixed.plan.power_kw[0], action.power_kw, atol=1e-7, rtol=0)
    np.testing.assert_allclose(fixed.plan.delivery_kw[0], action.delivery_kw, atol=1e-7, rtol=0)
    assert fixed.electricity_yuan >= free.electricity_yuan-1e-5
    power = ref.power_kw[0].copy()
    power[0] = np.nextafter(s.network.pump_min_kw[0], -np.inf)
    tiny = Action(power, ref.delivery_kw[0].copy())
    fixed = solve_control_window(s, state, ref, window_steps=4, fixed_first_action=tiny)
    assert fixed.status == "optimal", fixed.message
    assert fixed.statistics["fixed_action_residuals"]["pump_lower"] > 0
    assert fixed.plan.power_kw[0, 0] == tiny.power_kw[0]
