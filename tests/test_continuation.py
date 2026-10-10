"""An accepted continuation executes exactly the scored current action."""
from dataclasses import replace
import numpy as np
import pytest
from dr_research.candidates import generate_candidates
from dr_research.continuation import try_continuation, prepare_continuation_context
from dr_research.physics import validate_plan, transition, validate_action
from dr_research.types import Action, Plan
from helpers import small_case, suffix


def test_exact_action_acceptance_and_shared_tail_without_optimizer(monkeypatch):
    monkeypatch.setattr("scipy.optimize.linprog", lambda *a, **k: pytest.fail("Online optimizer"))
    s, anchor = small_case(16)
    raw = Action(anchor.power_kw[0]+.2, anchor.delivery_kw[0].copy())
    result = try_continuation(s, s.initial_state(), anchor, raw, 8)
    assert result.accepted, result.reason
    np.testing.assert_array_equal(result.plan.power_kw[0], raw.power_kw)
    np.testing.assert_array_equal(result.plan.delivery_kw[0], raw.delivery_kw)
    np.testing.assert_array_equal(result.plan.power_kw[8:], anchor.power_kw[8:])
    np.testing.assert_array_equal(result.plan.energy_kwh[8:], anchor.energy_kwh[8:])
    assert validate_plan(s, result.plan).feasible


def test_unknown_never_returns_a_shrunken_action_or_a_plan():
    s, anchor = small_case(8)
    raw = Action(anchor.power_kw[0]+.5, anchor.delivery_kw[0].copy())
    assert validate_action(s, s.initial_state(), raw).feasible
    result = try_continuation(s, s.initial_state(), anchor, raw, 2)
    assert result.status == "unknown" and result.plan is None
    assert "no_free_future_step" in result.reason
    np.testing.assert_array_equal(raw.power_kw, anchor.power_kw[0]+.5)


def test_backup_reuses_certified_plan_and_context_checks_identity():
    s, anchor = small_case()
    state = s.initial_state()
    context = prepare_continuation_context(s, state, anchor, 4)
    result = try_continuation(s, state, anchor, anchor.action(0), 4, context=context)
    assert result.accepted and result.plan is anchor and result.attempts == 0
    with pytest.raises(ValueError, match="context"):
        try_continuation(s, replace(state), anchor, anchor.action(0), 4, context=context)


@pytest.mark.parametrize("window", [12, 24, 48])
def test_replay_checks_only_changed_window_and_retains_exact_tail(monkeypatch, window):
    import dr_research.continuation as continuation
    s, anchor = small_case(120)
    state = s.initial_state()
    context = prepare_continuation_context(s, state, anchor, window)
    original = continuation.validate_plan
    horizons = []
    def observed(local, plan, *args, **kwargs):
        horizons.append(local.horizon)
        return original(local, plan, *args, **kwargs)
    monkeypatch.setattr(continuation, "validate_plan", observed)
    action = Action(anchor.power_kw[0]+.2, anchor.delivery_kw[0])
    result = try_continuation(s, state, anchor, action, window, context=context)
    assert result.accepted, result.reason
    assert horizons and set(horizons) == {window}
    np.testing.assert_array_equal(result.plan.power_kw[window:], anchor.power_kw[window:])
    assert result.replay_count == len(horizons)
    for array in (result.plan.power_kw, result.plan.delivery_kw, result.plan.energy_kwh):
        assert not array.flags.writeable


@pytest.mark.parametrize("mismatch", ["scenario", "state", "anchor", "window", "state_mutation"])
def test_context_is_bound_to_one_unchanged_control_state(mismatch):
    s, anchor = small_case()
    state = s.initial_state()
    context = prepare_continuation_context(s, state, anchor, 4)
    window = 4
    if mismatch == "scenario":
        s = replace(s, scenario_id="another")
    elif mismatch == "state":
        state = replace(state)
    elif mismatch == "anchor":
        anchor = replace(anchor)
    elif mismatch == "window":
        window = 6
    else:
        state.previous_power_kw[0] += .1
    with pytest.raises(ValueError, match="context"):
        try_continuation(s, state, anchor, anchor.action(0), window, context=context)


@pytest.mark.parametrize("corruption", ["shape", "initial_inventory", "terminal_inventory", "nonfinite", "future_power"])
def test_context_rejects_invalid_reference_interface_or_window(corruption):
    s, anchor = small_case()
    if corruption == "shape":
        anchor = replace(anchor, power_kw=anchor.power_kw[:-1])
    elif corruption in ("initial_inventory", "terminal_inventory"):
        energy = anchor.energy_kwh.copy()
        energy[0 if corruption == "initial_inventory" else -1, 0] += 1
        anchor = replace(anchor, energy_kwh=energy)
    else:
        power = anchor.power_kw.copy()
        power[1, 0] = np.nan if corruption == "nonfinite" else 1000
        anchor = replace(anchor, power_kw=power)
    with pytest.raises(ValueError):
        prepare_continuation_context(s, s.initial_state(), anchor, 4)


def test_short_rollout_preserves_exact_actions_and_final_inventory():
    s, anchor = small_case(12)
    state = s.initial_state()
    powers, flows, energies = [], [], [state.energy_kwh.copy()]
    chosen_nonbackup = 0
    for t in range(s.horizon):
        pool = generate_candidates(s, state, anchor, 32, 42)
        order = np.argsort([candidate.action.power_kw.sum() for candidate in pool])
        context = prepare_continuation_context(s, state, anchor, 6)
        for index in order:
            candidate = pool[int(index)]
            result = try_continuation(s, state, anchor, candidate, 6, context=context)
            if result.accepted:
                break
        assert result.accepted
        chosen_nonbackup += candidate.candidate_id != 0
        np.testing.assert_array_equal(result.plan.power_kw[0], candidate.action.power_kw)
        np.testing.assert_array_equal(result.plan.delivery_kw[0], candidate.action.delivery_kw)
        assert validate_plan(s, result.plan, state).feasible
        action = result.plan.action(0)
        powers.append(action.power_kw)
        flows.append(action.delivery_kw)
        state = transition(s, state, action)
        energies.append(state.energy_kwh.copy())
        anchor = Plan(state.t, result.plan.power_kw[1:], result.plan.delivery_kw[1:], result.plan.energy_kwh[1:])
    assert chosen_nonbackup > 0
    assert validate_plan(s, Plan(0, np.asarray(powers), np.asarray(flows), np.asarray(energies))).feasible


def test_terminal_current_candidate_joins_without_a_fictitious_future():
    s, anchor = small_case()
    state, tail = suffix(s, anchor, s.horizon-1)
    pool = generate_candidates(s, state, tail, 16, 42)
    result = try_continuation(s, state, tail, pool[0], 4)
    assert result.accepted and result.plan is tail


def test_numerical_replay_failure_is_unknown(monkeypatch):
    from dr_research.types import ValidationReport
    s, anchor = small_case()
    monkeypatch.setattr("dr_research.continuation.validate_plan",
                        lambda *a, **k: ValidationReport(False, ("forced replay failure",), {}))
    raw = Action(anchor.power_kw[0]+.2, anchor.delivery_kw[0])
    result = try_continuation(s, s.initial_state(), anchor, raw, 4)
    assert result.status == "unknown" and result.plan is None
    assert "forced replay failure" in result.reason
