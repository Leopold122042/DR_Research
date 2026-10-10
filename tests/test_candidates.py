"""Lightweight current-action coverage, equality constraints, and nesting."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from dr_research.candidates import generate_candidates, _POINT_CACHE
from dr_research.config import load_config
from dr_research.control import construct_reference
from dr_research.physics import validate_action, transition
from dr_research.scenario import generate_scenario
from helpers import small_case, suffix


def test_nested_prefix_ids_and_independent_current_physics(monkeypatch):
    monkeypatch.setattr("scipy.optimize.linprog", lambda *a, **k: pytest.fail("Online LP"))
    monkeypatch.setattr("dr_research.continuation.try_continuation", lambda *a, **k: pytest.fail("Premature continuation"))
    s, anchor = small_case()
    state = s.initial_state()
    small = generate_candidates(s, state, anchor, 16, 42)
    large = generate_candidates(s, state, anchor, 256, 42)
    repeated = generate_candidates(s, state, anchor, 256, 42)
    assert len(small) == 17 and len(large) == 257
    np.testing.assert_array_equal(small.ids, large.ids[:len(small)])
    for a, b in zip(small, large):
        np.testing.assert_array_equal(a.action.power_kw, b.action.power_kw)
        np.testing.assert_array_equal(a.action.delivery_kw, b.action.delivery_kw)
    for a, b in zip(large, repeated):
        assert not hasattr(a, "plan")
        assert validate_action(s, state, a.action).feasible
        np.testing.assert_array_equal(a.action.power_kw, b.action.power_kw)
        np.testing.assert_allclose(a.next_energy_kwh, transition(s, state, a.action).energy_kwh, atol=1e-10)
    assert large.stats["rejected_count"] == 0
    assert repeated.stats["sobol_cache_hit"]


def test_fixed_total_power_retains_hundreds_of_redistributions():
    s, anchor = small_case()
    total = anchor.power_kw[0].sum()
    s = replace(s, power_lower_kw=np.full(s.horizon, total), power_upper_kw=np.full(s.horizon, total))
    pool = generate_candidates(s, s.initial_state(), anchor, 256, 42)
    assert len(pool) == 257
    assert all(validate_action(s, s.initial_state(), c.action).feasible for c in pool)
    np.testing.assert_allclose([c.action.power_kw.sum() for c in pool], total, atol=1e-10)
    assert np.max(np.ptp(np.array([c.action.power_kw for c in pool]), axis=0)) > 1


def test_jointly_active_inventory_source_charge_and_total_bounds():
    s, anchor = small_case()
    total = anchor.power_kw[0].sum()
    source = np.sum((s.network.pump_cop-1)*anchor.power_kw[0])
    n = replace(s.network, tank_capacity_kwh=s.initial_energy_kwh.copy(),
                tank_charge_limit_kw=np.full(2, 8.4))
    s = replace(s, network=n, source_limit_kw=np.full((s.horizon, 1), source),
                power_lower_kw=np.full(s.horizon, total), power_upper_kw=np.full(s.horizon, total))
    pool = generate_candidates(s, s.initial_state(), anchor, 256, 42)
    assert len(pool) > 200
    assert pool.stats["rejected_count"] == 0
    assert all(validate_action(s, s.initial_state(), c.action).feasible for c in pool)
    assert np.max(np.ptp(np.asarray([c.action.power_kw for c in pool]), axis=0)) > .1


def test_prices_have_no_effect_and_cached_points_are_remapped_for_state():
    s, anchor = small_case()
    expensive = replace(s, price_yuan_per_kwh=np.full(s.horizon, 100.))
    a = generate_candidates(s, s.initial_state(), anchor, 32, 7)
    b = generate_candidates(expensive, expensive.initial_state(), anchor, 32, 7)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left.action.power_kw, right.action.power_kw)
        np.testing.assert_array_equal(left.action.delivery_kw, right.action.delivery_kw)
    state, tail = suffix(s, anchor, s.horizon-1)
    terminal = generate_candidates(s, state, tail, 32, 7)
    assert terminal.stats["sobol_cache_hit"]
    assert all(validate_action(s, state, candidate.action).feasible for candidate in terminal)
    for candidate in terminal:
        np.testing.assert_allclose(candidate.next_energy_kwh, s.terminal_energy_kwh, atol=1e-6)


def test_maximum_scale_retains_large_pool_and_no_plan_per_candidate():
    cfg = load_config(Path(__file__).resolve().parents[1]/"configs/experiment.toml")
    s = generate_scenario(cfg, "S5", 42, weather_week=0)
    anchor = construct_reference(s)
    pool = generate_candidates(s, s.initial_state(), anchor, 256, 42)
    assert len(pool) == 257 and pool.stats["rejected_count"] == 0
    for candidate in pool:
        assert validate_action(s, s.initial_state(), candidate.action).feasible
        assert not hasattr(candidate, "plan")
    large = generate_candidates(s, s.initial_state(), anchor, 1024, 42)
    assert len(large) == 1025 and large.stats["rejected_count"] == 0
    for small, bigger in zip(pool, large):
        np.testing.assert_array_equal(small.action.power_kw, bigger.action.power_kw)
        np.testing.assert_array_equal(small.action.delivery_kw, bigger.action.delivery_kw)


@pytest.mark.parametrize("count", [0, 3, 12, True, 1.5])
def test_invalid_counts_are_rejected(count):
    s, anchor = small_case()
    with pytest.raises(ValueError, match="raw_count"):
        generate_candidates(s, s.initial_state(), anchor, count, 42)


def test_invalid_seed_and_uncertified_current_anchor_rejected():
    s, anchor = small_case()
    with pytest.raises(ValueError, match="seed"):
        generate_candidates(s, s.initial_state(), anchor, 16, -1)
    bad = replace(anchor, power_kw=anchor.power_kw*100)
    with pytest.raises(ValueError, match="anchor current action"):
        generate_candidates(s, s.initial_state(), bad, 16, 42)
