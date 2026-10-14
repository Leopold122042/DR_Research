import pytest

from dr_research.diagnostic_metrics import summarize_snapshot


def record(candidate_id, cost, accepted=True, full=0.0, adaptive=0.0, status="optimal"):
    return dict(candidate_id=candidate_id, lp_status=status, q_yuan=cost,
                accepted=accepted, full_score=full, adaptive_score=adaptive)


def baseline(cost=100.0, status="optimal"):
    return dict(lp_status=status, q_yuan=cost)


def test_exact_three_part_decomposition_and_group_selection():
    rows = [record(0, 130, full=9, adaptive=9),
            record(1, 105, accepted=False, full=1, adaptive=1),
            record(2, 110, full=2, adaptive=3),
            record(3, 120, full=3, adaptive=2)]
    result = summarize_snapshot(rows, baseline(), baseline(102), 3,
                                raw_counts=(1, 3), policy_count=3, n_tanks=2)
    report = result["counts"]["3"]
    parts = report["decomposition"]
    assert result["status"] == "complete"
    assert report["execution_match"]
    assert parts["coverage"]["yuan"] == 5
    assert parts["certificate_miss"]["yuan"] == 5
    assert parts["ranking_and_attempt_limit"]["yuan"] == 10
    assert parts["total_gap"]["yuan_per_tank"] == 10
    assert parts["total_gap"]["percent_of_continuous_lp"] == 20
    assert parts["identity_residual_yuan"] == 0
    assert report["signed_differences"]["message_fidelity_E_minus_D"]["yuan"] == 10
    assert report["power_band_restriction"]["yuan"] == 2
    assert report["coverage_after_power_band"]["yuan"] == 3
    assert "groups" not in result["counts"]["1"]


def test_attempt_limit_fallback_can_have_negative_signed_difference():
    rows = [record(0, 110, full=3, adaptive=3),
            record(1, 105, accepted=False, full=1, adaptive=1),
            record(2, 120, full=2, adaptive=2)]
    report = summarize_snapshot(rows, baseline(), baseline(), 0,
                                max_attempts=1, raw_counts=(2,), policy_count=2)["counts"]["2"]
    assert report["fallback_due_to_attempt_limit"]
    assert report["groups"]["E_adaptive_unlimited"]["candidate_id"] == 2
    assert report["groups"]["F_current_policy"]["candidate_id"] == 0
    assert report["signed_differences"]["attempt_limit_F_minus_E"]["yuan"] == -10
    assert report["decomposition"]["ranking_and_attempt_limit"]["yuan"] == 0


def test_unknown_label_does_not_become_an_oracle_or_infeasible():
    rows = [record(0, 130, full=2, adaptive=2),
            record(1, None, accepted=False, full=1, adaptive=1, status="unknown")]
    result = summarize_snapshot(rows, baseline(), baseline(), 0,
                                raw_counts=(1,), policy_count=1)
    report = result["counts"]["1"]
    assert result["status"] == "incomplete_labels"
    assert report["best_known_feasible"]["q_yuan"] == 130
    assert report["lp_status_counts"]["unknown"] == 1
    assert report["lp_status_counts"]["infeasible"] == 0
    assert report["pool_oracle"] is None
    assert report["decomposition"] is None
    assert report["coverage"] is None


def test_nested_ids_are_raw_prefix_ids_and_expansion_needs_no_scores():
    rows = [record(0, 140, full=3, adaptive=3),
            record(7, 125, full=2, adaptive=2),
            record(64, 120, full=1, adaptive=1),
            record(257, 110, full=None, adaptive=None)]
    result = summarize_snapshot(rows, baseline(), baseline(), 64,
                                raw_counts=(8, 64, 1024), policy_count=64)
    assert result["counts"]["8"]["actual_candidate_count"] == 2
    assert result["counts"]["8"]["coverage"]["yuan"] == 25
    assert result["counts"]["64"]["coverage"]["yuan"] == 20
    assert result["counts"]["1024"]["coverage"]["yuan"] == 10
    assert "groups" not in result["counts"]["1024"]


def test_tied_scores_preserve_input_order_and_execution_mismatch_is_visible():
    rows = [record(0, 130, full=9, adaptive=9),
            record(7, 120, full=1, adaptive=1),
            record(3, 110, full=1, adaptive=1)]
    result = summarize_snapshot(rows, baseline(), baseline(), 3,
                                raw_counts=(8,), policy_count=8)
    assert result["status"] == "execution_mismatch"
    assert result["counts"]["8"]["groups"]["F_current_policy"]["candidate_id"] == 7
    assert result["counts"]["8"]["decomposition"] is None
    assert result["counts"]["8"]["signed_differences"] is None
    assert result["counts"]["8"]["decomposition_invalid_reason"] == "archived_execution_not_reproduced"


@pytest.mark.parametrize("rows, band_cost, match", [
    ([record(0, 120), record(1, None, status="infeasible")], 100, "accepted but LP proved infeasible"),
    ([record(0, 120), record(1, 99)], 100, "negative"),
    ([record(0, 120), record(1, 105)], 110, "band gap"),
])
def test_contradictions_raise(rows, band_cost, match):
    with pytest.raises(ValueError, match=match):
        summarize_snapshot(rows, baseline(), baseline(band_cost), 0,
                           raw_counts=(1,), policy_count=1)


def test_small_numerical_residual_is_preserved_with_warning():
    result = summarize_snapshot([record(0, 100 - 1e-6)], baseline(), baseline(), 0,
                                raw_counts=(1,), policy_count=1)
    assert result["counts"]["1"]["decomposition"]["coverage"]["yuan"] < 0
    assert result["warnings"]


def test_infeasible_candidate_is_counted_and_ignored_by_oracle():
    rows = [record(0, 125, full=2, adaptive=2),
            record(3, None, accepted=False, full=1, adaptive=1, status="infeasible")]
    result = summarize_snapshot(rows, baseline(), baseline(), 0,
                                raw_counts=(4,), policy_count=4)
    report = result["counts"]["4"]
    assert report["labels_complete"]
    assert report["lp_status_counts"]["infeasible"] == 1
    assert report["pool_oracle"]["candidate_id"] == 0
    assert report["coverage"]["yuan"] == 25
    assert report["raw_preferred"]["full"]["candidate_id"] == 3
    assert report["raw_preferred"]["full"]["lp_feasible"] is False
    assert report["raw_preferred"]["adaptive"]["lp_status"] == "infeasible"


@pytest.mark.parametrize("which", ["continuous", "band"])
def test_infeasible_baseline_conflicts_with_certified_backup(which):
    continuous = baseline(None, "infeasible") if which == "continuous" else baseline()
    band = baseline(None, "infeasible") if which == "band" else baseline()
    with pytest.raises(ValueError, match="despite a certified backup"):
        summarize_snapshot([record(0, 120)], continuous, band, 0,
                           raw_counts=(1,), policy_count=1)


def test_unknown_band_keeps_audit_incomplete_but_main_decomposition_available():
    result = summarize_snapshot([record(0, 120)], baseline(), baseline(None, "unknown"), 0,
                                raw_counts=(1,), policy_count=1)
    assert result["status"] == "incomplete_labels"
    assert result["counts"]["1"]["decomposition"]["total_gap"]["yuan"] == 20
    assert result["counts"]["1"]["power_band_restriction"] is None
