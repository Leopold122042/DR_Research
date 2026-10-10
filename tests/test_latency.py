"""Latency fits whole measured groups and reserves independent workloads."""
import numpy as np
import pytest

from dr_research.latency import (LatencyTree, fit_latency_profile,
                                measure_refinement_batches, workload_features)


def test_workload_counts_shared_receiver_and_candidate_once():
    # Pump receivers [0,0,1,1], two candidates; duplicate position 1 is omitted.
    features = workload_features([0, 1, 1, 4, 6], [0, 0, 1, 1], 4,
                                 cold_projections=6)
    np.testing.assert_array_equal(features, [4, 6, 3, 2, 2, 3, 2])


def test_tree_uses_median_leaves_and_roundtrips():
    x = np.zeros((5, 7))
    tree = LatencyTree(max_depth=0).fit(x, [.01, .02, .03, .04, 20.])
    assert tree.predict(x[0]) == pytest.approx(.03)
    restored = LatencyTree.from_dict(tree.to_dict())
    np.testing.assert_array_equal(tree.predict(x), restored.predict(x))
    assert restored.in_profile(x[0])
    assert not restored.in_profile(np.ones(7))


def test_l1_tree_distinguishes_workload_regimes():
    x = np.zeros((12, 7))
    x[:, 0] = np.arange(12)
    y = np.where(x[:, 0] < 6, .001, .1)
    tree = LatencyTree(min_leaf=2).fit(x, y)
    np.testing.assert_allclose(tree.predict(x), y)


def test_profile_holdout_excludes_all_duplicate_workloads():
    records = []
    for size in range(1, 9):
        for _ in range(2):
            records.append(dict(features=[size, 0, size, 1, 1, size, 1], median_s=size*.001))
    tree, report = fit_latency_profile(records, seed=42)
    assert report["split"] == "held_out_distinct_workload_vectors"
    assert report["train_batches"] == 12 and report["validation_batches"] == 4
    assert len(report["validation_actual_s"]) == 4
    assert tree.calibration == report
    assert tree.predict(records[0]["features"]) > 0


def test_measurements_rebuild_independent_operations_and_warm_up():
    builds, operations, sync = [], [], []

    def factory():
        token = object()
        builds.append(token)

        def operation():
            assert token not in operations
            operations.append(token)
        return operation

    rows = measure_refinement_batches([dict(features=[1]*7, factory=factory)],
                                      repeats=3, warmup=2, synchronize=lambda: sync.append(True))
    assert len(builds) == len(operations) == 5
    assert len(sync) == 10
    assert len(rows[0]["samples_s"]) == 3 and rows[0]["median_s"] > 0


@pytest.mark.parametrize("times", [[0.], [-1.], [np.nan]])
def test_latency_never_accepts_nonpositive_or_nonfinite_costs(times):
    with pytest.raises(ValueError):
        LatencyTree().fit(np.zeros((1, 7)), times)
