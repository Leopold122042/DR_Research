"""Measured incremental latency: a small L1 regression tree with median leaves.

The predictor estimates complete refinement batches. It guides allocation; its
median predictions are not worst-case execution-time guarantees.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np


FEATURE_NAMES = ("new_messages", "cold_projections", "receiver_updates",
                 "candidate_updates", "readouts", "node_reads", "max_receiver_degree")


def workload_features(positions, receiver_index, n_pumps: int, *,
                      readout_nodes: int | None = None, cold_projections: int = 0) -> np.ndarray:
    """Describe deduplicated new positions before evaluating complete messages."""
    positions = np.unique(np.asarray(positions, dtype=np.int64).reshape(-1))
    receivers = np.asarray(receiver_index, dtype=np.int64).reshape(-1)
    if n_pumps < 1 or len(receivers) != n_pumps or np.any(positions < 0):
        raise ValueError("Invalid charging positions or receiver map")
    if (readout_nodes is not None and readout_nodes < 0) or cold_projections < 0:
        raise ValueError("Workload counts must be nonnegative")
    if not len(positions):
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    candidates, edges = np.divmod(positions, n_pumps)
    pairs = np.column_stack((candidates, receivers[edges]))
    _, degrees = np.unique(pairs, axis=0, return_counts=True)
    candidate_count = len(np.unique(candidates))
    node_reads = len(degrees) if readout_nodes is None else candidate_count*readout_nodes
    return np.asarray([len(positions), cold_projections, len(degrees), candidate_count,
                       candidate_count, node_reads, degrees.max()], dtype=float)


@dataclass
class LatencyTree:
    max_depth: int = 4
    min_leaf: int = 3
    max_thresholds: int = 16
    root: dict | None = None
    lower: list | None = None
    upper: list | None = None
    calibration: dict | None = None

    def fit(self, features, seconds):
        x, y = np.asarray(features, dtype=float), np.asarray(seconds, dtype=float)
        if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.shape != (len(x),)
                or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(x < 0) or np.any(y <= 0)):
            raise ValueError("Latency training needs finite nonnegative features and positive times")
        if self.max_depth < 0 or self.min_leaf < 1 or self.max_thresholds < 1:
            raise ValueError("Invalid latency tree complexity")
        self.lower, self.upper = x.min(axis=0).tolist(), x.max(axis=0).tolist()

        def build(rows, depth):
            target = y[rows]
            median = float(np.median(target))
            node = {"median_s": max(median, 1e-9), "samples": int(len(rows))}
            if depth >= self.max_depth or len(rows) < 2*self.min_leaf:
                return node
            best_loss = float(np.abs(target-median).sum())
            best = None
            for feature in range(x.shape[1]):
                values = np.unique(x[rows, feature])
                thresholds = (values[:-1]+values[1:])/2
                if len(thresholds) > self.max_thresholds:
                    thresholds = thresholds[np.unique(np.linspace(
                        0, len(thresholds)-1, self.max_thresholds).astype(int))]
                for threshold in thresholds:
                    left = x[rows, feature] <= threshold
                    if min(int(left.sum()), int((~left).sum())) < self.min_leaf:
                        continue
                    loss = sum(float(np.abs(v-np.median(v)).sum())
                               for v in (target[left], target[~left]))
                    if loss < best_loss-1e-15:
                        best_loss, best = loss, (feature, float(threshold), left)
            if best is not None:
                feature, threshold, left = best
                node.update(feature=feature, threshold=threshold,
                            left=build(rows[left], depth+1), right=build(rows[~left], depth+1))
            return node

        self.root = build(np.arange(len(x)), 0)
        return self

    def predict(self, features):
        if self.root is None:
            raise ValueError("Latency model is not calibrated")
        x = np.asarray(features, dtype=float)
        scalar = x.ndim == 1
        x = np.atleast_2d(x)
        if x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError("Invalid workload features")
        values = []
        for row in x:
            node = self.root
            while "feature" in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            values.append(node["median_s"])
        return float(values[0]) if scalar else np.asarray(values)

    def in_profile(self, features) -> bool:
        if self.lower is None or self.upper is None:
            return False
        x = np.asarray(features, dtype=float)
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper))

    def to_dict(self):
        return dict(version=1, feature_names=list(FEATURE_NAMES), max_depth=self.max_depth,
                    min_leaf=self.min_leaf, max_thresholds=self.max_thresholds,
                    root=self.root, lower=self.lower, upper=self.upper,
                    calibration=self.calibration)

    @classmethod
    def from_dict(cls, payload):
        if payload.get("version") != 1 or payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Unsupported latency profile")
        return cls(**{key: payload[key] for key in
                      ("max_depth", "min_leaf", "max_thresholds", "root", "lower", "upper", "calibration")})


def synchronized_time(operation: Callable, *, synchronize: Callable | None = None) -> float:
    """CPU work is synchronous; GPU callers must supply device synchronization."""
    if synchronize is not None:
        synchronize()
    started = perf_counter()
    operation()
    if synchronize is not None:
        synchronize()
    return max(perf_counter()-started, 1e-9)


def measure_refinement_batches(workloads, *, repeats=5, warmup=2, synchronize=None):
    """Measure factory-built independent batches, excluding cache reset time.

    Each item has ``features`` and a ``factory`` returning an operation. The
    factory restores the starting cache; the timed operation must include all
    incremental projection/message/aggregate/node/readout work.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError("Invalid profiling repeats")
    records = []
    for item in workloads:
        for _ in range(warmup):
            operation = item["factory"]()
            synchronized_time(operation, synchronize=synchronize)
        samples = []
        for _ in range(repeats):
            operation = item["factory"]()
            samples.append(synchronized_time(operation, synchronize=synchronize))
        records.append(dict(features=np.asarray(item["features"], dtype=float).tolist(),
                            median_s=float(np.median(samples)), samples_s=samples))
    return records


def fit_latency_profile(records, *, seed=42, validation_fraction=.25):
    """Hold out whole workload configurations and select tree complexity by MAE."""
    if not 0 < validation_fraction < 1 or len(records) < 4:
        raise ValueError("Latency calibration needs at least four measured batches")
    x = np.asarray([r["features"] for r in records], dtype=float)
    y = np.asarray([r["median_s"] for r in records], dtype=float)
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Latency calibration needs distinct workload configurations")
    order = np.random.default_rng(seed).permutation(len(unique))
    heldout_count = min(len(unique)-1, max(1, round(len(unique)*validation_fraction)))
    validation = np.isin(inverse, order[:heldout_count])
    train = ~validation
    choices = []
    for depth in (1, 2, 3, 4):
        for min_leaf in (1, 2, 4):
            tree = LatencyTree(max_depth=depth, min_leaf=min_leaf).fit(x[train], y[train])
            prediction = tree.predict(x[validation])
            choices.append((float(np.mean(np.abs(prediction-y[validation]))), depth, min_leaf, prediction))
    mae, depth, min_leaf, predictions = min(choices, key=lambda row: row[:3])
    report = dict(train_batches=int(train.sum()), validation_batches=int(validation.sum()),
                  split="held_out_distinct_workload_vectors", seed=seed,
                  validation_mae_s=mae,
                  validation_median_absolute_percentage_error=float(np.median(
                      np.abs(predictions-y[validation])/y[validation])),
                  validation_actual_s=y[validation].tolist(), validation_predicted_s=predictions.tolist(),
                  chosen_depth=depth, chosen_min_leaf=min_leaf,
                  guarantee="median estimate; actual wall time and overruns must be reported")
    model = LatencyTree(max_depth=depth, min_leaf=min_leaf).fit(x, y)
    model.calibration = report
    return model, report
