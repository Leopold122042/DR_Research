"""Teacher-relative message error prediction and measured budgeted refinement.

Full messages are evaluated only after a group has been selected online. The
offline pairing/calibration helpers are explicit and separate from scheduling.
"""
from __future__ import annotations

import copy
import platform
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .latency import (fit_latency_profile, measure_refinement_batches,
                      workload_features)


MODES = ("coarse", "full", "random", "adaptive")


class MessageErrorPredictor(nn.Module):
    """Predict ||yH-yL|| from fixed coarse-visible features only (equation 25)."""

    def __init__(self, input_dim: int, hidden_dim: int = 24):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("Error predictor dimensions must be positive")
        self.input_dim, self.hidden_dim = int(input_dim), int(hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        self.register_buffer("target_scale", torch.ones(()))
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    @property
    def architecture(self):
        return dict(input_dim=self.input_dim, hidden_dim=self.hidden_dim)

    def forward(self, features):
        if features.shape[-1] != self.input_dim:
            raise ValueError("Wrong coarse-visible feature width")
        normalized = (features-self.feature_mean)/self.feature_scale
        return torch.nn.functional.softplus(self.net(normalized).squeeze(-1))*self.target_scale


def fit_error_predictor(features, residual_norms, *, validation_features=None,
                        validation_norms=None, epochs=80, hidden_dim=24,
                        learning_rate=.002, batch_size=1024, seed=42):
    """Fit on training-only pairs; explicit held-out states are preferred."""
    x = torch.as_tensor(features, dtype=torch.float32).detach().cpu().reshape(-1, features.shape[-1])
    y = torch.as_tensor(residual_norms, dtype=torch.float32).detach().cpu().reshape(-1)
    if (len(x) < 2 or len(x) != len(y) or not torch.isfinite(x).all()
            or not torch.isfinite(y).all() or (y < 0).any()
            or epochs < 1 or batch_size < 1 or learning_rate <= 0):
        raise ValueError("Invalid paired message errors or training settings")
    if (validation_features is None) != (validation_norms is None):
        raise ValueError("Supply both validation features and norms")
    generator = torch.Generator().manual_seed(seed)
    if validation_features is None:
        order = torch.randperm(len(x), generator=generator)
        count = max(1, len(x)//5)
        vx, vy, x, y = x[order[:count]], y[order[:count]], x[order[count:]], y[order[count:]]
        split = "held_out_message_positions; use explicit state-separated validation for experiments"
    else:
        vx = torch.as_tensor(validation_features, dtype=torch.float32).detach().cpu().reshape(-1, x.shape[-1])
        vy = torch.as_tensor(validation_norms, dtype=torch.float32).detach().cpu().reshape(-1)
        if not len(vx) or len(vx) != len(vy) or not torch.isfinite(vx).all() or not torch.isfinite(vy).all() or (vy < 0).any():
            raise ValueError("Invalid held-out message pairs")
        split = "caller_supplied_held_out_states"
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = MessageErrorPredictor(x.shape[1], hidden_dim)
    model.feature_mean.copy_(x.mean(0))
    model.feature_scale.copy_(x.std(0, unbiased=False).clamp_min(1e-5))
    model.target_scale.copy_(y.mean().clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best, best_weights, history = float("inf"), None, []
    started = perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        train_sum = 0.
        for first in range(0, len(order), batch_size):
            indices = order[first:first+batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = (((model(x[indices])-y[indices])/model.target_scale)**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite error-predictor training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            train_sum += float(loss.detach())*len(indices)
        model.eval()
        with torch.no_grad():
            val = float((((model(vx)-vy)/model.target_scale)**2).mean())
        history.append(dict(epoch=epoch+1, training_mse=train_sum/len(x), validation_mse=val))
        if val < best:
            best, best_weights = val, copy.deepcopy(model.state_dict())
            best_epoch = epoch+1
    if best_weights is None:
        raise RuntimeError("No finite message-error checkpoint")
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        absolute = (model(vx)-vy).abs()
    return model, dict(train_positions=len(x), validation_positions=len(vx), split=split,
                       best_epoch=best_epoch, best_validation_mse=best,
                       validation_mae_latent=float(absolute.mean()),
                       training_s=perf_counter()-started, history=history)


def paired_error_examples(model, prepared_pools):
    """Offline-only equation 25 labels; never called by evaluate_candidates."""
    features, targets = [], []
    with torch.no_grad():
        for prepared in prepared_pools:
            visible = model.error_features(prepared)
            coarse = model.coarse_messages(prepared)
            complete = model.complete_messages(prepared)
            features.append(visible.reshape(-1, visible.shape[-1]).detach().cpu())
            targets.append(torch.linalg.vector_norm(complete-coarse, dim=-1).detach().cpu())
    if not features:
        raise ValueError("At least one offline candidate pool is required")
    return torch.cat(features), torch.cat(targets)


def _numpy(values):
    return values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)


def shortlist(scores, *, leader_count=8, coverage_count=2, rng):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid candidate scores")
    ordered = np.argsort(scores, kind="stable")
    leaders = ordered[:max(1, min(leader_count, len(ordered)))]
    remaining = ordered[len(leaders):]
    coverage = (rng.choice(remaining, min(coverage_count, len(remaining)), replace=False)
                if len(remaining) and coverage_count else np.empty(0, dtype=int))
    return np.concatenate((leaders, coverage)).astype(np.int64)


def receiver_groups(receiver_index, candidate_ids, refined, *, group_size=64):
    """Chunk allowed charging positions by candidate and receiver locality."""
    receivers = np.asarray(receiver_index, dtype=np.int64).reshape(-1)
    mask = np.asarray(refined, dtype=bool)
    if group_size < 1 or mask.ndim != 2 or mask.shape[1] != len(receivers):
        raise ValueError("Invalid refinement group layout")
    edge_order = np.argsort(receivers, kind="stable")
    groups = []
    for candidate in candidate_ids:
        if not 0 <= candidate < len(mask):
            raise ValueError("Candidate index outside refinement cache")
        available = edge_order[~mask[int(candidate), edge_order]]
        for start in range(0, len(available), group_size):
            groups.append(int(candidate)*len(receivers)+available[start:start+group_size])
    return groups


def _workload(prepared, positions):
    cold = 0 if prepared.full_projection_cache_ready else 2*prepared.n_pumps+prepared.n_tanks
    return workload_features(positions, _numpy(prepared.receiver_index), prepared.n_pumps,
                             cold_projections=cold)


def calibrate_latency(model, prepared_pools, *, group_sizes=(1, 4, 16, 64),
                      repeats=3, warmup=1, seed=42):
    """Measure real cold/warm incremental batches on representative graph sizes.

    Coarse cache construction and reset are untimed. Lazy complete-branch public
    projections, selected messages, node updates, and readouts are timed as one
    operation. No test costs or labels are used by hardware calibration.
    """
    if not group_sizes or any(type(size) is not int or size < 1 for size in group_sizes):
        raise ValueError("Profiling group sizes must be positive integers")
    records, populations = [], []
    for prepared in prepared_pools:
        original_full_public = prepared.full_public
        with torch.no_grad():
            base = model.start_cache(prepared)
        edge_order = np.argsort(_numpy(prepared.receiver_index), kind="stable")
        populations.append(dict(candidates=prepared.candidate_count, pumps=prepared.n_pumps,
                                tanks=prepared.n_tanks, buildings=prepared.n_buildings))
        workloads = []
        try:
            for size in sorted({min(size, prepared.n_pumps) for size in group_sizes}):
                # End blocks include any less populated receiver groups.
                alternatives = [edge_order[:size], edge_order[-size:]]
                if np.array_equal(*alternatives):
                    alternatives = alternatives[:1]
                for positions in alternatives:
                    for cold in (True, False):
                        features = workload_features(positions, _numpy(prepared.receiver_index),
                            prepared.n_pumps, cold_projections=(2*prepared.n_pumps+prepared.n_tanks) if cold else 0)

                        def factory(positions=positions.copy(), cold=cold):
                            cache = copy.copy(base)
                            for key, value in vars(base).items():
                                if isinstance(value, torch.Tensor):
                                    setattr(cache, key, value.clone())
                            prepared.full_public = None
                            if not cold:
                                with torch.no_grad():
                                    model.complete_messages(prepared, positions[:1])

                            def operation():
                                with torch.no_grad():
                                    model.refine(cache, positions)
                                    model.score(cache)
                            return operation

                        workloads.append(dict(features=features, factory=factory))
            # This implementation is calibrated for CPU tensors. Prevent a
            # silently asynchronous GPU timing profile.
            if base.messages.device.type != "cpu":
                raise ValueError("Current latency calibration requires CPU model execution")
            records.extend(measure_refinement_batches(workloads, repeats=repeats, warmup=warmup))
        finally:
            prepared.full_public = original_full_public
    model_latency, report = fit_latency_profile(records, seed=seed)
    report.update(platform=platform.platform(), torch_version=str(torch.__version__),
                  torch_threads=torch.get_num_threads(), device="cpu",
                  dtype=str(next(model.parameters()).dtype), architecture=model.architecture,
                  populations=populations,
                  warmup=warmup, repeats=repeats, measured_batches=records)
    model_latency.calibration = report
    return model_latency, report


def profile_compatibility(model, prepared, latency_model):
    """Reject hardware/model changes and graphs exceeding calibrated pool size."""
    # Non-tree injected timing providers can implement their own in_profile.
    if not hasattr(latency_model, "calibration"):
        return []
    metadata = latency_model.calibration
    if not isinstance(metadata, dict):
        return ["missing_hardware_calibration_metadata"]
    expected = dict(platform=platform.platform(), torch_version=str(torch.__version__),
                    torch_threads=torch.get_num_threads(), device=str(prepared.device),
                    dtype=str(next(model.parameters()).dtype), architecture=model.architecture)
    problems = ["changed_"+key for key, value in expected.items() if metadata.get(key) != value]
    matching = [row for row in metadata.get("populations", [])
                if (row["pumps"], row["tanks"], row["buildings"]) ==
                   (prepared.n_pumps, prepared.n_tanks, prepared.n_buildings)]
    if not matching:
        problems.append("unprofiled_graph_dimensions")
    elif prepared.candidate_count > max(row["candidates"] for row in matching):
        problems.append("candidate_pool_exceeds_profile")
    return problems


def evaluate_candidates(model, scenario, state, actions, *, mode="adaptive",
                        error_predictor=None, latency_model=None, time_budget_s=1.,
                        seed=42, shortlist_size=8, coverage_count=2, group_size=64,
                        max_rounds=32, reserve_s=.001, strict_profile=True):
    """Return current+future scores and complete measured scoring accounting.

    Callers retain continuation acceptance and whole-control deadline checks.
    A median latency estimate can underpredict; actual overruns are reported.
    """
    if mode not in MODES or not np.isfinite(time_budget_s) or time_budget_s <= 0:
        raise ValueError("Invalid message-fidelity mode or time budget")
    if min(shortlist_size, group_size) < 1 or min(coverage_count, max_rounds, reserve_s) < 0:
        raise ValueError("Invalid scheduler limits")
    if mode in {"adaptive", "random"} and latency_model is None:
        raise ValueError("Refinement requires a calibrated incremental latency model")
    if mode == "adaptive" and error_predictor is None:
        raise ValueError("Adaptive refinement requires a trained coarse-visible error predictor")
    started = perf_counter()
    report = dict(mode=mode, prepare_s=0., coarse_scoring_s=0., error_prediction_s=0.,
                  sensitivity_s=0., scheduling_s=0., refinement_s=0., full_scoring_s=0.,
                  rounds=0, refined_messages=0, predicted_refinement_s=0.,
                  out_of_profile_groups=0, recalibration_required=False, selected_groups=[])
    clock = perf_counter()
    with torch.no_grad():
        prepared = model.prepare(scenario, state, actions)
    report["prepare_s"] = perf_counter()-clock
    total_positions = prepared.candidate_count*prepared.n_pumps
    mismatches = (profile_compatibility(model, prepared, latency_model)
                  if mode in {"adaptive", "random"} else [])
    report["profile_mismatches"] = mismatches
    report["recalibration_required"] = bool(mismatches)
    profile_allowed = not mismatches or not strict_profile
    if mode == "full":
        clock = perf_counter()
        with torch.no_grad():
            future = model.full(prepared)
        report["full_scoring_s"] = perf_counter()-clock
        report["refined_messages"] = total_positions
        scores = _numpy(future)+_numpy(prepared.current_yuan)
    else:
        clock = perf_counter()
        with torch.no_grad():
            cache = model.start_cache(prepared)
            scores = _numpy(model.score(cache))+_numpy(prepared.current_yuan)
        report["coarse_scoring_s"] = perf_counter()-clock
        errors = None
        if mode == "adaptive" and profile_allowed and perf_counter()-started+reserve_s < time_budget_s:
            clock = perf_counter()
            with torch.no_grad():
                errors = _numpy(error_predictor(model.error_features(cache)))
            if errors.shape != (prepared.candidate_count, prepared.n_pumps) or not np.isfinite(errors).all() or np.any(errors < 0):
                raise ValueError("Invalid coarse-visible message errors")
            report["error_prediction_s"] = perf_counter()-clock
        rng = np.random.default_rng(seed)
        for round_index in range(max_rounds if mode in {"adaptive", "random"} and profile_allowed else 0):
            if perf_counter()-started+reserve_s >= time_budget_s:
                break
            clock = perf_counter()
            selected_candidates = shortlist(scores, leader_count=shortlist_size,
                                             coverage_count=coverage_count, rng=rng)
            groups = receiver_groups(_numpy(prepared.receiver_index), selected_candidates,
                                     _numpy(cache.refined), group_size=group_size)
            report["scheduling_s"] += perf_counter()-clock
            if not groups or (mode == "adaptive" and errors is None):
                break
            if mode == "adaptive":
                clock = perf_counter()
                sensitivities = _numpy(model.sensitivity(cache, selected_candidates))
                if sensitivities.shape != errors.shape or not np.isfinite(sensitivities).all() or np.any(sensitivities < 0):
                    raise ValueError("Invalid candidate-message sensitivities")
                priority = (sensitivities*errors).reshape(-1)
                report["sensitivity_s"] += perf_counter()-clock
            else:
                priority = None
            clock = perf_counter()
            choices = []
            for positions in groups:
                workload = _workload(prepared, positions)
                in_profile = latency_model.in_profile(workload)
                if not in_profile:
                    report["out_of_profile_groups"] += 1
                    report["recalibration_required"] = True
                    if strict_profile:
                        continue
                predicted = float(latency_model.predict(workload))
                if not np.isfinite(predicted) or predicted <= 0:
                    raise ValueError("Invalid predicted refinement latency")
                utility = float(priority[positions].sum()) if priority is not None else float(rng.random())
                if utility > 0:
                    choices.append((utility/predicted if priority is not None else utility,
                                    predicted, positions, workload))
            remaining = time_budget_s-(perf_counter()-started)-reserve_s
            fitting = [item for item in choices if item[1] <= remaining]
            report["scheduling_s"] += perf_counter()-clock
            if not fitting:
                break
            _, predicted, positions, workload = max(fitting, key=lambda item: item[0])
            if perf_counter()-started+predicted+reserve_s > time_budget_s:
                break
            clock = perf_counter()
            with torch.no_grad():
                stats = model.refine(cache, positions)
                scores = _numpy(model.score(cache))+_numpy(prepared.current_yuan)
            measured = perf_counter()-clock
            report["refinement_s"] += measured
            report["predicted_refinement_s"] += predicted
            report["rounds"] += 1
            report["selected_groups"].append(dict(round=round_index, positions=positions.tolist(),
                features=workload.tolist(), predicted_s=predicted, actual_s=measured,
                new_messages=int(stats["new_messages"])))
        report["refined_messages"] = int(_numpy(cache.refined).sum())
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(actions),) or not np.isfinite(scores).all():
        raise ValueError("Nonfinite or malformed mixed candidate scores")
    elapsed = perf_counter()-started
    report.update(scoring_time_s=elapsed, budget_s=time_budget_s,
                  budget_exhausted=elapsed+reserve_s >= time_budget_s,
                  overrun_s=max(0., elapsed-time_budget_s), total_messages=total_positions,
                  refinement_fraction=report["refined_messages"]/max(1, total_positions))
    return scores, report
