"""Pure, auditable summaries for frozen-state candidate diagnostics.

These are overlapping-window decision differences, not additive weekly savings.
Candidate IDs are Sobol-prefix IDs, not positions in a deduplicated pool.
"""

from __future__ import annotations

import math
from typing import Any


_TOLERANCE_YUAN = 1e-4
_KNOWN_STATUSES = {"optimal", "infeasible", "unknown"}


def _label(record: dict[str, Any], name: str) -> dict[str, Any]:
    status = record.get("lp_status")
    if status not in _KNOWN_STATUSES:
        raise ValueError(f"{name}: unsupported lp_status {status!r}")
    value = record.get("q_yuan")
    if status == "optimal":
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{name}: optimal label needs finite q_yuan")
        value = float(value)
    elif value is not None:
        raise ValueError(f"{name}: non-optimal label must have q_yuan=None")
    return {"lp_status": status, "q_yuan": value}


def _choice(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "candidate_id": record["candidate_id"],
        "lp_status": record["lp_status"],
        "q_yuan": record["q_yuan"],
        "accepted": record["accepted"],
        "lp_feasible": {"optimal": True, "infeasible": False, "unknown": None}[record["lp_status"]],
    }


def _difference(value: float, baseline: float, n_tanks: int) -> dict[str, Any]:
    return {
        "yuan": float(value),
        "yuan_per_tank": float(value / n_tanks),
        "percent_of_continuous_lp": (
            float(100.0 * value / baseline) if baseline > 0 else None
        ),
    }


def _nonnegative(value: float, name: str, warnings: list[str]) -> float:
    if value < -_TOLERANCE_YUAN:
        raise ValueError(f"{name} is negative ({value:.12g} yuan); check model comparability")
    if value < 0:
        warnings.append(f"{name}: {value:.12g} yuan is within numerical tolerance")
    # Preserve the raw small residual so additive identities remain exact.
    return value


def _ranked(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    for record in records:
        value = record.get(key)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"candidate {record['candidate_id']}: {key} must be finite")
    # Python's stable sort preserves candidate-pool ordering for score ties.
    return sorted(records, key=lambda record: float(record[key]))


def _first_accepted(records: list[dict[str, Any]]) -> dict[str, Any]:
    return next(record for record in records if record["accepted"])


def summarize_snapshot(
    records: list[dict[str, Any]],
    continuous: dict[str, Any],
    band: dict[str, Any],
    executed_id: int,
    max_attempts: int = 8,
    n_tanks: int = 1,
    raw_counts: tuple[int, ...] = (64, 256),
    policy_count: int = 256,
) -> dict[str, Any]:
    """Summarize matched LP labels, acceptance outcomes, and recorded scores.

    A record has ``candidate_id``, ``lp_status``, ``q_yuan``, ``accepted``,
    ``full_score``, and ``adaptive_score``. ID 0 is the certified backup;
    positive IDs are retained raw Sobol indices, so gaps from deduplication
    are allowed. Unknown or infeasible LP labels must have ``q_yuan=None``.
    Baselines use the same LP status and cost fields.

    Only ``policy_count`` uses archived scores and reconstructs the policy.
    Other prefixes report candidate coverage alone, avoiding the implication
    that a subset of 256-pool scores is an independently run 64-pool policy.
    Tied scores retain input ordering, matching a stable controller argsort.

    An exact decomposition requires every LP label in its prefix, plus an
    optimal continuous LP. Unknown labels are reported explicitly and never
    silently discarded to claim an oracle minimum. Recorded execution
    mismatch is reported (rather than hidden or substituted).
    """
    if not isinstance(n_tanks, int) or isinstance(n_tanks, bool) or n_tanks <= 0:
        raise ValueError("n_tanks must be a positive integer")
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not raw_counts or any(not isinstance(count, int) or count < 1 for count in raw_counts):
        raise ValueError("raw_counts must contain positive integers")
    if len(set(raw_counts)) != len(raw_counts):
        raise ValueError("raw_counts must be unique")
    continuous_label = _label(continuous, "continuous baseline")
    band_label = _label(band, "power-band baseline")
    normal: list[dict[str, Any]] = []
    ids: set[int] = set()
    for source in records:
        candidate_id = source.get("candidate_id")
        if not isinstance(candidate_id, int) or isinstance(candidate_id, bool) or candidate_id < 0:
            raise ValueError("candidate_id must be a nonnegative integer")
        if candidate_id in ids:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        ids.add(candidate_id)
        label = _label(source, f"candidate {candidate_id}")
        accepted = source.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError(f"candidate {candidate_id}: accepted must be boolean")
        if accepted and label["lp_status"] == "infeasible":
            raise ValueError(f"candidate {candidate_id}: accepted but LP proved infeasible")
        normal.append({**source, **label})
    if 0 not in ids:
        raise ValueError("candidate pool must contain backup candidate_id 0")
    backup = next(record for record in normal if record["candidate_id"] == 0)
    if not backup["accepted"]:
        raise ValueError("backup candidate_id 0 must have a valid continuation certificate")
    if continuous_label["lp_status"] == "infeasible":
        raise ValueError("continuous baseline proved infeasible despite a certified backup")
    if band_label["lp_status"] == "infeasible":
        raise ValueError("power-band baseline proved infeasible despite a certified backup in its interval")

    warnings: list[str] = []
    counts: dict[str, Any] = {}
    for count in raw_counts:
        pool = [record for record in normal if record["candidate_id"] <= count]
        status_counts = {status: sum(record["lp_status"] == status for record in pool)
                         for status in sorted(_KNOWN_STATUSES)}
        complete = status_counts["unknown"] == 0
        optimum = [record for record in pool if record["lp_status"] == "optimal"]
        accepted_optimum = [record for record in optimum if record["accepted"]]
        best = min(optimum, key=lambda record: record["q_yuan"], default=None)
        accepted_best = min(accepted_optimum, key=lambda record: record["q_yuan"], default=None)
        report: dict[str, Any] = {
            "raw_count": count,
            "actual_candidate_count": len(pool),
            "lp_status_counts": status_counts,
            "accepted_count": sum(record["accepted"] for record in pool),
            "lp_feasible_but_not_accepted_count": sum(not record["accepted"] for record in optimum),
            "labels_complete": complete,
            "best_known_feasible": _choice(best),
            "best_known_accepted": _choice(accepted_best),
            "pool_oracle": _choice(best) if complete else None,
            "accepted_oracle": _choice(accepted_best) if complete else None,
            "coverage": None,
            "power_band_restriction": None,
            "coverage_after_power_band": None,
            "score_scope": "archived_policy_pool" if count == policy_count else "coverage_only",
        }
        baseline_value = continuous_label["q_yuan"]
        # Even an incomplete pool's observed feasible actions must respect LP lower bounds.
        if best is not None and baseline_value is not None:
            _nonnegative(best["q_yuan"] - baseline_value, f"count {count} observed coverage", warnings)
        if best is not None and band_label["q_yuan"] is not None:
            _nonnegative(best["q_yuan"] - band_label["q_yuan"], f"count {count} observed band gap", warnings)
        if band_label["q_yuan"] is not None and baseline_value is not None:
            restriction = _nonnegative(band_label["q_yuan"] - baseline_value,
                                       f"count {count} power-band restriction", warnings)
            report["power_band_restriction"] = _difference(restriction, baseline_value, n_tanks)
        if complete and best is not None and baseline_value is not None:
            coverage = _nonnegative(best["q_yuan"] - baseline_value, f"count {count} coverage", warnings)
            report["coverage"] = _difference(coverage, baseline_value, n_tanks)
            if band_label["q_yuan"] is not None:
                report["coverage_after_power_band"] = _difference(
                    best["q_yuan"] - band_label["q_yuan"], baseline_value, n_tanks)

        if count == policy_count:
            full_ranked = _ranked(pool, "full_score")
            full = _first_accepted(full_ranked)
            adaptive_ranked = _ranked(pool, "adaptive_score")
            adaptive = _first_accepted(adaptive_ranked)
            attempted = adaptive_ranked[:max_attempts]
            top_accepted = next((record for record in attempted if record["accepted"]), None)
            policy = top_accepted if top_accepted is not None else backup
            report.update({
                "groups": {
                    "A_continuous_lp": continuous_label,
                    "B_pool_oracle": _choice(best) if complete else None,
                    "C_accepted_oracle": _choice(accepted_best) if complete else None,
                    "D_full_unlimited": _choice(full),
                    "E_adaptive_unlimited": _choice(adaptive),
                    "F_current_policy": _choice(policy),
                },
                "recorded_executed_id": executed_id,
                "execution_match": policy["candidate_id"] == executed_id,
                "raw_preferred": {
                    "full": _choice(full_ranked[0]),
                    "adaptive": _choice(adaptive_ranked[0]),
                },
                "fallback_due_to_attempt_limit": top_accepted is None,
                "decomposition": None,
                "signed_differences": None,
                "decomposition_invalid_reason": (
                    "archived_execution_not_reproduced" if policy["candidate_id"] != executed_id else None
                ),
            })
            if (report["execution_match"] and complete and baseline_value is not None
                    and best is not None and accepted_best is not None):
                # An accepted action cannot be LP-infeasible; completeness ensures finite costs.
                q_f, q_a, q_exec = best["q_yuan"], accepted_best["q_yuan"], policy["q_yuan"]
                parts = {
                    "coverage": _nonnegative(q_f - baseline_value, "coverage", warnings),
                    "certificate_miss": _nonnegative(q_a - q_f, "certificate miss", warnings),
                    "ranking_and_attempt_limit": _nonnegative(q_exec - q_a, "ranking and attempt limit", warnings),
                }
                total = q_exec - baseline_value
                report["decomposition"] = {
                    "total_gap": _difference(total, baseline_value, n_tanks),
                    **{key: _difference(value, baseline_value, n_tanks) for key, value in parts.items()},
                    "identity_residual_yuan": float(total - sum(parts.values())),
                    "nonnegative_tolerance_yuan": _TOLERANCE_YUAN,
                    "refers_to": "reconstructed_current_policy",
                }
                report["signed_differences"] = {
                    "full_ranking_D_minus_C": _difference(full["q_yuan"] - q_a, baseline_value, n_tanks),
                    "message_fidelity_E_minus_D": _difference(adaptive["q_yuan"] - full["q_yuan"], baseline_value, n_tanks),
                    "attempt_limit_F_minus_E": _difference(q_exec - adaptive["q_yuan"], baseline_value, n_tanks),
                }
        counts[str(count)] = report

    policy_report = counts.get(str(policy_count))
    status = "complete" if (all(report["labels_complete"] for report in counts.values())
                            and continuous_label["lp_status"] == "optimal"
                            and band_label["lp_status"] == "optimal") else "incomplete_labels"
    if policy_report is not None and not policy_report["execution_match"]:
        status = "execution_mismatch"
    return {
        "schema_version": 1,
        "status": status,
        "n_tanks": n_tanks,
        "max_attempts": max_attempts,
        "policy_count": policy_count,
        "baselines": {"continuous": continuous_label, "power_band": band_label},
        "counts": counts,
        "warnings": warnings,
        "interpretation": "Frozen-state overlapping-window gaps; do not sum them as weekly causal cost shares.",
    }
