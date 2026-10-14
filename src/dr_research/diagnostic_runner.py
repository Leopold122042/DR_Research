"""Frozen-state diagnosis. Online controllers and their kernels stay unchanged.

Each immutable snapshot contains the historical action pool, restored backup,
and recorded message-refinement mask. Only the offline LP labels are resumable
work. A new process never makes new wall-clock-dependent GNN decisions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import tomllib

import numpy as np

from .candidates import generate_candidates
from .continuation import prepare_continuation_context, try_continuation
from .control import construct_reference, solve_control_window
from .diagnostic_metrics import summarize_snapshot
from .diagnostic_snapshots import reconstruct_snapshots, replay_scores, select_steps
from .experiment_config import digest, seed_for, validate_settings
from .io import load_scenario, scenario_fingerprint
from .types import Action, Plan, State

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    with Path(path).open("rb") as stream:
        result = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_hash(*arrays):
    result = sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        result.update(json.dumps([a.dtype.str, a.shape]).encode())
        result.update(a.tobytes())
    return result.hexdigest()


def atomic_bytes(path, content):
    """Replace only a named diagnostic artifact after a complete, flushed write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


@contextmanager
def output_lock(folder):
    """OS releases the lock even after a process crash; no stale PID deletion."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".diagnosis.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another diagnosis process is using this output directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_config(path):
    with Path(path).open("rb") as stream:
        cfg = tomllib.load(stream)
    keys = {"schema_version", "scales", "selection_seed", "counts", "policy_count",
            "solver_time_limit_s", "solver_threads", "aggregate_pumps"}
    if set(cfg) != keys or cfg["schema_version"] != 1:
        raise ValueError("Unsupported diagnosis configuration fields/version")
    if not cfg["scales"] or len(set(cfg["scales"])) != len(cfg["scales"]) or any(x not in {"S1", "S2", "S3", "S4", "S5"} for x in cfg["scales"]):
        raise ValueError("Invalid diagnosis scales")
    if type(cfg["selection_seed"]) is not int or not 0 <= cfg["selection_seed"] < 2**32:
        raise ValueError("Invalid selection seed")
    if (not cfg["counts"] or len(set(cfg["counts"])) != len(cfg["counts"])
            or any(type(k) is not int or not 1 <= k <= 16384 or k & (k - 1) for k in cfg["counts"])
            or cfg["policy_count"] not in cfg["counts"]):
        raise ValueError("Counts must be unique powers of two and include policy_count")
    if type(cfg["policy_count"]) is not int or cfg["policy_count"] != 256:
        raise ValueError("This diagnostic protocol replays the archived 256-action policy")
    if type(cfg["solver_threads"]) is not int or not 1 <= cfg["solver_threads"] <= 64:
        raise ValueError("Invalid solver threads")
    if type(cfg["aggregate_pumps"]) is not bool:
        raise ValueError("aggregate_pumps must be boolean")
    if type(cfg["solver_time_limit_s"]) not in (int, float) or not np.isfinite(cfg["solver_time_limit_s"]) or cfg["solver_time_limit_s"] <= 0:
        raise ValueError("Invalid offline solver time limit")
    return cfg


def verify_source(source, checkpoint):
    evaluation = Path(source).resolve()
    if not (evaluation / "protocol.json").exists():
        evaluation /= "evaluation"
    archived = read_json(evaluation / "protocol.json")
    protocol = archived["protocol"]
    if digest(protocol) != archived["protocol_id"]:
        raise ValueError("Archived protocol hash mismatch")
    validate_settings(protocol["settings"])
    if protocol["settings"]["control"]["window_steps"] != protocol["settings"]["control"]["safety_window_steps"]:
        raise ValueError("Diagnosis requires the archived control and safety windows to match")
    if file_hash(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the archived experiment")
    # New diagnostic files may be added; every historical kernel must still match.
    for name, expected in protocol["source_sha256"].items():
        if file_hash(Path(__file__).parent / name) != expected:
            raise ValueError("Historical source differs: " + name)
    return evaluation, archived


def contract_for(evaluation, archived, checkpoint, cfg):
    inputs = {}
    week = archived["protocol"]["settings"]["data"]["test_weeks"]
    if len(week) != 1:
        raise ValueError("Select a source containing one held-out test week")
    for scale in cfg["scales"]:
        for name in ("scenario.npz", "gnn_value_r0.json", "gnn_value_r0.npz"):
            rel = f"{scale}_week{week[0]}/{name}"
            inputs[rel] = file_hash(evaluation / rel)
    scripts = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("diagnostic_*.py"))}
    scripts["diagnose_control_gap.py"] = file_hash(ROOT / "scripts" / "diagnose_control_gap.py")
    return dict(schema_version=SCHEMA_VERSION, source=str(evaluation), protocol_id=archived["protocol_id"],
                checkpoint_sha256=file_hash(checkpoint), config=cfg, source_files=inputs,
                diagnostic_sources=scripts, week=week[0])


def emit(**message):
    print(json.dumps(message, ensure_ascii=False, allow_nan=False), flush=True)


def prepare_snapshots(evaluation, archived, checkpoint, cfg, out, *, requested=None, extend_count=None):
    import torch
    from .training import load_checkpoint
    from .candidates import CandidatePool

    settings = archived["protocol"]["settings"]
    torch.set_num_threads(settings["control"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    model, _ = load_checkpoint(checkpoint, archived["protocol"]["physical"], settings, "value")
    week = settings["data"]["test_weeks"][0]
    manifest = []
    common_steps = None
    for scale in cfg["scales"]:
        folder = evaluation / f"{scale}_week{week}"
        s = load_scenario(folder / "scenario.npz")
        report = read_json(folder / "gnn_value_r0.json")
        if (not report["accepted"] or report["executed_steps"] != s.horizon
                or report["protocol_id"] != archived["protocol_id"]
                or report["scenario_fingerprint"] != scenario_fingerprint(s)
                or report["gnn_mode"] != "adaptive"
                or report["candidate_count_requested"] != cfg["policy_count"]):
            raise ValueError("Incomplete or mismatched archived controller report: " + scale)
        selected = select_steps(s, cfg["selection_seed"])
        if common_steps is None:
            common_steps = [row["t"] for row in selected]
        if common_steps != [row["t"] for row in selected]:
            raise ValueError("Selected times differ between scales")
        wanted = []
        for row in selected:
            ident = f"{scale}_t{row['t']:03d}"
            manifest.append(dict(id=ident, scale=scale, **row))
            if requested and ident not in requested:
                continue
            path = out / "snapshots" / ident / "snapshot.json"
            if path.exists():
                meta = read_json(path)
                if not extend_count or meta["raw_count"] >= extend_count:
                    continue
            wanted.append(row["t"])
        if not wanted:
            continue
        with np.load(folder / "gnn_value_r0.npz", allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        snapshots = reconstruct_snapshots(s, arrays, report, settings, wanted)
        for t in wanted:
            snap = snapshots[t]
            state, anchor = snap["state"], snap["anchor"]
            ident = f"{scale}_t{t:03d}"
            target = out / "snapshots" / ident
            target.mkdir(parents=True, exist_ok=True)
            nraw = extend_count or max(cfg["counts"])
            pool = generate_candidates(s, state, anchor, nraw, seed_for(settings["master_seed"], "sobol", s.seed))
            base = CandidatePool([c for c in pool if c.candidate_id <= cfg["policy_count"]], pool.stats)
            archived_scoring = next(row for row in report["scoring_steps"] if row["t"] == t)
            full, adaptive, timing = replay_scores(model, s, state, base, archived_scoring)
            preferred = int(base.ids[np.argsort(adaptive, kind="stable")[0]])
            if preferred != int(arrays["score_preferred_candidate"][t]):
                raise ValueError(f"Archived preferred candidate not reproduced: {ident}")
            selected_id = int(arrays["selected_candidate"][t])
            selected_candidate = next(c for c in base if c.candidate_id == selected_id)
            if not (np.array_equal(selected_candidate.action.power_kw, arrays["power_kw"][t])
                    and np.array_equal(selected_candidate.action.delivery_kw, arrays["delivery_kw"][t])):
                raise ValueError("Archived executed action not reproduced: " + ident)
            context = prepare_continuation_context(s, state, anchor, settings["control"]["safety_window_steps"])
            rows = []
            for index, candidate in enumerate(pool):
                started = perf_counter()
                result = try_continuation(s, state, anchor, candidate, settings["control"]["safety_window_steps"], context=context)
                rows.append(dict(candidate_id=candidate.candidate_id,
                    action_sha256=array_hash(candidate.action.power_kw, candidate.action.delivery_kw),
                    accepted=result.accepted, continuation_reason=result.reason,
                    continuation_s=perf_counter()-started, continuation_profiles=result.attempts,
                    full_score=float(full[index]) if index < len(base) else None,
                    adaptive_score=float(adaptive[index]) if index < len(base) else None))
            order = np.argsort(adaptive, kind="stable")[:settings["control"]["max_continuation_attempts"]]
            actual = next((rows[int(i)]["candidate_id"] for i in order if rows[int(i)]["accepted"]), 0)
            if actual != selected_id:
                raise ValueError("Archived continuation selection not reproduced: " + ident)
            frozen = dict(energy=state.energy_kwh, previous_power=state.previous_power_kw,
                anchor_power=anchor.power_kw, anchor_delivery=anchor.delivery_kw, anchor_energy=anchor.energy_kwh,
                candidate_power=np.stack([c.action.power_kw for c in pool]),
                candidate_delivery=np.stack([c.action.delivery_kw for c in pool]), candidate_ids=pool.ids)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **frozen)
            content = buffer.getvalue()
            # Versioned payload prevents interrupted extensions from invalidating old metadata.
            array_name = f"arrays_{nraw}.npz"
            meta = dict(id=ident, scale=scale, t=t, n_tanks=s.network.n_tanks,
                protocol_id=archived["protocol_id"],
                raw_count=nraw, scenario_rel=f"{scale}_week{week}/scenario.npz",
                scenario_fingerprint=scenario_fingerprint(s), arrays_file=array_name,
                arrays_sha256=sha256(content).hexdigest(), candidates=rows,
                selected_id=selected_id, preferred_id=preferred,
                max_attempts=settings["control"]["max_continuation_attempts"],
                total_power_interval_kw=pool.stats["total_power_interval_kw"],
                diagnostic_generation_s=pool.stats["generation_time_s"],
                diagnostic_score_times=timing, historical_scoring=archived_scoring,
                historical_control_time_s=float(arrays["control_latency_s"][t]),
                historical_generation_s=next(r for r in report["sampling_steps"] if r["t"] == t)["generation_time_s"],
                historical_execution_reason=str(arrays["execution_reason"][t]),
                scoring_scope="replay recorded refinement positions; not a new timed online run")
            if (target / "snapshot.json").exists():
                old = read_json(target / "snapshot.json")
                old_hashes = {r["candidate_id"]: r["action_sha256"] for r in old["candidates"]}
                new_hashes = {r["candidate_id"]: r["action_sha256"] for r in rows}
                if any(new_hashes.get(k) != v for k, v in old_hashes.items()):
                    raise ValueError("Nested candidate identity changed during extension")
            atomic_bytes(target / array_name, content)
            atomic_json(target / "snapshot.json", meta)
            emit(phase="snapshot_prepared", snapshot=ident, candidates=len(rows), archived_decision_verified=True)
    if requested and not set(requested) <= {r["id"] for r in manifest}:
        raise ValueError("Unknown snapshot IDs: " + str(set(requested)-{r["id"] for r in manifest}))
    atomic_json(out / "snapshot_index.json", dict(snapshots=manifest, selection="time/price strata, fixed seed; not selected by observed regret"))
    return manifest


def snapshot_arrays(folder, meta):
    if file_hash(folder / meta["arrays_file"]) != meta["arrays_sha256"]:
        raise ValueError("Frozen snapshot array checksum mismatch")
    with np.load(folder / meta["arrays_file"], allow_pickle=False) as data:
        a = {k: data[k].copy() for k in data.files}
    if len(meta["candidates"]) != len(a["candidate_power"]) or len(meta["candidates"]) != len(a["candidate_delivery"]):
        raise ValueError("Frozen candidate count mismatch")
    for i, row in enumerate(meta["candidates"]):
        if array_hash(a["candidate_power"][i], a["candidate_delivery"][i]) != row["action_sha256"]:
            raise ValueError("Frozen candidate action checksum mismatch")
    return a


def label_identity(meta, arrays, name, fingerprint):
    return digest(dict(snapshot=meta["id"], scenario=meta["scenario_fingerprint"],
        state=array_hash(arrays["energy"], arrays["previous_power"]), kind=name,
        action=fingerprint, protocol=meta["protocol_id"]))


def load_snapshot(out, item, evaluation):
    folder = out / "snapshots" / item["id"]
    meta = read_json(folder / "snapshot.json")
    if meta["id"] != item["id"]:
        raise ValueError("Frozen snapshot ID mismatch")
    a = snapshot_arrays(folder, meta)
    s = load_scenario(evaluation / meta["scenario_rel"])
    if scenario_fingerprint(s) != meta["scenario_fingerprint"]:
        raise ValueError("Frozen scenario changed")
    state = State(meta["t"], a["energy"], a["previous_power"])
    return meta, a, s, state


def solve_label(path, s, state, reference, settings, cfg, *, action=None, key, retry_unknown=False):
    old = read_json(path) if Path(path).exists() else None
    if old and old["identity"] != key:
        raise ValueError("Cached LP label identity differs: " + str(path))
    if old and (old["lp_status"] in {"optimal", "infeasible"} or not retry_unknown):
        return old, False
    solved = solve_control_window(s, state, reference,
        window_steps=settings["control"]["window_steps"], fixed_first_action=action,
        time_limit_s=cfg["solver_time_limit_s"], solver_method=settings["control"]["solver_method"],
        solver_threads=cfg["solver_threads"], aggregate_pumps=cfg["aggregate_pumps"])
    label = dict(identity=key, lp_status=solved.status if solved.status in {"optimal", "infeasible"} else "unknown",
        solver_status=solved.status,
        q_yuan=solved.electricity_yuan if solved.status == "optimal" else None,
        feasible_upper_yuan=solved.electricity_yuan, message=solved.message,
        timings_s=solved.timings_s, statistics=solved.statistics,
        time_limit_s=cfg["solver_time_limit_s"], previous_attempts=[])
    if old:
        label["previous_attempts"] = old.get("previous_attempts", []) + [{k: v for k, v in old.items() if k != "previous_attempts"}]
    atomic_json(path, label)
    return label, True


def band_scenario(s, t, interval):
    lower, upper = s.power_lower_kw.copy(), s.power_upper_kw.copy()
    lower[t], upper[t] = max(lower[t], interval[0]), min(upper[t], interval[1])
    return replace(s, power_lower_kw=lower, power_upper_kw=upper)


def run_labels(out, evaluation, archived, cfg, items, *, max_solves=None, retry_unknown=False):
    settings = archived["protocol"]["settings"]
    count = 0
    for item in items:
        meta, arrays, s, state = load_snapshot(out, item, evaluation)
        samples = []
        reference = construct_reference(s)
        folder = out / "snapshots" / item["id"] / "labels"
        tasks = [("continuous", s, None, "continuous"),
                 ("band", band_scenario(s, state.t, meta["total_power_interval_kw"]), None, digest(meta["total_power_interval_kw"]))]
        for i, row in enumerate(meta["candidates"]):
            action = Action(arrays["candidate_power"][i], arrays["candidate_delivery"][i])
            if array_hash(action.power_kw, action.delivery_kw) != row["action_sha256"]:
                raise ValueError("Candidate action identity mismatch")
            tasks.append((f"candidate_{row['candidate_id']:05d}", s, action, row["action_sha256"]))
        for name, case, action, fingerprint in tasks:
            path = folder / (name + ".json")
            if meta["protocol_id"] != archived["protocol_id"]:
                raise ValueError("Frozen snapshot protocol changed")
            key = label_identity(meta, arrays, name, fingerprint)
            cached = read_json(path) if path.exists() else None
            pending = not cached or (retry_unknown and cached["lp_status"] not in {"optimal", "infeasible"})
            if pending and max_solves is not None and count >= max_solves:
                emit(phase="paused_at_requested_limit", new_lp_solves=count)
                return
            label, fresh = solve_label(path, case, state, reference, settings, cfg,
                                      action=action, key=key, retry_unknown=retry_unknown)
            if fresh:
                count += 1
                samples.append(label["timings_s"]["total"])
                # Report measured throughput, never extrapolate small scales to larger ones.
                emit(phase="label", snapshot=meta["id"], label=name, status=label["lp_status"],
                     new_lp_solves=count, last_seconds=samples[-1],
                     remaining_in_snapshot=sum(not (folder / (task[0]+".json")).exists() for task in tasks),
                     estimated_seconds_per_similar_solve=float(np.median(samples[-10:])))
        summarize_output(out, cfg)


def summarize_output(out, cfg):
    index = read_json(out / "snapshot_index.json")["snapshots"]
    reports, flat = [], []
    unknown = dict(lp_status="unknown", q_yuan=None)
    for item in index:
        folder = out / "snapshots" / item["id"]
        if not (folder / "snapshot.json").exists():
            reports.append(dict(**item, status="not_prepared"))
            continue
        meta = read_json(folder / "snapshot.json")
        arrays = snapshot_arrays(folder, meta)
        manifest_path = out / "diagnosis_manifest.json"
        if manifest_path.exists() and meta["protocol_id"] != read_json(manifest_path)["protocol_id"]:
            raise ValueError("Snapshot protocol differs from output manifest")
        labels = folder / "labels"
        identities = {"continuous": "continuous", "band": digest(meta["total_power_interval_kw"]),
                      **{f"candidate_{r['candidate_id']:05d}": r["action_sha256"] for r in meta["candidates"]}}
        def get(name):
            if not (labels / (name + ".json")).exists():
                return unknown
            value = read_json(labels / (name + ".json"))
            if value.get("identity") != label_identity(meta, arrays, name, identities[name]):
                raise ValueError("Cached LP label identity differs during summary: " + name)
            return value
        records = []
        offline_seconds = 0.0
        offline_attempts = 0
        for label_file in labels.glob("*.json"):
            label = read_json(label_file)
            attempts = label.get("previous_attempts", []) + [label]
            offline_seconds += sum(a.get("timings_s", {}).get("total", 0.) for a in attempts)
            offline_attempts += len(attempts)
        for row in meta["candidates"]:
            label = get(f"candidate_{row['candidate_id']:05d}")
            records.append({**row, "lp_status": label["lp_status"], "q_yuan": label["q_yuan"]})
            flat.append(dict(snapshot=item["id"], category=item["category"], scale=item["scale"], t=item["t"], **records[-1]))
        counts = sorted(set(cfg["counts"] + [meta["raw_count"]]))
        report = summarize_snapshot(records, get("continuous"), get("band"),
            executed_id=meta["selected_id"], max_attempts=meta.get("max_attempts", 8), n_tanks=meta["n_tanks"],
            raw_counts=counts, policy_count=cfg["policy_count"])
        report["timing"] = dict(offline_lp_seconds=offline_seconds, offline_lp_attempts=offline_attempts,
            diagnostic_generation_s=meta.get("diagnostic_generation_s"),
            diagnostic_score_replay=meta.get("diagnostic_score_times"),
            diagnostic_all_candidate_continuation_s=sum(r.get("continuation_s", 0.) for r in records),
            archived_online_control_s=meta.get("historical_control_time_s"),
            archived_online_generation_s=meta.get("historical_generation_s"))
        reports.append(dict(**item, **report))
        atomic_json(folder / "summary.json", reports[-1])
    summary = dict(schema_version=SCHEMA_VERSION, snapshots=reports,
        scope="Frozen observed GNN states; overlapping optimal-continuation window gaps. Do not sum as weekly bill attribution.",
        timing_scope="Historical online times and new offline label/score replay times are distinct.",
        completed_snapshot_count=sum(r.get("status") == "complete" for r in reports),
        planned_snapshot_count=len(index))
    atomic_json(out / "summary.json", summary)
    keys = ["snapshot", "category", "scale", "t", "candidate_id", "lp_status", "q_yuan", "accepted", "full_score", "adaptive_score", "continuation_s", "continuation_reason", "action_sha256"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flat)
    atomic_bytes(out / "candidate_labels.csv", buffer.getvalue().encode("utf-8-sig"))
    table_rows = []
    for report in reports:
        for count, entry in report.get("counts", {}).items():
            d = entry.get("decomposition") or {}
            signed = entry.get("signed_differences") or {}
            table_rows.append(dict(snapshot=report["id"], scale=report["scale"], t=report["t"],
                category=report["category"], raw_count=int(count), status=report["status"],
                actual_candidates=entry["actual_candidate_count"],
                unknown_labels=entry["lp_status_counts"]["unknown"],
                continuous_yuan=report["baselines"]["continuous"]["q_yuan"],
                coverage_yuan=(entry.get("coverage") or {}).get("yuan"),
                coverage_yuan_per_tank=(entry.get("coverage") or {}).get("yuan_per_tank"),
                coverage_percent=(entry.get("coverage") or {}).get("percent_of_continuous_lp"),
                power_band_restriction_yuan=(entry.get("power_band_restriction") or {}).get("yuan"),
                certificate_miss_yuan=(d.get("certificate_miss") or {}).get("yuan"),
                ranking_and_attempt_limit_yuan=(d.get("ranking_and_attempt_limit") or {}).get("yuan"),
                total_gap_yuan=(d.get("total_gap") or {}).get("yuan"),
                full_ranking_yuan=(signed.get("full_ranking_D_minus_C") or {}).get("yuan"),
                message_fidelity_yuan=(signed.get("message_fidelity_E_minus_D") or {}).get("yuan"),
                attempt_limit_yuan=(signed.get("attempt_limit_F_minus_E") or {}).get("yuan"),
                execution_match=entry.get("execution_match")))
    if table_rows:
        table = io.StringIO(newline="")
        writer = csv.DictWriter(table, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
        atomic_bytes(out / "snapshot_summary.csv", table.getvalue().encode("utf-8-sig"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/rerun_20261009_194453")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/full_20261009/hpo/selected.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cause_diagnosis.toml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="restore snapshots and scores, without any LP labels")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-solves", type=int, help="pause after this many new offline LP solves; rerun same output to resume")
    parser.add_argument("--retry-unknown", action="store_true")
    parser.add_argument("--solver-time-limit", type=float, help="offline retry limit; does not alter the control problem")
    parser.add_argument("--snapshots", nargs="+", help="process selected IDs shown in snapshot_index.json")
    parser.add_argument("--extend-count", type=int, help="extend selected snapshots to a larger nested power-of-two pool; coverage only")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.max_solves is not None and args.max_solves < 1:
        parser.error("--max-solves must be positive")
    if args.extend_count is not None and (not args.snapshots or args.extend_count <= max(cfg["counts"]) or args.extend_count > 16384 or args.extend_count & (args.extend_count-1)):
        parser.error("--extend-count requires --snapshots and a larger power of two up to 16384")
    if args.solver_time_limit is not None and (not np.isfinite(args.solver_time_limit) or args.solver_time_limit <= 0):
        parser.error("--solver-time-limit must be finite and positive")
    out = args.out.resolve()
    evaluation, archived = verify_source(args.source, args.checkpoint)
    if out == evaluation or out in evaluation.parents or evaluation in out.parents:
        parser.error("Use a separate diagnosis output directory outside the source evaluation")
    contract = contract_for(evaluation, archived, args.checkpoint, cfg)
    with output_lock(out):
        path = out / "diagnosis_manifest.json"
        if path.exists():
            if read_json(path) != contract:
                raise ValueError("Resume contract changed; use a fresh output directory")
        else:
            if any(p.name != ".diagnosis.lock" for p in out.iterdir()):
                raise ValueError("New output directory is not empty")
            atomic_json(path, contract)
        if args.summarize_only:
            summary = summarize_output(out, cfg)
        else:
            index = prepare_snapshots(evaluation, archived, args.checkpoint, cfg, out,
                                      requested=args.snapshots, extend_count=args.extend_count)
            items = [x for x in index if not args.snapshots or x["id"] in args.snapshots]
            run_cfg = {**cfg, "solver_time_limit_s": args.solver_time_limit or cfg["solver_time_limit_s"]}
            try:
                if not args.prepare_only:
                    run_labels(out, evaluation, archived, run_cfg, items,
                               max_solves=args.max_solves, retry_unknown=args.retry_unknown)
            except KeyboardInterrupt:
                summarize_output(out, cfg)
                emit(phase="interrupted", message="Completed labels saved; rerun the same command to resume")
                return 130
            summary = summarize_output(out, cfg)
        emit(phase="summary", completed=summary["completed_snapshot_count"],
             planned=summary["planned_snapshot_count"], output=str(out))
    return 0
