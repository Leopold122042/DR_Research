"""CLI for candidate-value GNN training and paired weekly LP-MPC comparison."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy
import torch

from .config import load_config
from .dataset import case_seed, prepare_dataset
from .dispatch import METHODS, run_controller
from .experiment_config import digest, load_settings, seed_for
from .io import load_scenario, save_scenario, scenario_fingerprint, write_json
from .latency import LatencyTree
from .scenario import generate_scenario
from .scheduler import MODES
from .training import load_checkpoint, tune_models


def progress(event):
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def _atomic_json(path, value):
    """Replace only this run's resume index or derived final manifest."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def compare(physical, settings, hpo_path, output, report_progress=progress, *,
            gnn_mode=None, scales=None, candidate_count=None, latency_profile=None, resume=False):
    hpo_path, output = Path(hpo_path), Path(output)
    mode = gnn_mode or settings["refinement"]["mode"]
    if mode not in MODES:
        raise ValueError("Unknown internal GNN ablation mode")
    test_scales = list(scales if scales is not None else settings["data"]["test_scales"])
    if (not test_scales or len(set(test_scales)) != len(test_scales)
            or any(scale not in settings["data"]["test_scales"] for scale in test_scales)):
        raise ValueError("Comparison scales must be a distinct subset of configured test scales")
    runtime_settings = deepcopy(settings)
    runtime_settings["refinement"]["mode"] = mode
    if candidate_count is not None:
        if (type(candidate_count) is not int or not 1 <= candidate_count <= 16384
                or candidate_count & (candidate_count-1)):
            raise ValueError("Evaluation candidate count must be a power of two in [1,16384]")
        runtime_settings["control"]["candidate_count"] = candidate_count
    override_latency, override_latency_hash = None, None
    if latency_profile is not None:
        profile_bytes = Path(latency_profile).read_bytes()
        override_latency_hash = sha256(profile_bytes).hexdigest()
        override_latency = LatencyTree.from_dict(json.loads(profile_bytes.decode("utf-8")))
    hpo = json.loads((hpo_path/"hpo_report.json").read_text(encoding="utf-8"))
    selected = hpo["models"]["value"]
    if selected["selected_checkpoint"] is None:
        raise ValueError("Candidate-value GNN needs an eligible selected checkpoint before testing")
    checkpoint = (hpo_path/selected["selected_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(hpo_path.resolve()):
        raise ValueError("Checkpoint outside HPO directory")
    trial = next(row for row in selected["trials"] if row["trial"] == selected["selected_trial"])
    checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != trial["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum mismatch")
    # Check against the original training contract, before applying eval-only
    # mode/scale overrides. Each measured GNN run reloads and pays its load time.
    load_checkpoint(checkpoint, physical, settings, "value")
    source_hashes = {p.name: sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, torch=str(torch.__version__))
    ablation = dict(gnn_mode=mode, evaluated_scales=test_scales,
                    candidate_count=runtime_settings["control"].get("candidate_count"),
                    latency_profile_sha256=override_latency_hash,
                    refinement=runtime_settings["refinement"],
                    interpretation="GNN internal ablation; controller methods remain lp_mpc and gnn_value")
    protocol = dict(physical=physical, settings=settings, ablation=ablation,
                    checkpoint_sha256=checkpoint_hash, source_sha256=source_hashes,
                    versions=versions, platform=platform.platform(),
                    threading={key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    protocol_id = digest(protocol)
    if resume:
        existing = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
        if existing != dict(protocol_id=protocol_id, protocol=protocol):
            raise ValueError("Resume protocol differs: settings, source, checkpoint or runtime changed")
        index = json.loads((output/"completed.json").read_text(encoding="utf-8"))
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/"protocol.json", dict(protocol_id=protocol_id, protocol=protocol))
        index = dict(protocol_id=protocol_id, runs={})
        write_json(output/"completed.json", index)
    if index.get("protocol_id") != protocol_id:
        raise ValueError("Resume completion index has the wrong protocol")
    torch.set_num_threads(settings["control"]["torch_threads"])
    records = []
    for scale in test_scales:
        for week in settings["data"]["test_weeks"]:
            scenario = generate_scenario(physical, scale, case_seed(settings, "test", scale, week), weather_week=week)
            case_dir = output/f"{scale}_week{week}"
            fingerprint = scenario_fingerprint(scenario)
            if case_dir.exists():
                if not resume or scenario_fingerprint(load_scenario(case_dir/"scenario.npz")) != fingerprint:
                    raise ValueError("Saved scenario does not match the frozen comparison case")
            else:
                case_dir.mkdir()
                save_scenario(case_dir/"scenario.npz", scenario)
            order = list(np.random.default_rng(seed_for(settings["master_seed"], "method_order", scale, week)).permutation(METHODS))
            for repeat in range(settings["evaluation"]["repeats"]):
                rotated = order[repeat % len(order):]+order[:repeat % len(order)]
                for method in rotated:
                    stem = f"{method}_r{repeat}"
                    run_key = f"{case_dir.name}/{stem}"
                    report_path, trace_path = case_dir/f"{stem}.json", case_dir/f"{stem}.npz"
                    if run_key in index["runs"]:
                        saved = index["runs"][run_key]
                        if (sha256(report_path.read_bytes()).hexdigest() != saved["report_sha256"]
                                or sha256(trace_path.read_bytes()).hexdigest() != saved["trace_sha256"]):
                            raise ValueError("Completed comparison artifact checksum mismatch")
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        if report["scenario_fingerprint"] != fingerprint or report["protocol_id"] != protocol_id:
                            raise ValueError("Completed report has incompatible scenario or protocol")
                        records.append(report)
                        if report_progress:
                            report_progress(dict(phase="resume", method=method, scale=scale, week=week, repeat=repeat))
                        continue
                    # A controller interrupted before returning creates no
                    # result artifacts. Preserve unusual interrupted writes.
                    if report_path.exists() or trace_path.exists():
                        raise ValueError("Unindexed partial result exists; preserve it and choose a fresh output directory")
                    model, loading = None, 0.0
                    if method == "gnn_value":
                        clock = perf_counter()
                        model, _ = load_checkpoint(checkpoint, physical, settings, "value")
                        loading = perf_counter()-clock
                    report, traces = run_controller(scenario, runtime_settings, method, model,
                        model_load_s=loading, progress=report_progress,
                        error_predictor=getattr(model, "error_predictor", None),
                        latency_model=(override_latency if override_latency is not None
                                       else getattr(model, "latency_model", None)))
                    report.update(scale=scale, weather_week=week, repeat=repeat,
                                  scenario_fingerprint=fingerprint, protocol_id=protocol_id,
                                  gnn_mode=mode if method == "gnn_value" else None)
                    with trace_path.open("xb") as stream:
                        np.savez_compressed(stream, **traces)
                    write_json(report_path, report)
                    index["runs"][run_key] = dict(report_sha256=sha256(report_path.read_bytes()).hexdigest(),
                                                  trace_sha256=sha256(trace_path.read_bytes()).hexdigest())
                    _atomic_json(output/"completed.json", index)
                    records.append(report)
                    if report_progress:
                        report_progress(dict(phase="test", method=method, scale=scale, week=week,
                            accepted=report["accepted"], electricity_yuan=report["total_electricity_yuan"],
                            control_time_s=report["control_time_s"], fallback=report["fallback_count"],
                            solver_calls=report["solver_calls"]))
    fields = ["scale", "weather_week", "repeat", "method", "gnn_mode", "accepted", "status", "total_electricity_yuan",
              "control_time_s", "latency_p95_s", "latency_max_s", "fallback_count", "deadline_misses",
              "solver_calls", "anchor_action_count", "anchor_selected_count", "candidate_pool_mean",
              "candidate_generation_time_s", "inference_time_s", "candidate_verification_time_s"]
    temporary = output/"summary.csv.pending"
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, output/"summary.csv")
    result = dict(records=records, physical=physical, settings=settings, ablation=ablation,
                  protocol_id=protocol_id, checkpoint_sha256={"value": checkpoint_hash},
                  versions=versions, threading=protocol["threading"], platform=protocol["platform"],
                  source_sha256=source_hashes,
                  test_access="Only after candidate-value hyperparameter selection was frozen")
    _atomic_json(output/"manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-config", type=Path, default=Path("configs/experiment.toml"))
    parser.add_argument("--settings", type=Path, default=Path("configs/comparison.toml"))
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Generate/cache offline candidate continuation-cost labels")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("tune", help="Train and validate candidate-value GNN configurations")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="Reuse verified teacher trials before coarse HPO output is committed")
    p = sub.add_parser("compare", help="Evaluate frozen selections on the independent test week(s)")
    p.add_argument("--hpo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gnn-mode", choices=MODES, help="Internal message-fidelity ablation")
    p.add_argument("--scales", nargs="+", help="Subset of configured test scales")
    p.add_argument("--candidate-count", type=int, help="Evaluation-only raw Sobol count for density ablation")
    p.add_argument("--latency-profile", type=Path, help="Explicit recalibrated LatencyTree JSON; teacher weights stay fixed")
    p.add_argument("--resume", action="store_true", help="Resume verified completed cases under the exact saved protocol")
    p = sub.add_parser("run", help="Prepare labels, tune the candidate-value GNN, and compare with LP-MPC")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    physical, settings = load_config(args.physical_config), load_settings(args.settings)
    torch.set_num_threads(settings["control"]["torch_threads"])
    if args.command == "prepare":
        prepare_dataset(physical, settings, args.out, progress)
    elif args.command == "tune":
        result = tune_models(physical, settings, args.dataset, args.out, ("value",), progress,
                             resume=args.resume)
        if any(result["models"][kind]["selected_checkpoint"] is None for kind in ("value",)):
            raise RuntimeError("HPO completed without an eligible selection; inspect retained reports")
    elif args.command == "compare":
        result = compare(physical, settings, args.hpo, args.out, gnn_mode=args.gnn_mode,
                         scales=args.scales, candidate_count=args.candidate_count,
                         latency_profile=args.latency_profile, resume=args.resume)
        if not all(r["accepted"] and r["deadline_misses"] == 0
                   for r in result["records"]):
            return 2
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        prepare_dataset(physical, settings, args.out/"dataset", progress)
        hpo = tune_models(physical, settings, args.out/"dataset", args.out/"hpo",
                          kinds=("value",), progress=progress)
        if any(hpo["models"][kind]["selected_checkpoint"] is None for kind in ("value",)):
            raise RuntimeError("No eligible candidate-value model; reports retained, test set unopened")
        result = compare(physical, settings, args.out/"hpo", args.out/"evaluation")
        if not all(r["accepted"] and r["deadline_misses"] == 0
                   for r in result["records"]):
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

