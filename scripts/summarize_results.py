"""Render the completed five-scale study without loading models or rerunning it.

Example (from the repository root):
  python scripts/summarize_results.py --root artifacts/full_20261009 \
      --out docs/results_20261009.md --tests-passed 256

Requires the paired weekly comparison, training reports, validation-state
fidelity ablation and LP-label error decomposition. Missing or inconsistent
evidence is an error; no partial report is written. Only Python's standard
library is used. All derived comparisons come from the saved measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
import json
import math
import os
from pathlib import Path
from statistics import mean, median


SCALES = ("S1", "S2", "S3", "S4", "S5")
METHODS = ("lp_mpc", "gnn_value")
MODES = ("coarse", "full", "random", "adaptive")
MODE_NAMES = {"coarse": "全粗", "full": "完整 GNN", "random": "随机精化", "adaptive": "自适应精化"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file(), f"Required completed artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    require(path.is_file(), f"Missing evidence file: {path}")
    result = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(value, name, *, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"Nonfinite/missing numeric field: {name}")
    require(not positive or value > 0, f"Nonpositive field: {name}")
    return float(value)


def f(value, digits=2):
    return f"{value:,.{digits}f}"


def pct(value, digits=1):
    return f"{100 * value:.{digits}f}%"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "| " + " | ".join("---" for _ in header) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def load_evidence(root):
    project = root.parent.parent
    evaluation = read_json(root / "evaluation/manifest.json")
    protocol = read_json(root / "evaluation/protocol.json")
    completion = read_json(root / "evaluation/completed.json")
    hpo = read_json(root / "hpo/hpo_report.json")
    coarse = read_json(root / "hpo/coarse_hpo.json")
    error = read_json(root / "hpo/error_report.json")
    latency = read_json(root / "hpo/latency_report.json")
    dataset = read_json(root / "dataset/manifest.json")
    ablation = read_json(root / "ablation/results.json")
    decomposition = read_json(root / "value_error_decomposition.json")
    settings, physical = evaluation["settings"], evaluation["physical"]
    require(physical["simulation"] == {"horizon": 336, "dt_hours": .5}, "Expected one complete 336-step week")
    require(settings["master_seed"] == 42, "This report protocol requires master seed 42")
    require(settings["data"]["test_scales"] == list(SCALES), "Expected all five test scales")
    require(len(settings["data"]["test_weeks"]) == 1 and settings["evaluation"]["repeats"] == 1,
            "This renderer requires one held-out test week and one paired repeat")
    require(settings["data"]["train_scales"] == ["S1", "S2"], "Unexpected supervised training scales")
    require(settings["control"]["torch_threads"] == latency["torch_threads"] == 1
            and settings["control"]["solver_threads"] == 1, "Thread protocol mismatch")
    require(read_json(root / "hpo/settings.json") == settings, "Training and evaluation settings differ")
    pid = evaluation["protocol_id"]
    require(protocol["protocol_id"] == completion["protocol_id"] == pid, "Protocol identity mismatch")
    frozen = protocol["protocol"]
    require(frozen["physical"] == physical and frozen["settings"] == settings, "Frozen protocol mismatch")
    require(evaluation["source_sha256"] == frozen["source_sha256"], "Source manifest mismatch")
    for name, digest in evaluation["source_sha256"].items():
        require(file_hash(project / "src/dr_research" / name) == digest, f"Source changed after comparison: {name}")
    selected = hpo["models"]["value"]
    checkpoint = root / "hpo" / selected["selected_checkpoint"]
    digest = file_hash(checkpoint)
    require(digest == evaluation["checkpoint_sha256"]["value"] == frozen["checkpoint_sha256"]
            == ablation["checkpoint_sha256"] == decomposition["checkpoint_sha256"], "Model weights differ across evidence")
    require(dataset["key"] == hpo["dataset_key"] == decomposition["dataset_key"], "Dataset identity mismatch")
    require(file_hash(root / "dataset/samples.pt") == dataset["samples_sha256"], "Dataset sample checksum mismatch")
    require(file_hash(root / "dataset/exploration.json") == dataset["exploration_sha256"], "Exploration checksum mismatch")
    for name, artifact in (("ablate_snapshots.py", ablation), ("diagnose_value_errors.py", decomposition)):
        require(file_hash(project / "scripts" / name) == artifact["script_sha256"], f"Analysis script changed: {name}")
    require(len(selected["trials"]) == len(hpo["search_space"]) == 4 and len(coarse["trials"]) == 4,
            "Expected all four complete-model and four coarse-model HPO trials")
    for key in ("train_weeks", "validation_weeks"):
        require(not set(settings["data"][key]) & set(settings["data"]["test_weeks"]), "Test-week leakage")

    week = settings["data"]["test_weeks"][0]
    expected = set(product(SCALES, METHODS))
    records, indexed = evaluation["records"], {}
    require(len(records) == 10, "Need exactly ten completed paired controller runs")
    require(len(completion["runs"]) == 10, "Completion index is incomplete")
    for record in records:
        key = record["scale"], record["method"]
        require(key in expected and key not in indexed, "Missing, duplicate, or unexpected controller run")
        require(record["weather_week"] == week and record["repeat"] == 0, "Run split differs")
        require(record["executed_steps"] == record["decision_attempts"] == 336, f"Incomplete weekly rollout: {key}")
        require(record["protocol_id"] == pid, "Run protocol differs")
        require(record["window_steps"] == settings["control"]["window_steps"], "Controller forecast window differs")
        finite(record["control_time_s"], "control_time_s", positive=True)
        if record["accepted"]:
            finite(record["total_electricity_yuan"], "accepted weekly bill", positive=True)
        stem = f"{key[0]}_week{week}/{key[1]}_r0"
        saved = completion["runs"].get(stem)
        require(saved is not None, f"Run absent from completion index: {stem}")
        report_path = root / "evaluation" / (stem + ".json")
        require(file_hash(report_path) == saved["report_sha256"], f"Run report checksum mismatch: {stem}")
        require(file_hash(root / "evaluation" / (stem + ".npz")) == saved["trace_sha256"], f"Trace checksum mismatch: {stem}")
        require(read_json(report_path) == record, f"Run and final manifest disagree: {stem}")
        if key[1] == "lp_mpc":
            require(len(record["solver_steps"]) == record["solver_calls"], "Incomplete per-step solver evidence")
        else:
            require(record["solver_calls"] == 0, "Learned online controller unexpectedly called LP")
            require(record["candidate_count_requested"] == settings["control"]["candidate_count"], "Candidate count differs")
        indexed[key] = record
    require(set(indexed) == expected, "Five-scale paired results incomplete")
    for scale in SCALES:
        require(indexed[scale, "lp_mpc"]["scenario_fingerprint"] == indexed[scale, "gnn_value"]["scenario_fingerprint"],
                f"Methods did not share the same physical scenario: {scale}")

    require(ablation["scales"] == ["S1", "S5"] and ablation["counts"] == [64, 256, 1024]
            and ablation["steps"] == [0, 60, 156, 252] and ablation["repeats"] == 3,
            "Incomplete or changed ablation design")
    expected_ablation = set(product(ablation["scales"], ablation["counts"], ablation["steps"], range(3), MODES))
    seen, grouped, minima = set(), defaultdict(list), defaultdict(list)
    for row in ablation["records"]:
        key = row["scale"], row["raw_count"], row["t"], row["repeat"], row["mode"]
        require(key in expected_ablation and key not in seen, "Incomplete/duplicate ablation cell")
        require(row["validation_week"] in settings["data"]["validation_weeks"], "Ablation used a nonvalidation week")
        finite(row["scoring_time_s"], "ablation scoring time", positive=True)
        finite(row["teacher_relative_regret_yuan_per_tank"], "ablation regret")
        require(row["teacher_relative_regret_yuan"] >= -1e-4, "Negative teacher-relative regret")
        seen.add(key)
        grouped[row["scale"], row["raw_count"], row["mode"]].append(row)
        minima[row["scale"], row["t"], row["raw_count"]].append(row["teacher_minimum_yuan"])
    require(seen == expected_ablation, "Missing ablation states, densities, modes or repeats")
    for scale, t in product(ablation["scales"], ablation["steps"]):
        previous = math.inf
        for count in ablation["counts"]:
            values = minima[scale, t, count]
            require(max(values) - min(values) <= 1e-6, "Teacher minimum differs within the same frozen pool")
            require(values[0] <= previous + 1e-3, "Nested-pool teacher minimum increased")
            previous = values[0]
    rows = decomposition["records"]
    require(len(rows) == dataset["counts"]["validation"]["groups"], "Validation error decomposition incomplete")
    require(len({(r["case"], r["t"]) for r in rows}) == len(rows), "Repeated decomposition state")
    for key in ("coverage_gap_yuan_per_tank", "ranking_regret_yuan_per_tank"):
        actual = mean(finite(r[key], key) for r in rows)
        require(math.isclose(actual, decomposition["summary"][key], rel_tol=1e-9, abs_tol=1e-9), "Error summary inconsistent")
    return dict(root=root, project=project, evaluation=evaluation, settings=settings, physical=physical,
                indexed=indexed, hpo=hpo, coarse=coarse, error=error, latency=latency, dataset=dataset,
                ablation=ablation, grouped=grouped, decomposition=decomposition, checkpoint=checkpoint)


def render(e, out, tests_passed):
    root, cfg, p, data = e["root"], e["settings"], e["physical"], e["dataset"]
    def link(path, label):
        return f"[{label}](<{Path(os.path.relpath(path, out.parent)).as_posix()}>)"
    def evidence(relative, label):
        return link(root / relative, label)
    r = e["indexed"]
    paired = [scale for scale in SCALES if all(r[scale, method]["accepted"] for method in METHODS)]
    parts = ["# 一周闭环与候选保真度实验结果（2026-10-09）"]
    if paired:
        gaps = [r[s, "gnn_value"]["total_electricity_yuan"] / r[s, "lp_mpc"]["total_electricity_yuan"] - 1 for s in paired]
        faster = [s for s in paired if r[s, "lp_mpc"]["control_time_s"] > r[s, "gnn_value"]["control_time_s"]]
        parts.append(f"已完成五组规模、两种控制器各 336 步的配对实验。物理回放通过 {sum(x['accepted'] for x in r.values())}/10 组。"
                     f"GNN 周电费相对 LP-MPC 的差值为 {100 * min(gaps):+.2f}% 至 {100 * max(gaps):+.2f}%；"
                     f"GNN 在 {'、'.join(faster) if faster else '零个规模'} 的完整控制时间更短。"
                     "主对比体现了在线 LP 求解被候选评价与代数续接替代后的代价和收益；双保真调度自身的收益由后面的同状态消融判断。")
    else:
        parts.append("十组运行均已结束；物理验收未形成可用配对结果，电费优劣结论暂不成立。")
    parts.append(f"主随机种子 {cfg['master_seed']}；南京典型气象周 {cfg['data']['test_weeks'][0]}；半小时一步，连续一周；"
                 f"预测窗口 {cfg['control']['window_steps']} 步，续接窗口 {cfg['control']['safety_window_steps']} 步；"
                 f"每步 {cfg['control']['candidate_count']} 个原始 Sobol 点加一个已认证备用动作；"
                 f"主比较使用 {MODE_NAMES[e['evaluation']['ablation']['gnn_mode']]}。Torch 与 HiGHS 均为单线程。"
                 "周电费来自已执行动作的物理回放；时间包含初始化、模型加载、候选生成、评价、续接及最终动作检查，环境状态推进与事后整周审计另计。"
                 + evidence("evaluation/manifest.json", "配对结果与运行协议") + "。")

    parts.append("## 实际电费与完整控制时间")
    rows = []
    for scale in SCALES:
        lp, gnn = r[scale, "lp_mpc"], r[scale, "gnn_value"]
        valid = lp["accepted"] and gnn["accepted"]
        size = "×".join(map(str, p["scales"][scale]))
        rows.append([f"{scale} · {size}", f(lp["total_electricity_yuan"]) if lp["accepted"] else "验收失败",
                     f(gnn["total_electricity_yuan"]) if gnn["accepted"] else "验收失败",
                     f"{100 * (gnn['total_electricity_yuan'] / lp['total_electricity_yuan'] - 1):+.2f}%" if valid else "—",
                     f(lp["control_time_s"]), f(gnn["control_time_s"]),
                     f"{lp['control_time_s'] / gnn['control_time_s']:.2f}×",
                     f"{lp['fallback_count']}/{gnn['fallback_count']}",
                     f"{lp['deadline_misses']}/{gnn['deadline_misses']}",
                     f"{'通过' if lp['accepted'] else '失败'}/{'通过' if gnn['accepted'] else '失败'}"])
    parts.append(table(["规模 H×S×B", "LP 电费/元", "GNN 电费/元", "电费差", "LP 时间/s", "GNN 时间/s", "LP/GNN 时间", "回退 LP/GNN", "超时 LP/GNN", "物理验收 LP/GNN"], rows))
    parts.append("时间比大于 1 表示 GNN 更快。回退次数表示本轮评分后的有限续接搜索或运行异常触发备用方案；按评分正常选中备用动作单独统计。"
                 "续接 UNKNOWN 表示构造器未找到原动作的证书，不代表该动作已被证明不可行。每轮完整控制限时 "
                 f"{cfg['control']['time_limit_s']:g} 秒。")
    lp_status = Counter(step["status"] for scale in SCALES for step in r[scale, "lp_mpc"]["solver_steps"])
    residuals = {name: max(finite(row["maximum_residuals"][name], name) for row in r.values())
                 for name in ("building_heat_balance", "trajectory_dynamics", "terminal_energy")}
    parts.append(f"LP-MPC 共调用 {sum(r[s, 'lp_mpc']['solver_calls'] for s in SCALES)} 次求解，状态计数 {dict(lp_status)}；"
                 "GNN 在线 LP 调用为 0。LP 使用保持目标与约束等价的热泵聚合，并在原始设备网络展开验收。"
                 f"各运行最大建筑供热残差 {residuals['building_heat_balance']:.3g} kW，"
                 f"库存递推残差 {residuals['trajectory_dynamics']:.3g} kWh，周末库存残差 {residuals['terminal_energy']:.3g} kWh。")

    parts.append("## GNN 时间花在哪里")
    rows = []
    stages = (("candidate_generation_time_s", "候选生成"), ("inference_time_s", "GNN 评价"),
              ("candidate_verification_time_s", "动作续接"))
    for scale in SCALES:
        row = r[scale, "gnn_value"]
        rows.append([scale, *(f"{f(row[key])} ({pct(row[key] / row['control_time_s'])})" for key, _ in stages),
                     f(row["latency_p95_s"], 4), f(row["latency_max_s"], 4),
                     pct(row["refinement_fraction"], 2), row["continuation_unknown_count"]])
    parts.append(table(["规模", "生成/s（占比）", "评价/s（占比）", "续接/s（占比）", "单步 P95/s", "单步最大/s", "消息精化比例", "续接 UNKNOWN 次数"], rows))
    s5 = r["S5", "gnn_value"]
    substage = [("上下文与批量特征", "score_prepare_s"), ("粗分支", "score_coarse_scoring_s"),
                ("残差预测", "score_error_prediction_s"), ("灵敏度", "score_sensitivity_s"),
                ("分组调度", "score_scheduling_s"), ("精细消息更新", "score_refinement_s")]
    parts.append("S5 的评价内部累计耗时：" + "；".join(f"{name} {f(s5[key])} s" for name, key in substage) + "。"
                 f"评分预算耗尽 {s5['scoring_budget_exhausted_count']} 次，评分预算超出累计 {f(s5['scoring_overrun_s'], 4)} s，"
                 f"提示重标定 {s5['recalibration_required_count']} 次。三项主阶段未包含的时间是初始化、最终动作检查和控制循环开销。")

    parts.append("## 离线训练与验证")
    parts.append(f"完整教师 H 与粗分支 L 各训练 4 组超参数。监督样本来自 S1、S2：训练周 {cfg['data']['train_weeks']}，"
                 f"验证周 {cfg['data']['validation_weeks']}；训练 {data['counts']['train']['groups']} 个状态组、"
                 f"{data['counts']['train']['labels']} 个有效标签，验证 {data['counts']['validation']['groups']} 个状态组、"
                 f"{data['counts']['validation']['labels']} 个标签。{data['candidate_labels_attempted']} 次 LP 标签尝试中，"
                 f"{data['candidate_labels_available']} 次有效，{data['candidate_labels_infeasible']} 次已证不可行，"
                 f"{data['candidate_labels_unknown']} 次未知。标签固定候选的当前动作，在相同 {data['teacher_window_steps']} 步窗口和公共终端条件下求最低电费。"
                 "S3–S5 没有参与 H/L 的有标签训练，属于规模外推；计算时间标定覆盖了这些规模。"
                 + evidence("dataset/manifest.json", "数据清单") + "。")
    rows = []
    best_h = e["hpo"]["models"]["value"]["selected_trial"]
    for trial in e["hpo"]["models"]["value"]["trials"]:
        hp, fit = trial["hyperparameters"], trial["fit"]
        rows.append([f"H{trial['trial']}" + ("（选中）" if trial["trial"] == best_h else ""),
                     f"hidden={hp['hidden_dim']}, latent={hp['latent_dim']}, lr={hp['learning_rate']:g}",
                     fit["best_epoch"], f(fit["training_s"]),
                     f(fit["validation"]["mean_regret_yuan_per_tank"], 5),
                     pct(fit["validation"]["optimal_fraction"], 2)])
    parts.append(table(["完整教师", "参数", "最佳轮次", "训练/s", "LP 标签后悔值/元·罐⁻¹", "选中标签最优比例"], rows))
    trials = e["coarse"]["trials"]
    best_l = min(range(len(trials)), key=lambda i: trials[i]["validation"]["mean_teacher_regret_yuan_per_tank"])
    rows = [[f"L{i}" + ("（选中）" if i == best_l else ""), trial["width"], trial["temperature"], trial["best_epoch"],
             f(trial["training_s"]), f(trial["validation"]["mean_teacher_regret_yuan_per_tank"], 5),
             pct(trial["validation"]["teacher_optimal_fraction"], 2)] for i, trial in enumerate(trials)]
    parts.append(table(["粗分支", "width", "温度参数", "最佳轮次", "训练/s", "相对 H 后悔值/元·罐⁻¹", "选中 H 最优比例"], rows))
    selected_fit = next(t["fit"] for t in e["hpo"]["models"]["value"]["trials"] if t["trial"] == best_h)
    parts.append(f"H 的候选决策误差与数值预测误差分别评价：选中 H 的未来电费 RMSE 为 "
                 f"{f(selected_fit['validation']['future_rmse_yuan_per_tank'], 3)} 元/罐。"
                 "L 的式（24）目标是相对冻结 H 的决策损失，其较小值说明复现教师排序，并不直接证明较低的真实电费。"
                 f"残差预测器在独立验证状态的潜在消息范数 MAE 为 {f(e['error']['validation_mae_latent'], 5)}；"
                 f"时间预测树选取树参数时的留出工作量验证 MAE 为 {1000 * e['latency']['validation_mae_s']:.3f} ms，"
                 f"中位绝对百分比误差为 {pct(e['latency']['validation_median_absolute_percentage_error'], 2)}。"
                 "这组留出数据用于选择树参数，最终时间树用全部标定样本重新拟合；该数值属于模型选择误差，额外独立测试误差尚未测量。"
                 + evidence("hpo/hpo_report.json", "H 训练报告") + "、" + evidence("hpo/coarse_hpo.json", "L 训练报告") + "、"
                 + evidence("hpo/latency_report.json", "单线程时间标定") + "。")
    parts.append("离线训练、标签生成与时间标定不计入在线控制时间。原始标签生成采用了 HiGHS 默认线程选择；其 "
                 f"{f(data['generation_s'])} s 是该次实际离线运行时间，不作为单线程性能对比。"
                 f"四组 H 累计训练 {f(sum(t['fit']['training_s'] for t in e['hpo']['models']['value']['trials']))} s，"
                 f"四组 L 累计训练 {f(sum(t['training_s'] for t in trials))} s；断点恢复复用了已完成 H 权重。")

    parts.append("## 固定状态上的密度与消息保真度消融")
    parts.append("S1、S5 各使用验证周已认证定库存参考轨迹的 4 个状态；每个密度和模式测量 3 次，模式顺序打乱。"
                 "所有模式共享冻结模型与嵌套候选池。表内后悔值为所选候选在完整 H 下的评分减去同池 H 最小评分，再除以储热罐数；它是教师一致性指标。"
                 "时间取 12 次评分的中位数，后悔值取均值；重复测量同时反映时间波动与随机精化、覆盖候选选择的随机性。该实验没有执行新的整周闭环。"
                 + evidence("ablation/results.json", "消融记录") + "。")
    rows = []
    for scale, count in product(("S1", "S5"), (64, 256, 1024)):
        row = [scale, count]
        for mode in MODES:
            values = e["grouped"][scale, count, mode]
            row.append(f"{1000 * median(v['scoring_time_s'] for v in values):.2f} / "
                       f"{mean(v['teacher_relative_regret_yuan_per_tank'] for v in values):.5f}")
        rows.append(row)
    parts.append(table(["规模", "原始 N", *(f"{MODE_NAMES[m]}：ms / 后悔值" for m in MODES)], rows))
    dense = e["grouped"]["S5", 1024, "adaptive"]
    base_overruns = sum(v["prepare_s"] + v["coarse_scoring_s"] > v["budget_s"] for v in dense)
    score_overruns = sum(v["overrun_s"] > 0 for v in dense)
    no_refinements = sum(v["refined_messages"] == 0 for v in dense)
    dense_stages = (("上下文与批量特征", "prepare_s"), ("粗分支", "coarse_scoring_s"),
                    ("残差预测", "error_prediction_s"), ("灵敏度", "sensitivity_s"),
                    ("分组调度", "scheduling_s"), ("精细消息更新", "refinement_s"))
    parts.append(f"S5、N=1024 的 {len(dense)} 次自适应评分中，基础特征与粗评分已经超过预算 {base_overruns} 次，"
                 f"完整评分超过预算 {score_overruns} 次，未执行消息精化 {no_refinements} 次；"
                 f"消息精化比例中位数 {pct(median(v['refinement_fraction'] for v in dense), 2)}。"
                 "各阶段时间中位数：" + "；".join(f"{name} {1000 * median(v[key] for v in dense):.2f} ms" for name, key in dense_stages) + "。"
                 "各阶段中位数分别计算，其和不等于总时间中位数。预算是运行时调度目标，基础计算本身也可能超过预算；"
                 "完整 H 模式直接计算全部精细消息，表中时间包含当次准备与首次精细投影，未复用另一个模式的缓存。")
    faster = better_than_random = 0
    for scale, count in product(("S1", "S5"), (64, 256, 1024)):
        group = {m: e["grouped"][scale, count, m] for m in MODES}
        faster += median(x["scoring_time_s"] for x in group["adaptive"]) < median(x["scoring_time_s"] for x in group["full"])
        better_than_random += mean(x["teacher_relative_regret_yuan_per_tank"] for x in group["adaptive"]) < mean(x["teacher_relative_regret_yuan_per_tank"] for x in group["random"]) - 1e-9
    parts.append(f"自适应精化在 6 个规模×密度组合中的 {faster} 个比直接完整 H 评分更快，在 {better_than_random} 个的平均教师后悔值低于随机精化。"
                 "判断双保真策略的价值时，应同时查看这两项及调度额外开销。主对比中的 GNN/LP 时间比包含了整个求解器替换的收益，不能全部归于消息精化。")
    density_rows = []
    for scale, count in product(("S1", "S5"), (64, 256, 1024)):
        rows_for_pool = [next(v for v in e["grouped"][scale, count, "full"] if v["t"] == t)
                         for t in e["ablation"]["steps"]]
        density_rows.append([scale, count, f(mean(x["candidates"] for x in rows_for_pool), 1),
                             f(1000 * median(x["candidate_generation_s"] for x in rows_for_pool), 2),
                             f(mean(x["teacher_minimum_yuan"] for x in rows_for_pool), 3)])
    parts.append(table(["规模", "原始 N", "平均保留数（含备用）", "4 个状态的生成中位数/ms", "同状态 H 最优评分均值/元"], density_rows))
    parts.append("采样脚本逐一验证已有候选 ID 对应的动作完全相同，并验证每个固定状态下 N=64→256→1024 的 H 最优评分不增（容差 0.001 元）。"
                 "生成耗时每个池只测一次，表内没有把跨模式复制的数值当成额外重复。嵌套性保证候选集合扩张；实际周电费还受教师误差、续接接受率和闭环状态变化影响。")

    parts.append("## 误差归因与后续方向")
    dec = e["decomposition"]
    parts.append(f"在 {len(dec['records'])} 个已保存验证状态、每状态最多 {max(row['candidates'] for row in dec['records'])} 个已标注候选上，"
                 f"候选集最佳 LP 电费与连续动作 LP 最优值的平均差为 {f(dec['summary']['coverage_gap_yuan_per_tank'], 5)} 元/罐；"
                 f"完整 H 排序造成的额外平均损失为 {f(dec['summary']['ranking_regret_yuan_per_tank'], 5)} 元/罐。"
                 "前者衡量这个已标注子集的动作覆盖，后者衡量同子集内的教师排序。标注动作按候选池索引等距抽取，仅保留 LP 求得最优解的标签；未标注、已证不可行和 UNKNOWN 候选均未计入该子集。"
                 f"该分解没有求在线最多 {cfg['control']['candidate_count'] + 1} 个候选的 LP 真值，也没有覆盖续接筛选和整周状态反馈。"
                 + evidence("value_error_decomposition.json", "误差分解证据") + "。")
    parts.append("下一轮优先按上述三条证据推进：改善候选对连续动作最优解的覆盖，校准完整 H 的未来电费排序，降低精化调度开销并测量它相对直接完整 H 的净收益。"
                 "固定总功率后仍可调整分配的物理等价性、续接 UNKNOWN 的集中时段，以及高电价前的储热行为，是可直接定位的检查对象。")
    parts.append("本轮范围：固定 COP、固定 H→S 接收端、设备全时可用且保留最小负荷与爬坡约束；住宅周末占用率影响需求。"
                 "建筑随机系数在整周固定，S1 归一化后为 1；没有逐时需求噪声，也没有天气或需求预测误差。"
                 "主实验是随机种子 42、单一测试周、每方法一次运行；这些结果描述当前模型条件下的配对表现，跨种子、跨季节和真实设备泛化仍需独立实验。")
    parts.append("## 可复核证据")
    parts.append(f"自动化测试由调用者提供的已完成记录为 {tests_passed} passed；本脚本不重新运行测试。"
                 "生成报告前已核验十组完成索引、报告与轨迹 SHA-256、冻结 Python 源码、训练样本、跨分析模型权重一致性，以及全部 288 个消融测量单元。"
                 "物理验收状态按原结果列示。")
    parts.append("协议：" + evidence("evaluation/protocol.json", "配置、源码摘要与运行环境") + "；"
                 + evidence("evaluation/completed.json", "十组报告及轨迹校验和") + "；"
                 + link(e["checkpoint"], "选定模型权重") + "；"
                 + link(e["project"] / "scripts/summarize_results.py", "本报告生成脚本") + "。\n\n"
                 f"协议 SHA-256：`{e['evaluation']['protocol_id']}`。模型 SHA-256：`{e['evaluation']['checkpoint_sha256']['value']}`。")
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tests-passed", type=int, required=True)
    args = parser.parse_args()
    try:
        require(args.tests_passed > 0, "tests-passed must identify a completed positive test count")
        evidence = load_evidence(args.root.resolve())
        out = args.out.resolve()
        content = render(evidence, out, args.tests_passed)
        out.parent.mkdir(parents=True, exist_ok=True)
        pending = out.with_name(out.name + ".pending")
        pending.write_text(content, encoding="utf-8")
        pending.replace(out)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"Report not written: {exc}\n")
    print(json.dumps({"report": str(out), "weekly_runs": 10,
                      "ablation_measurements": len(evidence["ablation"]["records"]),
                      "report_sha256": file_hash(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
