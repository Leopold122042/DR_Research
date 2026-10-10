# 高温热泵 LP MPC 与候选评价 GNN

本工程比较 `lp_mpc` 与 `gnn_value` 两种控制方法，主指标是一周实际总电费和端到端控制计算时间。候选评价GNN包含轻量候选、完整/粗消息、教师相对决策训练、混合消息缓存、误差与灵敏度估计、运行时间模型和预算调度。模型效果通过独立闭环实验评价，工程测试用于检验接口、守恒、精化等价性和验收合同。

| 方法 | 每轮控制决策 | 离线工作 |
|---|---|---|
| LP MPC | 求未来24小时最低电费LP，执行当前半小时动作 | 无神经模型训练 |
| 候选评价GNN | 生成当前合法动作，粗评分并按预算精化消息，按排名续接验收，执行首个通过者 | 条件LP标签、完整教师HPO、粗分支HPO、残差模型及硬件延迟校准 |

GNN内部的 `coarse`、`full`、`random`、`adaptive` 是消息计算消融模式，控制方法名称始终为 `gnn_value`。研究结构、实验适配和能力边界见 [方法与实现范围](docs/method_status.md)。

当前交接入口：[本地运行、费用量级与导出可比性](docs/local_run_and_comparability_20261009.md)。已完成全部5组LP-MPC和前4组GNN，S5 GNN按用户要求中断，等待用户自行恢复；双保真消融尚未运行。原始数据、冻结模型和完整逐轮轨迹保存在 `artifacts/full_20261009/`。

## 共同实验条件

- 主种子42；连续168小时，每30分钟重新决策，共336步。
- 五档热泵、储罐、建筑规模：8×2×1、32×8×4、128×32×16、512×128×64、2048×512×256。
- 每台热泵固定接一个储罐，多泵可接同一罐；储罐与建筑允许多对多连接。
- 固定COP、外生开启状态、功率与爬坡、逐栋需求、库存、热损耗、热源及总功率约束。
- 南京CSWD天气与住宅工作日/周末占用共同决定已知需求。建筑差异在场景生成时一次抽取后整周固定；逐时需求噪声与预测误差未启用。
- 相同场景、费用预测窗口和终端约定。实际电费按最终执行功率重新计算；GNN在线评分与续接不调用LP。

## 配置和基本流程

物理配置为 [experiment.toml](configs/experiment.toml)，完整设置为 [comparison.toml](configs/comparison.toml)，小规模流程检查为 [comparison_smoke.toml](configs/comparison_smoke.toml)。默认完整设置使用256个原始Sobol点，备用动作另计；合法性检查和去重决定实际池大小。参数及来源见 [参数说明](docs/parameters.md)，当前候选与续接约定见 [安全续接说明](docs/safety_decoder.md)。

在仓库根目录使用已安装依赖的虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m dr_research --help
.\.venv\Scripts\python.exe -m dr_research --settings configs/comparison_smoke.toml prepare --out artifacts/smoke_labels
.\.venv\Scripts\python.exe -m dr_research --settings configs/comparison_smoke.toml tune --dataset artifacts/smoke_labels --out artifacts/smoke_hpo
.\.venv\Scripts\python.exe -m dr_research --settings configs/comparison_smoke.toml compare --hpo artifacts/smoke_hpo --out artifacts/smoke_evaluation
```

完整实验显式采用 `--settings configs/comparison.toml`，或使用这一默认设置。`run --out artifacts/new_run` 依次执行标签生成、训练调参和独立测试。小规模配置仅用于流程检验，其训练预算和数据量有意缩小。

训练依次完成四项工作：完整教师按独立验证状态的LP相对决策损失选型；粗分支按冻结教师的硬决策损失选型；使用训练后完整/粗消息对拟合残差；使用验证周场景测量各规模的实际增量延迟。粗分支训练的费用尺度从训练池按储罐数归一化后确定，随后固定于各状态和候选密度。完整教师选型使用验证状态内的LP相对硬排序损失，粗分支选型使用教师相对硬排序损失；这两项选型没有使用验证周完整闭环电费。测试周保持独立。

新输出目录默认禁止覆盖。教师试验已经保存且粗分支结果尚未提交时，可用 `tune --resume` 复用通过合同和哈希检查的教师检查点；粗分支之后的训练阶段不支持从中间位置恢复。

## 正式实验与已保存结果复核

下列命令对应本次正式产物布局，按顺序串行执行；计时期间保持其他实验和训练停止。`prepare`、`tune`、`compare` 及两个分析脚本的实验输出禁止覆盖。**已有 `artifacts/full_20261009` 结果时，直接使用后面的恢复/汇总命令；从头重跑应将下面所有路径中的 `full_20261009` 统一换成新的目录名。** 新运行的耗时重新测量。

```powershell
.\.venv\Scripts\python.exe -m dr_research --physical-config configs/experiment.toml --settings configs/comparison.toml prepare --out artifacts/full_20261009/dataset
.\.venv\Scripts\python.exe -m dr_research --physical-config configs/experiment.toml --settings configs/comparison.toml tune --dataset artifacts/full_20261009/dataset --out artifacts/full_20261009/hpo
.\.venv\Scripts\python.exe -m dr_research --physical-config configs/experiment.toml --settings configs/comparison.toml compare --hpo artifacts/full_20261009/hpo --out artifacts/full_20261009/evaluation
.\.venv\Scripts\python.exe scripts/ablate_snapshots.py --physical-config configs/experiment.toml --settings configs/comparison.toml --checkpoint artifacts/full_20261009/hpo/selected.pt --out artifacts/full_20261009/ablation --scales S1 S5 --counts 64 256 1024 --steps 0 60 156 252 --repeats 3
.\.venv\Scripts\python.exe scripts/diagnose_value_errors.py --physical-config configs/experiment.toml --settings configs/comparison.toml --dataset artifacts/full_20261009/dataset --checkpoint artifacts/full_20261009/hpo/selected.pt --out artifacts/full_20261009/value_error_decomposition.json
```

消融使用冻结权重和验证状态，报告教师相对后悔值与评分时间；误差分解使用已标注验证候选，两者与正式测试周电费分别解释。复用已保存周实验、重新生成结果文档：

```powershell
.\.venv\Scripts\python.exe -m dr_research --physical-config configs/experiment.toml --settings configs/comparison.toml compare --hpo artifacts/full_20261009/hpo --out artifacts/full_20261009/evaluation --resume
.\.venv\Scripts\python.exe scripts/summarize_results.py --root artifacts/full_20261009 --out docs/results_20261009.md --tests-passed 256
```

`compare --resume` 只复用与保存合同一致、且报告和轨迹校验和正确的完整运行；源码、配置、权重、依赖版本及线程环境变化会使恢复失败。合同和校验和见 `evaluation/protocol.json`、`evaluation/completed.json`。汇总脚本读取保存产物，核验五组规模×两方法的完整336步、全部288条消融记录及跨分析权重一致性，缺失时拒绝生成。`--tests-passed 256` 是本轮已完成测试数量的记录，不会运行测试；新实验应填入实际验收数量。

## 内部消融与恢复

```powershell
.\.venv\Scripts\python.exe -m dr_research compare --hpo artifacts/hpo --out artifacts/full --gnn-mode full --scales S1 S2
.\.venv\Scripts\python.exe -m dr_research compare --hpo artifacts/hpo --out artifacts/coarse --gnn-mode coarse --scales S1 S2
.\.venv\Scripts\python.exe -m dr_research compare --hpo artifacts/hpo --out artifacts/random --gnn-mode random --scales S1 S2
.\.venv\Scripts\python.exe -m dr_research compare --hpo artifacts/hpo --out artifacts/adaptive --gnn-mode adaptive --scales S1 S2
```

每个比较目录仍包含LP MPC与指定GNN模式。`--candidate-count 1024` 可以在教师权重固定时改变评估取样密度。更大的池、不同硬件、线程或模型宽度需要重新测量延迟；`--latency-profile` 指定重新校准并保存的 `LatencyTree.to_dict()` JSON。未覆盖的工作负载会记录重新校准需求，严格模式停止该部分精化。

`compare --resume` 要求设置、代码、权重、运行环境和场景完全匹配，并校验已保存报告/轨迹的哈希。它跳过已完成的控制运行，未完成的单条周轨迹重新开始。无法核验的部分写入会保留并报错。

每次比较生成固定协议、场景、两方法的报告与轨迹、汇总CSV和最终清单。报告包括候选规模、评分各环节时间、精化比例、续接UNKNOWN、备用使用、预算超限和最终物理验收。工程验收与经济效果分别报告。命令成功要求完成物理验收且没有整轮超时；已验证备用方案的执行次数另行统计。

## 环境和验证

数值/测试依赖见 `requirements-lock.txt`，CPU学习依赖见 `requirements-learning.txt`。依赖安装完成后，可用 `python -m pip install -e . --no-deps --no-build-isolation` 安装项目入口。`python -m dr_research` 与 `dr-research` 使用同一入口。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`artifacts/` 保存本次按需生成的实验产物并由Git忽略。旧单保真检查点与新能量消息结构采用不同版本，不能混用。

