# 本地运行、费用量级和导出可比性

## 当前交付状态

计算运行于本机 Windows 的仓库 `.venv`，使用 CPU；Torch、HiGHS 都设置为 1 线程。完整教师与粗分支各完成 4 组超参数训练，选定模型为 `artifacts/full_20261009/hpo/selected.pt`。无需重新生成标签或训练即可继续当前实验。

2026-10-09 按用户要求停止自动实验。LP-MPC 的 S1—S5 全部完成；GNN 的 S1—S4 完成。S5 GNN 在中途被中断，其进度日志仅用于诊断，没有写入完整周结果。消息保真度消融和 LP 标签误差分解尚未执行。

最大规模 LP-MPC 的 4694.07 秒是 336 次控制的累计计算时间，平均每步 13.97 秒，最大单步 19.85 秒；它模拟的是一周 168 小时。此前长时间等待主要来自这一组 LP 求解。

## 339.06 万元的来源与现实性

按保存的原始功率轨迹重新计算：

\[
C=\sum_{t=0}^{335}c_t\,\Delta t\sum_iP_{i,t}
=6{,}280{,}806.47\ \mathrm{kWh}\times0.53984245\ \mathrm{元/kWh}
=3{,}390{,}645.92\ \mathrm{元}.
\]

其中，$P_{i,t}$ 为热泵电功率，单位 kW；$\Delta t=0.5$ 小时；$c_t$ 为该时段电价；0.53984245 是按实际耗电加权的电价。核查结果显示半小时时长只计入一次，功率使用电功率，价格单位为元/kWh，报告数值单位为元。

| S5 量级 | 数值 |
|---|---:|
| 热泵/储罐/建筑节点 | 2048 / 512 / 256 |
| 总额定制热功率 | 174.592 MW |
| 总额定电功率 | 75.480 MW |
| 平均实际电功率 | 37.386 MW |
| 一周建筑需热量 | 13,870.466 MWh（热量） |
| 平均/峰值热需求 | 82.562 / 119.264 MW |
| 单建筑平均热需求 | 322.509 kW |
| 单建筑一周分摊电费 | 13,244.71 元 |

结论：该费用与现有大系统参数及实际轨迹一致，当前证据没有显示单位或重复计费错误。**它是规模化仿真系统的电量电费，现实项目的费用水平尚未校准。** 当前建筑需求按配套设备额定供热量、温差、占用率生成；建筑面积、围护结构和实测负荷尚未确定。因此节点数量增加时，容量、需求和电费几乎同比增加。

63/90/125 kW 与 COP 2.29/2.38/2.29 对应 [2G C-90 产品表](https://2-g.com/en/products/heat-pumps/afilia-water-series) 的 70/90°C 热侧、10/5°C 源侧额定工况。采用真实设备额定点不等于已经复现真实全年效率；当前固定 COP 也没有跟随实际源温、供水温度及部分负荷变化。

现实校准应先确定每个建筑节点代表的建筑面积、用途和负荷，再配设备与储热容量；独立确认项目电价。现有指标只含热泵电量电费，未独立计入循环泵等辅机、容量/需量收费和其他账单项目。固定现有参数完成比较可用于评估算法；修改物理需求、COP 或电价后，两方法应在新场景重新配对，并重新生成适用标签和模型。

## 由你继续当前 GNN 实验

在 PowerShell 中执行：

```powershell
Set-Location 'C:\Users\10632\Documents\GitHub\DR_Research'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -m dr_research --physical-config configs/experiment.toml --settings configs/comparison.toml compare --hpo artifacts/full_20261009/hpo --out artifacts/full_20261009/evaluation --resume
```

**本次命令跳过已经完成的 9 组运行，只补跑 S5 GNN，LP-MPC 无需重跑。** S5 GNN 从第 0 步重新开始；目前恢复粒度是“完整的一周运行”，不支持从中断的控制步继续。完成后自动生成 `evaluation/summary.csv` 与 `evaluation/manifest.json`。

模型、物理参数、候选数、源码和运行环境保持原合同。当前原始候选数为 256，另含备用动作；预测与续接窗口为 48 步。主种子为 42。当前已保存线程环境 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 均未显式设置。若恢复提示合同不符，应先核对差异，不要覆盖 `protocol.json` 或删除已有 LP 结果来绕过检查。

运行时间参考：S5 GNN 中断前完成 138 步，累计控制时间约 187 秒，约 1.36 秒/步；一周可按约 8 分钟准备，实际以完整运行记录为准。这个估计不含重新训练。运行期间避免同时训练或执行其他 CPU 密集任务。

完成后只做数据检查、不会运行控制器的命令：

```powershell
.\.venv\Scripts\python.exe scripts/audit_exports.py --evaluation artifacts/full_20261009/evaluation --out artifacts/full_20261009/comparability_after_resume.json
```

输出文件禁止覆盖；重复审计时换一个输出文件名。`scripts/summarize_results.py` 是包含完整消融的研究报告生成器，要求消融和误差分解均完成，当前交接阶段无需运行。

## 导出可比性检查结果

本次 [审计记录](../artifacts/full_20261009/comparability_audit.json) 对 9 组已完成记录进行了报告/轨迹哈希检查、独立整周物理回放、逐步电费重算及总耗时求和检查，全部通过。S1—S4 形成 4 对完整结果，S5 当前只有 LP-MPC，完整 GNN 结果补齐前不计算其周电费差。

| 比较项 | 统一口径 |
|---|---|
| 外部场景 | 同规模共享相同 `scenario.npz` 与场景指纹 |
| 时间轴 | 336 个半小时步，索引 0—335 |
| 预测窗口 | 48 步，临近周末随剩余时段缩短 |
| 物理边界 | 相同初始状态、设备、图、热需求、电价和周末库存目标 |
| 电费 | 按最终执行电功率重算，不采用模型预测费用作为实测结果 |
| 完整控制时间 | 初始化（包括 GNN 加载）加每步生成/求解/评分/续接/动作检查；两法均排除环境推进与事后审计 |
| 离线耗时 | 标签、训练、调参、标定单独报告，不加入在线控制时间 |
| 物理验收 | 全周动作与库存轨迹通过同一校验器 |
| 备用处理 | 备用动作实际电费与计算时间保留，回退与超时次数单列 |
| 原始轨迹列 | 两种方法使用完全相同 NPZ 字段、物理单位和时间轴 |

相同轨迹字段不意味着所有字段都有相同算法含义：`selected_candidate`、`candidate_pool_size` 等只对 GNN 有意义；LP 的占位值不用于比较候选行为。比较主指标使用 `total_electricity_yuan`、`control_time_s`，同时检查 `accepted`、`executed_steps`、`fallback_count`、`deadline_misses` 和单步时延。

当前两法经济结果的协议可比；耗时是本机同版本单次测量，跨电脑或后台负载差异较大时，应在相同条件重新配对测量。候选评价使用有限续接规则，LP-MPC 直接规划窗口，两者执行策略空间存在差异，这属于方法差异，应结合回退次数解释。

## 返回哪些文件

把 `artifacts/full_20261009/evaluation/` 整个目录打包即可，其中包含 `summary.csv`、`manifest.json`、`protocol.json`、`completed.json` 及每个规模的场景、两方法 JSON 报告和 NPZ 轨迹。另附 `comparability_after_resume.json`。需要分析训练质量时，再附 `hpo` 下的 JSON 报告；不必默认上传大型标签数据或模型权重。

中途日志不作为完整周结果；请保留原始元、kW、kWh、秒单位，不在导出时将一方改成万元或分钟。
