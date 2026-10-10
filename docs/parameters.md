# 实验参数

两种控制方法共享物理配置 `configs/experiment.toml`。完整实验设置与小规模流程检查设置分别为 `configs/comparison.toml` 与 `configs/comparison_smoke.toml`。当前物理配置仅支持一周336个半小时步。

## 需求与温度

每栋建筑的供热需求为

\[
D_b(t)=\lambda Q_{b,\mathrm{nom}}f_b
\frac{[T_{\mathrm{in}}-T_{\mathrm{out}}(t)]_+}{\Delta T_{\mathrm{ref}}}
\left(1+\gamma o(t)\right).
\]

其中，\(D_b(t)\) 为建筑 \(b\) 在时刻 \(t\) 的热需求，单位kW；\(Q_{b,\mathrm{nom}}\) 是其骨架设备额定供热量；\(\lambda\) 是负荷系数；\([x]_+=\max(x,0)\)；\(T_{\mathrm{in}}\) 是需求参考室温；\(T_{\mathrm{out}}\) 是室外温度；\(\Delta T_{\mathrm{ref}}\) 是参考温差；\(o(t)\) 是占用比例；\(\gamma\) 是占用增益。\(f_b\) 是一次抽取后整周固定的建筑系数。

| 配置字段 | 默认值 | 含义 |
|---|---:|---|
| `demand.mean_load_fraction` | 0.45 | \(\lambda\) |
| `demand.indoor_c` | 20°C | \(T_{\mathrm{in}}\) |
| `demand.reference_degree_difference_c` | 15 K | \(\Delta T_{\mathrm{ref}}\) |
| `demand.building_factor_half_width` | 0.05 | 原始建筑系数从[0.95,1.05]均匀抽取 |
| `demand.occupancy_gain_gamma` | 0.10 | \(\gamma\)，净供热增益假设 |
| `demand.occupancy_transition_hours` | 1 h | 占用切换的过渡时长 |

原始建筑系数按本站额定供热量加权均值归一化，归一化后的系数可略超出原始区间。S1只有一栋建筑，其归一化系数恒为1。当前没有随控制时段重新抽取的负荷噪声，也没有预测值与真实需求之间的偏差。固定场景种子和天气周可复现同一需求曲线。

工作日07:00前及20:00后占用，周末全天占用；默认切换区间为06:30—07:30和19:30—20:30。该曲线是住宅净供热需求的可调仿真假设，当前不模拟室内温度动态、人员得热和生活热水的独立状态。

温度默认取南京CSWD典型气象年七天片段，小时结束时刻的温度线性插值到半小时。每个实验周指定周一开端。训练周0—7、验证周8—9、测试周10互不重叠。原始气象来源、哈希与物理可行周筛选保存在 `src/dr_research/data/weather/manifest.json`。`weather.provider="synthetic_week"` 可用配置中的日正弦温度做隔离诊断。

## 设备与连接

| 参数 | 默认设置 |
|---|---|
| 热泵额定供热量 | 63、90、125 kW，比例50%、25%、25% |
| 热泵COP | 2.29、2.38、2.29，当前固定 |
| 额定电功率 | 额定供热量/COP |
| 储热罐容积 | 50 m³ |
| 储热温度范围 | 70—90°C，初末80°C |
| 充放热效率 | 均为0.98 |
| 罐散热系数 | 0.026 kW/K |
| 罐周围温度 | 10°C |
| 最低负荷比例 | 额定电功率的0.10 |
| 单步爬坡 | 额定电功率的0.50 |
| 源热容量比例 | 额定取热量的0.90 |
| 总功率带 | 额定总电功率的[0.15,0.95] |

热泵额定值对应[2G afilia water系列](https://2-g.com/en/products/heat-pumps/afilia-water-series)，储罐容积参照[Dehoust 112065](https://www.dehoust.com/en-001/Products/745FEEDA-F724-451E-B37D-972AF3124B89/50-m-D-2700-mm-L-10900-mm)。固定COP、完全混合罐、效率、散热系数、爬坡和运行模式为实验近似。

每罐连接四台热泵，每栋建筑有两罐骨架连接，每站最多四栋建筑。额外S→B边只在站内按概率0.45生成；半径为相邻两栋。图拓扑、设备排列和建筑系数使用由主种子派生的场景种子。热泵接收罐固定，当前没有开停机或接收罐切换决策。

## 电价

谷、平、峰、尖峰价格依次为0.3815、0.6756、1.0376、1.2005元/kWh。每日24个价格档位由 `tariff.hourly_periods` 指定，半小时内保持对应小时档位。该配置是研究采用的江苏时段曲线，修改时直接调整四个价格及档位数组。

## 控制和训练

以下数值来自 `configs/comparison.toml`，小规模检查以其独立配置文件为准。

| 配置字段 | 默认值 | 用途 |
|---|---:|---|
| `schema_version` | 2 | 能量消息与预算精化配置合同 |
| `master_seed` | 42 | 场景、取点、探索、初始化及训练顺序的派生种子 |
| `control.window_steps` | 48 | 24小时费用预测及LP窗口 |
| `control.safety_window_steps` | 48 | 评分后续接的最大窗口 |
| `control.time_limit_s` | 60秒 | 单轮端到端控制时限 |
| `control.solver_threads` / `torch_threads` | 1 / 1 | 求解器与神经计算线程数 |
| `control.candidate_count` | 256 | 原始Sobol点数N，备用动作另计 |
| `control.max_continuation_attempts` | 8 | 按评分顺序验收的候选上限 |
| `data.sample_stride` | 24步 | 收集离线标签状态的步间隔，使用场景相关固定相位 |
| `data.sampling_count` | 128 | 离线探索的原始Sobol点数 |
| `data.label_candidates` | 17 | 每个采样状态最多求LP标签的候选数 |
| `training.epochs` / `patience` | 60 / 12 | 教师训练轮数上限及早停等待 |
| `training.batch_size` | 8 | 训练分组批量设置 |
| `training.hidden_dims` / `latent_dims` | [32,64] / [8] | 完整消息隐藏宽度与共享潜消息宽度 |
| `training.learning_rates` | [0.0003,0.001] | 教师学习率搜索 |
| `training.coarse_widths` | [2,4] | 粗门控宽度搜索 |
| `training.coarse_epochs` / `coarse_warmup_epochs` | 30 / 5 | 粗分支决策训练与初始化轮数 |
| `training.temperatures` | [0.05,0.2] | 公式24选择温度搜索 |
| `training.error_epochs` | 40 | 消息残差预测器训练轮数 |
| `evaluation.repeats` | 1 | 每个测试场景的重复次数 |

训练周0—7，验证周8—9，测试周10。教师训练规模为S1、S2，闭环测试规模为S1—S5。控制、取样和训练分支只接收其对应分割的数据。

`candidate_count` 与 `sampling_count` 必须为2的幂。原始点、当前合法点、拒绝点、重复点和最终候选数分别记录。候选0是已验收备用动作，其余候选ID等于原始Sobol行号加1；过滤和去重保留原ID，因此ID可能不连续。

每个候选只表示当前一步动作及后继库存。评分前不生成未来控制轨迹。取样器先采样总功率，再条件分配，并通过保持约束的功率和供热转移探索合法动作。受限可行盒、逐栋需求适配及续接限制见 [方法实现范围](method_status.md)。

离线标签固定候选首动作，求共同窗口LP的后续最低电费。按状态保存候选组、原始ID、精确当前费用与未来标签。探索动作按固定随机流选择并通过续接验收，不使用标签引导。LP未产生可用最优标签时记录其原始状态，不填入零标签。

## 消息精化

| 配置字段 | 默认值 | 用途 |
|---|---:|---|
| `refinement.mode` | adaptive | coarse、full、random或adaptive内部模式 |
| `refinement.time_budget_s` | 2秒 | 准备、基础评分、梯度、调度及精化的总预算 |
| `refinement.shortlist_size` | 8 | 领先与近邻竞争候选数 |
| `refinement.coverage_count` | 2 | 每轮额外覆盖候选数 |
| `refinement.group_size` | 64 | 每个精化组最多包含的充热消息数 |
| `refinement.max_rounds` | 16 | 最大精化轮数 |
| `refinement.reserve_s` | 0.001秒 | 评分流程末尾输出预留 |

增量延迟回归树依据固定硬件、模型宽度、精度、缓存和消息工作负载校准。预测为中位耗时，实际超时单独记录。整轮控制还受 `control.time_limit_s` 约束，评分结束后需要保留续接验收时间。

使用 `compare --gnn-mode full` 等参数做内部消融；`--scales S1 S2` 限定已声明测试规模的子集。`--candidate-count 1024` 仅改变评估时的原始取样密度；更大候选池需要用相同权重重新测量对应规模的增量批次，并通过 `--latency-profile` 指定保存的回归树JSON。硬件、线程、精度、网络架构或图规模改变时也需重新校准；在线池大小超过已校准上限时停止精化并记录原因。这些评估覆盖项记录于输出协议，不修改已保存模型的训练合同。`--resume` 仅复用设置、代码、权重、运行环境和场景都匹配且报告/轨迹校验通过的已完成运行。

费用预测窗口与安全续接窗口分别决定标签/输入和评分后验收。修改物理设置、窗口、训练数据或模型架构后重新生成相应数据与权重。新旧数据及检查点版本分离，旧单保真权重不能装载为完整消息模型。
