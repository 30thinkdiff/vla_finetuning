# Recovery-Aware SmolVLA：集成缺口与可执行验收顺序

**状态：P0 静态实现/纯CPU合同检查。没有真实Recovery样本或GPU训练/闭环验证。**
与正在运行的Corrector训练隔离；不要修改 \`/root/autodl-tmp/corrector\`。

## 已有资产与新模块

| 工程点 | 当前代码 | 验证边界 |
|---|---|---|
| 原始LeRobot CLI训练命令/显式GPU空闲门槛 | \`src/vla_finetuning/engine.py\`, \`guards.py\` | **仅针对Plus协议**；不能自动用于ModelScope Expert多数据混训 |
| v3分片解析 | \`dataset_files.py\` | **只有**Expert实测为v3且字段符合时才能复用 |
| 显式ModelScope单文件和子集获取计划 | \`scripts/download_recovery_metadata.py\`, \`scripts/download_recovery_subset.py\` | SDK下载后核大小，**下载过程并非硬字节上限** |
| 场景/Expert/动作静态审计 | \`recovery_contract.py\` | 读取本地JSON/JSONL；绝不推断Demo与失败场景一一对应 |
| 强制验证来源的global split | \`recovery_experiments.py\` | 验证人工标签schema与隔离，**不能证明人工标签属实** |
| M0/M1/M2可复现的Episode抽样计划 | \`recovery_experiments.py\` | 只是Episode ID日程；**不是**真实LeRobot DataLoader/训练 |
| 独立失败闭环结果配对分析 | \`recovery_reports.py\` | 源聚类bootstrap＋level/gripper分组，但仍需逐回合视频/环境Trace审核 |
| CLI | \`python -m vla_finetuning.recovery_cli --help\` | 无外部依赖，默认只读 |

## 真实训练与评测尚需开发/验收的5个硬阻塞

1. **实际Expert文件版本和解码**：ModelScope固定修订、文件真实大小、Parquet/JSONL索引/视频、任务标签、\`[C,H,W]\`图像与动作Chunk，验证至少一条完整Episode。
2. **控制/归一化契约**：逐维确认7D相对动作与抓手码、坐标系与旋转约定、摄像头翻转/颜色顺序、state 8D、FPS/20Hz映射；逐值比较官方Recovery evaluator与SmolVLA处理器，不容只检查Shape。
3. **真实源关联和防泄漏**：Expert Episode到失败 scene/source rollout 的证据，官方测试来源保留，普通示教与pretrained模型已见场景的范围。无法关联时只能作受限适配实验，不能宣称独立失败场景泛化。
4. **真正的双数据源训练DataLoader**：LeRobot单仓库 \`--dataset.repo_id\` 入口不等于跨ModelScope Expert与HF LIBERO的混合训练。需验证上游 \`MultiLeRobotDataset\` 或实现兼容数据物化/采样器；保持action归一化和每个Batch的任务语言/处理器；用真数据做50步GPU smoke、权重更新/保存重载后才可扩长训。
5. **真实Recovery仿真策略适配**：将SmolVLA策略服务器/推理处理器接到官方 \`eval_libero_custom_scene.py\`（不要假设能直接调用Plus的 \`lerobot-eval\`）；冻结场景、初始sim_state、每次±2cm扰动、抓手open/close、种子和520步终止协议，输出逐场景逐回合报告和视频。

## 预先冻结实验矩阵

| 方法 | 定义 | 作用 | 开工门槛 |
|---|---|---|---|
| M0 | 原始 \`smolvla_libero\`，0训练步 | Recovery + normal baseline | 样本语义和评测适配通过 |
| M1 | 仅Expert，LoRA | 恢复学习与正常遗忘 | 真Batch/optimizer/reload成功 |
| M2 | Expert + normal，固定预算/LoRA | 已有论文思路的混合训练基线 | 真正等预算多源DataLoader |
| M3 | 数据支持后**仅选一个**机制（如冲突梯度/来源采样） | 验证明确研究假设 | M0–M2完整闭环、独立预注册机制 |

所有方法必须注明：base模型revision、初始权重及processor、LoRA目标层/rank/alpha、
输入源/数据split hash、采样单位（frame vs episode）、seed、steps、有效batch、梯度累积、
计算耗时/GPU-hours、动作控制频率/执行窗口、每场景评测种子与初始状态hash。
研究假设未成立之前**不开发所谓“创新M3”强行搜好成绩**。

## 离线P0操作（**不启动训练**）

\`\`\`bash
cd /root/autodl-tmp/vla_finetuning
git fetch origin
# 合并 PR 后再 git pull --ff-only；未合并时可单独检出PR分支
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m vla_finetuning.recovery_cli --help

# 下列文件只有真实下载后才执行；不存在时预期STOP
PYTHONPATH=src .venvs/data/bin/python -m vla_finetuning.recovery_cli \
  audit-expert --info datasets/recovery_metadata/PINNED/meta/info.json

PYTHONPATH=src python -m vla_finetuning.recovery_cli \
  audit-scenes --index datasets/recovery_scenes/index/selection.jsonl

# 正式来源映射需要人工验证 source_evidence；模板默认BLOCKED
PYTHONPATH=src python -m vla_finetuning.recovery_cli \
  audit-split --split configs/recovery_split.template.json

# M1/M2样本日程仅为Episode抽样计划，不是训练：
PYTHONPATH=src python -m vla_finetuning.recovery_cli plan-mixture \
  --split runs/REAL_AUDITED_SPLIT.json --method M2 \
  --steps 50 --batch-size 2 --recovery-fraction 0.5 \
  --seed 42 --out runs/M2_DRAW_SCHEDULE_001.json
\`\`\`

报告CSV至少包含：
\`suite,level,scene_id,source_group,gripper_init,seed,method,success,initial_state_sha256,protocol_sha256,rollout_evidence\`。
这里 \`initial_state_sha256\` 必须描述**每次实际扰动并settle后的同一评测状态**，
不是所有trial复用某个原场景文件hash。对 M0/M1/M2 用相同“场景＋seed＋gripper初始状态”配对；
报告先拒绝重复、缺组及SHA不一致，再按来源场景聚类重采样。

## 拒绝完成的行为（故意保留阻断）

- 不把读取 \`meta/info.json\` 当作 action semantically verified；
- 不把Expert/FailureScenes两套ID按 task_name 自动join；
- 不在没有基模型、原训练数据与专家恢复来源证据时启动正式LoRA；
- 不复制原正确工作中的Corrector的venv，也不修改其GPU训练调度；
- 不因CI纯CPU通过就把M0–M3标记为训练完成；
- 不凭相似视觉场景的训练后成功率宣称零样本泛化。

## 后续真实数据到位时的优先级

(1) 一条Expert Episode和一条FailureScenes官方场景的**只读逐值**检查；
(2) 双相机/动作/统计处理器与官方 evaluator 的契约测试；
(3) 单回合M0闭环、视频与控制动作核对；
(4) 独立训练环境50步LoRA实测和Checkpoint验证；
(5) 冻结M1/M2实验矩阵，再考虑源样本混合，最后才是M3。

## 新增：仅生成Recovery闭环试验清单（不执行仿真）

```bash
PYTHONPATH=src python -m vla_finetuning.recovery_cli plan-eval \
  --index datasets/recovery_scenes/index/selection.jsonl \
  --methods M0,M1,M2 --seed 42 --per-suite-level 2 \
  --open-trials 1 --close-trials 1 \
  --out runs/recovery_eval_plan_001.json
```

这会按官方索引的suite×level分桶稳定选择场景、列出方法与预期回合数，但**不会**检查真实BDDL/npy、运行官方随机采样/settle、加载模型或产生真实初始状态SHA。输出`perturbed_state_sha256=null`是故意的：只有实际仿真状态生成以后才能填入。它不是作者默认采样器的等价复现。M0/M1/M2对应的checkpoint依然必须另行固定并审核。
