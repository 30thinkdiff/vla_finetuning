# SmolVLA Fine-tuning 工程使用手册

> 2026-11-19。这里的**框架已实现**不等于已在AutoDL执行、下载数据、完成真实训练或LIBERO-Plus闭环。所有需要GPU/大量CPU/网络/磁盘的操作必须先由使用者独立检查2号机资源。

## 0. 技术基线及目录

- **唯一正式训练引擎：**固定LeRobot源码 `7e241bd630a3719a56157a497ce5d08f244784f1` 的 `lerobot-train`；训练业务逻辑不重写。继续使用历史已实际训练过的LoRA参数配置作为初始候选，实际性能须通过新的真实GPU试验确认。
- **数据：**LeRobot v3.0 `lerobot/libero_plus`；只允许使用固定40位Revision。少量已审计的真实Episode由source_group组划分后选择，不能拿`task_index`当仿真variant task ID。
- **模型：**`lerobot/smolvla_libero`，已经见过原始LIBERO数据；不重复把原LIBERO示范作为新任务泛化证据。
- **Recovery：**ModelScope `ataier/LIBERO_Recovery_Expert` 在审计阶段只读取单个小元数据文件。完整恢复示范、失败场景、模拟器快照的训练与评测适配未完成真实集成，不得假设能直接拼到Plus。
- **仿真：**固定LIBERO-plus `4976dc30028e805ff8094b55501d532c48fec182`；通过官方`lerobot-eval`接入，而不是自写一个隐含改变控制语义的评测循环。

## 1. 安全边界

2号机截至最近输出：550GB数据盘已用415GB/约135GB余量；4090正在执行Corrector S2_s42。**目前只运行本地/远程的轻量只读Python标准库命令。不要执行脚本的`--apply`，不要运行训练、仿真、Torch安装、批量下载。**

源码和配置进入Git；真实数据集、模型、Checkpoint、缓存、视频全部属于.gitignore。训练与仿真各自独立Venv，不修改`/root/autodl-tmp/corrector`及其`runtime/venv`。

## 2. 已实现，资源安全的代码（无需Torch）

```bash
cd /root/autodl-tmp/vla_finetuning
git pull --ff-only origin main
PYTHONPATH=src python -m unittest discover -s tests -v

# 正确实现参考：数据任务解析、H步动作padding、配对回合分析
PYTHONPATH=src python -m vla_finetuning.pipeline reference-demo

# 不需要真实Checkpoint/数据，用来验证安全拒绝
PYTHONPATH=src python -m vla_finetuning.pipeline validate-config \
  --config configs/pipeline.template.json

# 用虚构评测记录测试分析工具；不能当作算法成功率
PYTHONPATH=src python -m vla_finetuning.pipeline report \
  --csv examples/paired_results.synthetic.csv
```

`validate-config`返回STOP是**预期**，因为模型SHA、数据SHA及实际Episode尚未冻结；下一步填真实数据后才可通过。

完整参考函数：`src/vla_finetuning/reference.py`。学生可先阅读实现，再到`exercises/student.py`重新写一遍，通过`exercises/tests/test_student.py`验证；这些教学TODO不会参与正式训练与评测。

## 3. 环境搭建脚本（**当前不要执行--apply**）

```bash
bash scripts/bootstrap_train_env.sh   # 只打印将要安装的内容
bash scripts/bootstrap_sim_env.sh     # 只打印独立仿真环境安装计划
```

- 经用户检查CPU/GPU/IO、确认空闲时：设置`VLA_APPROVED_CPU_IO=YES`并手动追加`--apply`。
- `bootstrap_train_env.sh`：固定LeRobot源码SHA；在`.venvs/train`安装Torch、PEFT、PyArrow/PyAV及LeRobot依赖；需实际网络与包兼容性集成验证。
- `bootstrap_sim_env.sh`：独立`.venvs/sim`；安装固定LIBERO-plus、robosuite、MuJoCo等；**资产ZIP需要额外人工确认，并未自动下载**。GitHub源码和模拟器资源是独立依赖。
- 验证环境可用必须检查`pip check`、`import lerobot`、`import libero`、EGL最小场景reset/step；这不是CPU单元测试可以替代的。

## 4. 安全下载（默认只规划，不下载）

### 4.1 固定版本、读取元数据

已有`python -m vla_finetuning hf-pin`与`fetch-metadata`支持**显式允许联网、单文件上限512KiB**。完整操作见[阶段A说明](data_stage_walkthrough.md)。再通过`meta/info.json`识别格式，必要时按budget拉取整个`meta/`供v3索引读取。

### 4.2 计划真实Episode所需分片

完整数据规划使用下列步骤，均需在你审核机器资源以后：

```bash
# 仅需独立工具环境中的huggingface_hub + PyArrow，不能依赖Corrector runtime。
PYTHONPATH=src python -m vla_finetuning.pipeline shard-plan \
  --metadata-root /root/autodl-tmp/vla_finetuning/datasets/libero_plus_PINNED \
  --split /root/autodl-tmp/vla_finetuning/runs/AUDITED_SPLIT.json \
  --repo-id lerobot/libero_plus \
  --out /root/autodl-tmp/vla_finetuning/runs/shards_001.json

# 仅调用Hub文件清单API（需手动 --allow-network）
PYTHONPATH=src python -m vla_finetuning.pipeline hub-plan \
  --kind dataset_payload --repo lerobot/libero_plus \
  --revision "$DATA_SHA" --shard-plan runs/shards_001.json \
  --allow-network --out runs/download_payload_001.json

# 真正下载前先核对服务器IO、CPU、剩余盘空间；默认不执行
PYTHONPATH=src python -m vla_finetuning.pipeline hub-download \
  --plan runs/download_payload_001.json \
  --target /root/autodl-tmp/vla_finetuning/datasets/libero_plus_PINNED \
  --project-root /root/autodl-tmp/vla_finetuning --max-gib 10
```

最后一条**没有--apply会拒绝**。默认如果发现GPU存在Corrector进程，即使加`--apply`也会拒绝下载；只有计算资源实际空闲以后才允许正式大文件传输。

模型下载同理：`hub-plan --kind model --repo lerobot/smolvla_libero --revision MODEL_SHA --allow-network`，再`hub-download`，限制总GiB并单线程。无需为了下一步学习而提前下载完整权重。

### 4.3 Recovery元数据

```bash
PYTHONPATH=src python scripts/download_recovery_metadata.py \
  --revision REPLACE_WITH_REAL_MS_REVISION
```

默认只显示计划。实际`--apply`使用ModelScope SDK获取单个`meta/info.json`，由于其SDK接口**不能保证传输过程中严格字节上限**，必须先确认远程文件大小和机器IO空闲。**目前不执行。**

## 5. 真实Batch与训练（等待4090空卡）

先冻结`configs/pipeline.template.json`的拷贝，将真实40位SHA、源轨迹审核split和训练Episode写入。确保真实模型/数据已经在本地独立目录完整下载。

```bash
# 只生成官方CLI，不训练（模型与数据真实可用以后）
PYTHONPATH=src python -m vla_finetuning.pipeline train \
  --config configs/your_frozen_run.json \
  --binary /root/autodl-tmp/vla_finetuning/.venvs/train/bin/lerobot-train

# 可选：你审查资源并安装独立环境后，CPU只读取1个真实样本
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  .venvs/train/bin/python -m vla_finetuning.real_batch \
  --repo lerobot/libero_plus --revision "$DATA_SHA" \
  --root /root/autodl-tmp/vla_finetuning/datasets/libero_plus_PINNED \
  --episode 123
```

**正式训练强制条件**：用户明确`--apply --confirm-gpu-idle`、nvidia-smi没有其他Compute PID、数据盘余量足够、源代码/模型/数据SHA已冻结、独立训练环境和真实Batch的维度/动作语义经过人工验收。单卡4090先50步smoke，统计实际参数梯度、显存与Checkpoint后再决定正式训练步数。不要直接从训练Loss推断任务成功率。

Checkpoint静态审计：

```bash
PYTHONPATH=src python -m vla_finetuning.pipeline checkpoint-audit \
  --checkpoint /path/to/checkpoints/000050
```

只列举文件，**不能代替真实保存/重载、optimizer和RNG状态恢复测试**。

## 6. LIBERO-Plus闭环评测（等待仿真集成验收）

```bash
PYTHONPATH=src python -m vla_finetuning.pipeline evaluate \
  --config configs/your_frozen_run.json \
  --checkpoint /path/to/real/checkpoint/pretrained_model \
  --output-dir /root/autodl-tmp/vla_finetuning/runs/eval_001 \
  --binary /root/autodl-tmp/vla_finetuning/.venvs/sim/bin/lerobot-eval
```

此命令默认**只打印官方eval argv**；加`--apply --confirm-gpu-idle`才会尝试执行。还需验证eval的`policy.path`、相机/7维动作语义、`control_mode`、任务语言、H20执行策略、EGL资源和checkpoint PEFT处理器恢复，才算真正完成联调。一个`task_id`代表一个**LIBERO-plus变体**，不要直接当作训练集`task_index`。

真实配对CSV字段：`domain,condition_id,seed,method,success`。至少保留B0原始策略、L1 LoRA策略，在相同物理初始状态、语言、相机、控制频率、horizon、噪声/seed条件下完成独立回合，最后：

```bash
PYTHONPATH=src python -m vla_finetuning.pipeline report \
  --csv /path/to/real_paired_rollouts.csv \
  --out runs/paired_summary_001.json
```

报告包含baseline/candidate成功率、双方都成功、都失败、救回和退化回合。**统计结果必须来自实际运行、视频和环境记录，而不是虚构CSV。**

## 7. 工程质量与责任边界

| 完成状态 | 说明 |
|---|---|
| 已实现、纯CPU CI覆盖 | 来源审核、组划分、下载计划、安全开关、参考动作Chunk、配对统计、协议静态检查 |
| 已实现、等待外部依赖联调 | pinned train/sim环境Bootstrap、官方数据分片解析、Hub下载、真实样本检查、官方train/eval命令适配 |
| 尚未完成外部验证 | LIBERO-plus原始场景EGL实测、SmolVLA真实Batch/优化器梯度、Checkpoint续训、两模型成对闭环 |
| 第二阶段 | LIBERO-Recovery实际示范转换、场景按失败源隔离和针对性恢复微调；不得与第一阶段视角改进混合归因 |

关键风险：Hugging Face/ModelScope SDK、PyTorch/PyAV以及LIBERO-plus版本变化可能影响上游API。固定Revision只是必要条件，不是实际兼容性测试。只有当2号机资源空闲且用户主动进行真实联调，才能把这些部分验收为生产可运行。

## 8. Checkpoint断点续训（只生成命令）

```bash
PYTHONPATH=src python -m vla_finetuning.pipeline resume-plan \
  --config configs/your_frozen_run.json \
  --checkpoint /root/autodl-tmp/vla_finetuning/runs/plus_expert_lora_001/checkpoints/000050 \
  --binary /root/autodl-tmp/vla_finetuning/.venvs/train/bin/lerobot-train
```

此处只验证Checkpoint目录存在`pretrained_model/train_config.json`、`training_state/`、相同的训练输出路径与数据集身份，打印`--resume=true --config_path=...`。**暂不自动Resume**；实际优化器、调度器、随机状态恢复是否无损，需要第一次真实GPU训练保存Checkpoint后才能验证。

## 9. 从LeRobot官方结果自动生成成对比较

固定两个策略使用同一注册变体集和种子、相同模拟器revision/初始状态/任务语言与H20执行窗口，待两组评测结束且`eval_info.json`存在时执行：

```bash
PYTHONPATH=src python -m vla_finetuning.pipeline merge-eval \
  --baseline-info runs/B0/eval_info.json \
  --candidate-info runs/L1/eval_info.json \
  --start-seed 1000 --out runs/paired.csv

PYTHONPATH=src python -m vla_finetuning.pipeline report \
  --csv runs/paired.csv --out runs/paired_summary.json
```

这套解析器直接读取固定LeRobot源码的`per_task[].metrics.successes`，根据受控`start_seed`建立注册Variant的成对结果。**它只验证结果记录的数学匹配，不验证两个Rollout的真实物理初始状态完全一致。**数据标签应结合保存的视频和环境Trace核对，不能仅凭CSV发布性能提升结论。

Recovery专家示范还可以在审核远程文件清单及每项`size_bytes`之后，使用`scripts/download_recovery_subset.py --manifest <人工审核清单>`生成下载计划。默认不下载；加`--apply`仍要求GPU无其他计算进程，且ModelScope SDK实际传输字节上限**不属于强制流式配额**，必须由用户先检查后才能执行。
