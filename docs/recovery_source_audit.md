# Recovery 来源与数据契约审计（P0，2026-11-19）

> 本文基于 [LIBERO-Recovery 官方仓库](https://github.com/liulin815/LIBERO-Recovery) README 与
> [eval_libero_custom_scene.py](https://github.com/liulin815/LIBERO-Recovery/blob/main/examples/LIBERO/eval_files/eval_libero_custom_scene.py)。
> 审核时的官方代码 HEAD 为 832d54dfc43f0c9e4240bb8d92f5fda7001bb093（2026-10-22，v1.0.0）。
> **并未**读取 ModelScope 官方 Expert 数据集的真实 meta/info.json 或任一训练 Episode，
> 不应凭官方仓库的示例代码断言下载后的列名、数据体积、归一化或数据源对应关系。

## 1. 公开来源分工（不要互换）

| ID | 类型 | 已核实用途 | 仍未知 |
|---|---|---|---|
| ModelScope \`ataier/LIBERO_Recovery_FailureScenes\` | **失败场景库** | 4,803 条带 L1–L4 标签的场景；\`scenes/<suite>/*.bddl\` + 相邻 \`*_sim_state.npy\`，\`index/selection.jsonl\`，\`annotations/\` | 模型仓库固定版本、实际包大小、当下完整文件清单及SHA |
| ModelScope \`ataier/LIBERO_Recovery_Expert\` | **专家恢复示教** | 官方 README 称其为 LeRobot 格式，用于**训练**而不是自动作为评测场景 | 实际 LeRobot 版本、\`meta/info.json\` 字段、图像顺序、动作语义、Episode 文件清单、远端体积、Expert→源失败场景映射 |
| ModelScope \`ataier/LIBERO_Recovery_Assets\` | **可选精简场景** | 880 个作者选择的场景，默认 launcher 场景路径 | 是否与全量 index 的场景ID一致、固定版本 |
| ModelScope \`ataier/LIBERO_10_LL\` | **失败前历史轨迹** | 官方 README 定义为 pre-failure history training data | 真实字段/采样时序/关联方法 |
| ModelScope \`ataier/LIBERO_Recovery_Rollouts\` | **历史原始Rollout** | 1,446 条原始 .npz，官方README标注约 80GB，仅重现来源所需 | 下载包内容校验/精确体积 |
| HF \`lerobot/libero_plus\` | **视觉扰动数据（另一个研究域）** | 本地已下载 info/tasks及一个Episode索引分片，固定 revision \`f3f49f426d75030177b18778374005bc12ccd588\` | 不可用其任务或source ID推导 Recovery Expert 来源 |

请勿把 4,803 场景理解为 4,803 条专家示教；也不要把论文其他实验设定的场景数当成完整发布索引的数量。
官方 README 对 L1/L2/L3/L4 的定义依次为 Action Retry / Action Adaptation / Object State Recovery /
Environmental Recovery。作者的场景索引中有 \`scene_id/source/task/step/stage/level/npz_path\`
等**示例**字段；即便存在 \`npz_path\`，也不能证明任意 Expert Episode 来自那个源。

## 2. 上游评测动作与抓手协议

官方 \`eval_libero_custom_scene.py\` 的当前源码：
- 图像输入来自 \`agentview_image\` 与 \`robot0_eye_in_hand_image\`，源码进行二维翻转；
- 状态拼接 EEF position、quaternion 转 axis-angle、gripper qpos（合计预期8维，但必须在真实样本检查）；
- 从策略响应的 \`world_vector\` 3维、\`rotation_delta\` 3维、\`open_gripper\` 1维拼接7维 delta action；
- evaluator中的 gripper env 编码：\`+1=close, -1=open\`；它另有转换函数标示数据集 \`0=close,1=open\`；
- 使用 \`env.step\` 后按 \`done\` 判定成功；默认每场景 5 closed-gripper + 5 open-gripper 回合，
  每次随机 ±2 cm free-joint 物体水平扰动，最多520步。

**注意：上述是官方评测代码的接口描述，不代表某个下载的 Expert 数据集具备相同编码。**
不能把源码中的 \`_gripper_env_to_dataset\` 直接推广到任意 LeRobot Episode；
也不能假设 SmolVLA policy 输出归一化空间就等于真实 robot-controller 动作空间。
\`dataset_gripper_to_env\` 函数仅作为已核实情况下的纯CPU参考，默认拒绝转换。

## 3. 版本、元数据与训练前硬阻塞

必须取得并固定：
1. ModelScope Expert 和 FailureScenes 各自的**明确修订版、文件清单和校验值**；
2. Expert 的 **真实** \`meta/info.json\`、任务索引、Episode索引与至少一条可解码样本；
3. 图像键与前端/腕部相机含义，镜像/通道/空间预处理、state/action 各维、FPS与动作频率；
4. Expert Episode 与失败 scene/source rollout 的可靠来源链接，或明确无法证明；
5. 基础 SmolVLA checkpoint 与历史训练范围、归一化统计、PEFT target modules、processor一致性。

参考来源代码：
- 训练和官方CLI入口：\`src/vla_finetuning/engine.py\`、\`scripts/bootstrap_train_env.sh\`
- 原有LeRobot v3数据分片规划：\`src/vla_finetuning/dataset_files.py\`，**仅适用于核实为v3的数据**；
- 显式下载脚本：\`scripts/download_recovery_metadata.py\` 和 \`scripts/download_recovery_subset.py\`，
  默认dry-run、仍需要真实ModelScope文件大小及SDK行为确认；
- P0本地CPU审计：\`src/vla_finetuning/recovery_contract.py\`。

## 4. 第一批下载要怎样规划（此处不联网）

先在ModelScope网页/API或者只读清单中确认 **Expert 的真实路径、字节数和修订版本**。
README只是说LeRobot格式，**不能直接假设** \`meta/tasks.parquet\`、\`meta/episodes/*.parquet\`
一定存在，也不能默认 \`meta/tasks.jsonl\` 与 \`meta/episodes.jsonl\` 一定存在。
确认之后优先获取 \`meta/info.json\`，再按真实版本选择 tasks/episodes 文件；下载记录写：
\`modelscope_repo_id, revision, relative_path, bytes, sha256, local_path\`。

本地只读命令（**必须已经有实际文件**）：

\`\`\`bash
cd /root/autodl-tmp/vla_finetuning
PYTHONPATH=src .venvs/data/bin/python -m vla_finetuning.recovery_cli \
  audit-expert --info /root/autodl-tmp/vla_finetuning/datasets/recovery_metadata/PINNED/meta/info.json

PYTHONPATH=src python -m vla_finetuning.recovery_cli \
  audit-scenes --index /root/autodl-tmp/vla_finetuning/datasets/recovery_scenes/index/selection.jsonl
\`\`\`

这些命令没有下载、CUDA、Torch、MuJoCo，也不修改Corrector。示例路径尚未存在时会STOP。
下载之前另行检查 \`nvidia-smi\`、\`vmstat 1 5\`、
\`/proc/pressure/io\`、\`df -h /root/autodl-tmp\`和Corrector的训练吞吐；
不要在运行中的Corrector旁启动GPU任务或无预算的大文件下载。

## 5. Expert→场景关联审计

\`source_group\`不能简单按 \`task_index\`、语言文本、Episode编号构造。
建议先收集来自**原始场景索引/数据生成脚本/可验证轨迹记录**的：
\`source_scene_id, source_rollout_id, source_failure_step, original_task_id, expert_episode_id,
 provenance_evidence, review_status\`。
它们是**候选审核字段**，不是声称ModelScope已经提供。

\`recovery_cli audit-linkage --jsonl ...\`要求人工审核后的真实来源字段，
\`audit-split\`检查全局source-group跨划分泄漏，并要求独立保留官方基准来源。
已有源码不会静默将任务相同的Episode拼成一个源，也不会把元数据存在当作来源已验证。

## 6. 当前可以得出的结论

- 已**代码级核查**公开场景组织和官方评测接口。
- 本仓库已具备 CPU-only 离线审核、冻结划分计划和严格配对统计工具。
- **没有**取得真实 ModelScope Expert meta/data，无法确认其版本、相机/动作字段或训练适配；
  **没有**真实 SmolVLA Recovery Batch、训练Checkpoint或闭环结果。
- P0结论：**NOT READY TO TRAIN**，解除阻塞需由真实文件、逐值转换测试和来源审核提供证据。
