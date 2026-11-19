# 数据来源、重叠与划分协议（先于训练）

> 状态：研究设计与审计工具已提交；**尚未下载并逐轨迹验证真实训练数据**。任何成功率改进均不得从此文推断。

## 初始策略与数据重叠

- `lerobot/smolvla_base`：用于从通用预训练策略适配LIBERO的独立学习实验。
- `lerobot/smolvla_libero`：此前已经通过`lerobot/libero`数据微调。沿用原始LIBERO示范只能用于流程复核、能力保持或再训练研究，**不能**称为未见任务适配。
- `HuggingFaceVLA/libero`与`lerobot/libero`可能是同一批示范的不同存储形式。原始Checkpoint是否曾接触某个测试Episode，无法通过本次随机拆分消除。
- 不要把测试Episode/训练Episode无交叉与**基础策略从未见过任务**混为一谈。

## 三种后续数据

| 候选 | 核实前限制 | 第一阶段用途 |
|---|---|---|
| `lerobot/libero_plus`（LeRobot v3.0，公开元数据约14347回合） | `task_index`不等于扰动`task_id`；来源相机视角、源示范、scene mapping未审计 | 少量示范的视角适配 |
| `Sylvest/libero_plus_lerobot`（v2.1） | 不应混合两个版本的同源样本，也不能单凭总量相同断言数据逐帧相等 | 兼容性备选 |
| `ataier/LIBERO_Recovery_Expert`（ModelScope） | 真实状态/动作形状和source failure scene映射需现场验证 | 之后研究L1重试/L2动作调整 |

数据源审计需要同时记录：HF/ModelScope snapshot SHA、每Episode的原始来源ID、任务指令、相机内外参/扰动参数、初始仿真状态指纹、帧率、两路相机键、状态和动作含义、是否成功以及原始资产修订。没有来源ID就先停下来调查，不造`source_group`。

## 防止泄漏的基本规则

1. 对同一原始示教在不同视角重渲染的数据：`source_group = original_demonstration_id`，所有视角必须处于同一split。
2. 对同一失败场景的多条Recovery专家示教：`source_group = original_failure_scene_id`，所有接管版本必须处于同一split。
3. 训练/验证/测试三者`source_group`无交集；测试视角或类别保留协议是**另一层**隔离，需要人工核对真实相机映射。
4. Validation可以用于选Checkpoint，Test和保留视角不能调参；模型的原始LIBERO训练历史必须单独披露。
5. 10Hz示教、20Hz在线控制不能凭`[50,7]`动作块形状视为等价；必须核对动作的时间单位、归一化、7维控制量和重采样协议。

工具只读取本地`meta/info.json`与人工审定的CSV，默认无网络，绝不下载整套数据。先运行：

```bash
python -m vla_finetuning audit-metadata --info /path/to/meta/info.json
python -m vla_finetuning split --csv examples/provenance.synthetic.csv --out runs/example_split.json
```

示例CSV**完全是虚构数据，仅用于测试**。真实实验须另建经过逐项审计的来源表；输出首次创建后不覆写。

## 必须完成的人工验收

- 是否能从数据集每条轨迹追溯到原始示范、相机位姿或失败源状态？
- `front/wrist`与原模型`agentview/robot0_eye_in_hand`是否仅仅键名不同，还是成像/坐标系也不同？
- 如果真实样本`observation.state`是8维、`action`是7维，它们每个位置的物理量到底是什么，幅度/归一化如何？
- 能否用前后两帧和动作验证控制时间间隔及padding不跨Episode？
- 如果数据源无法提供扰动来源，就只能做**整体域继续微调**，不能声称真正的“未见视角泛化”。
