# Recovery Expert × SmolVLA 控制语义审查（2026-11-20）

## 证据和当前结论

**本文件记录已核对的源码事实，不是训练集语义已经被证明的结论。**

固定对照代码：
- 官方恢复评测仓库：`liulin815/LIBERO-Recovery` commit `832d54dfc43f0c9e4240bb8d92f5fda7001bb093`，来自2026-10-22 v1.0.0；原文件 `examples/LIBERO/eval_files/eval_libero_custom_scene.py`。
- 同仓库 `examples/LIBERO/eval_files/model2libero_interface.py`，以及 `LIBERO/scripts/collect_demonstration.py`。
- 本仓库已验证 Expert 数据：`ataier/LIBERO_Recovery_Expert` revision `2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73`（3个Episode 0/15/30实测），SmolVLA策略 `lerobot/smolvla_libero` revision `a24f1f380461b665158d52acd140152c79e8b36d`。

## 官方评测端直接确认的契约

| 项 | 官方源码位置 | 源码已确认 | Expert数据集是否已独立确认 |
|---|---|---|---|
| 观测图像 | `eval_libero_custom_scene.py:384-385` | 从`agentview_image`和`robot0_eye_in_hand_image`取图并以`[::-1, ::-1]`旋转180° | **未确认**真实Expert MP4是否已完成相同旋转，禁止重复旋转 |
| 状态 | `eval_libero_custom_scene.py:389-395` | `[robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3), robot0_gripper_qpos(2)]`，合计8维 | **未确认**Expert v2.1 `observation.state`的逐维生成来源 |
| 模型回传动作 | `eval_libero_custom_scene.py:401-416`、`model2libero_interface.py:149-155` | `world_vector(3)`、`rotation_delta(3)`、`open_gripper(1)` | Expert action `[7]`形状已确认；动作生成方式和尺度未独立确认 |
| 夹爪送入模拟器 | `eval_libero_custom_scene.py:44-53, 413-416` | Open值大于0.5时映射为环境-1（张开），否则+1（闭合） | 尚未证明Expert action[6]编码与其同义 |
| 评测落盘数据集 | `eval_libero_custom_scene.py:484-492` | 环境夹爪+1闭合、-1张开转为数据集0闭合、1张开 | **注意这里落盘的是评测回合记录，不是Expert数据仓库的构建代码** |
| 仿真默认控制器 | `LIBERO/scripts/collect_demonstration.py:240-245, 274-281` | 遥操作脚本默认`OSC_POSE`，动作来自`input2action` | 不代表全部Expert数据均来自该脚本或遵循默认配置 |

**严禁由维度相等推出控制语义一致。** 尤其不能仅根据第4个维度的数值差异断言旋转坐标系错误——上述模型回传格式提示此维度可能对应第一轴旋转增量，但Expert的真实转换代码尚未独立追踪。

## P6 真实统计结果与可检验问题

用户在AutoDL实际运行`scripts/audit_recovery_normalization_distribution.py`（2026-11-20）：
- Episode 0/15/30，共2,072帧。
- `state`按原Checkpoint `MEAN_STD` 标准化后各维p95|z|：
  `[1.213,1.556,1.234,4.290,2.153,3.199,1.828,1.705]`。
- `action`各维p95|z|：
  `[0.947,1.496,1.390,5.647,2.555,1.740,1.051]`。
- state D4（数组index3）、D6（index5）、action D4（index3）超出参考值3。**p95|z|>3并非模型禁止输入，更不代表95%的数据越界。**
- 仅有同一小任务族的三个示教Episode；不存在已验证的全数据集统计/正式训练统计策略。

后续逐维必须核查：
1. checkpoint的`mean`、`std`与Expert Pilot均值、标准差（全报告中已有），特别是D4；判断偏移主要来自均值还是尺度。
2. `per_episode`里的D4 p95和`fraction_abs_z_gt_3`：是全部Episode普遍出现，还是少数操纵动作造成。
3. 原始Expert动作逐帧与状态、视频对应关系；针对旋转坐标及gripper，需找到正式Expert构建/转换脚本并与官方评测边界逐值对齐。
4. 图像方向：检查**最终Expert MP4**是原相机方向还是已预旋转，而非见到`[::-1,::-1]`就对训练图片再次旋转。
5. 若考虑重算训练统计：必须先冻结无泄漏train split，保存独立训练/推理反归一化统计，并让原基线M0保留原预训练统计。禁止从评测/测试数据计算训练统计、禁止用试验中的Pilot3直接代替全train数据。

## 对M0/M1的硬门槛

- M0：严格固定原预训练政策、预训练pre/postprocessor和相机协议；在官方FailureScenes真实仿真完成至少1次闭环、保存观测/动作trace和录像；不得用单步LoRA Adapter当M0。
- M1：控制协议一致性证据、无泄漏split、训练全样本来源审核、正常/Recovery评估都具备后，才进入正式LoRA训练。
- 当前完成的是P5单步真实更新/Adapter保存重载，以及P6小样本统计异常定位。**没有任何关于Recovery成功率或泛化增益的证据。**

源码本地交叉核查顺序：先固定以上官方仓库SHA下载源文件并检查行号；其次阅读Expert数据发布对应的采集/转换实现（若缺失，标记UNKNOWN并请求发布者澄清）；最后做视频帧、控制动作和状态差分的三方交叉验证。
