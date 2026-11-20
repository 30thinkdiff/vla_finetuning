# M0 原始SmolVLA × LIBERO-Recovery真实闭环：实施门槛（2026-11-20）

## 本阶段的实际目标
用已核实SHA的`lerobot/smolvla_libero`**未加LoRA的原始模型**，在官方`LIBERO_Recovery_FailureScenes`某一个经核实的真实失败初始状态（`.bddl` + 相邻`_sim_state.npy`）运行**至少一个真正的环境回合**，并记录完整Trace/录像/成功判定。**截至文档创建时该回合尚未运行。**

不把P5单步LoRA Adapter充当M0，不把离线Loss当成功率，也不声称通用训练数据与Recovery Expert来源场景不重叠。

## P6实测结果，决定停止重复统计
真实Expert `2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73` 三条轨迹（Ep0/15/30）：
| Ep | state[6]/state[7] Pearson r | mean abs(state[6]+state[7]) | 轴角模长>π | 最大相邻帧旋转向量L2跳变 |
|---|---:|---:|---:|---:|
| 0 | -0.94418 | 0.003491 | 26/629（4.13%） | 0.04111 |
| 15 | -0.98059 | 0.001023 | 256/796（32.16%） | 0.04676 |
| 30 | -0.88207 | 0.003424 | 147/647（22.72%） | 0.07044 |

两个夹爪状态关节的取值和强负相关**强烈支持**其为`robot0_gripper_qpos`两个实际关节，而非第一维为全零pad；`meta/modality.json`字段`pad`不能被当作底层Parquet state[6]常量0。官方评测函数`_quat2axisangle`没有规范化四元数标量分量非负，轴角模长大于π在实现上允许。三条轨迹未观察到明显的相邻帧轴角向量突跳，但不代表所有轨迹/物理角速度已经验证。关于`action[6]`的0/1语义与坐标系，还须在实际控制器/视频中核实。

## 官方接口事实（固定 liulin815/LIBERO-Recovery@832d54dfc43f0c9e4240bb8d92f5fda7001bb093）
- `examples/LIBERO/eval_files/eval_libero_custom_scene.py`从`agentview_image`及`robot0_eye_in_hand_image`取两路图，按`[::-1,::-1]`转180°，并组装`eef_pos(3)+quat2axisangle(3)+gripper_qpos(2)`8维状态。
- 评测端预测格式为`world_vector(3)+rotation_delta(3)+open_gripper(1)`，其中0/1映射到仿真器+1关闭/-1张开。
- 官方`model2libero_interface.ModelClient`是**StarVLA**专用，向服务端仅发送图像、任务文本（没有8维状态），因此**不能直接传入SmolVLA模型路径使用**。
- SmolVLA本地`Policy.select_action`包含自身的动作队列行为，M0桥接需要逐环境步提供最新图像、8维状态及任务，使用**原预训练的preprocessor与postprocessor**再由模拟器执行一个反归一化后的动作；不应手动重复pop多个动作造成跳步。
- 官方Bash启动脚本包含作者设备专属`/H20_vepfs/...`绝对路径和StarVLA服务器，**不能直接在AutoDL执行**。

## 环境隔离与已知陷阱
- AutoDL目前`.venvs/train`是**Python3.12+固定LeRobot0.6.1**，训练前向/Backward和Adapter存取已真实通过。
- 官方LIBERO-Recovery建议仿真侧单独Python3.10/NumPy1.24.4及MuJoCo；应使用**独立**`.venvs/recovery_sim`，不要把Python3.10包装进`.venvs/train`。
- 仓库已有`scripts/bootstrap_sim_env.sh`目标是**LIBERO-plus（不是官方Recovery）**，且当前默认从Python3.10创建虚拟环境同时安装需Python>=3.12的LeRobot0.6.1，**存在明确版本冲突**。在修复/替代之前不要运行该脚本的`--apply`。
- 两进程桥接先做本机回环IP/受限端口、明确手动启动与退出，无云端服务无外网暴露；服务器与评测器都需要回合开始时重置策略内部动作队列/任务缓存。
- 若需要初始失败场景，先下载`ataier/LIBERO_Recovery_FailureScenes`固定版的官方index与单对`.bddl`+`_sim_state.npy`，不要下载80GB`Recovery_Rollouts`。

## 接下来的实际阶段
1. **资产清单与单场景选择（只用CPU/小网络）**：通过仓库`scripts/recovery_hub.py inventory`收集固定`FailureScenes` Revision与大小；核实`index/selection.jsonl`中的单一测试scene和实际路径、级别、来源。确保不会和未来训练数据来源group重合。
2. **隔离的官方Recovery仿真初始化**：锁定官方源码、Python3.10/MuJoCo/robosuite/模型资产，单独新目录运行纯环境reset+一次合法保守控制；检查两路图方向、state8/动作7、种子/成功判定，录制Trace。此阶段不调用SmolVLA。
3. **真正的M0策略桥接**：使用`.venvs/train`加载原预训练Policy+pre/post、从仿真进程交付图像/状态/文本；严格区分归一化动作（模型侧）和仿真器控制动作（环境侧），gripper阈值0.5并转±1。首测一个场景的一个回合，设置总步数上限，出错终止、录像。
4. **M0冻结评测协议**：固定场景subset、初始化/扰动/夹爪状态/seed/动作执行窗口与统计口径；运行M0两种初始抓手状态并生成可复核report与trace。随后M1才使用同一测试协议。

## 可立即进行的只读准备（不下载场景、不安装模拟器）
```bash
cd /root/autodl-tmp/vla_finetuning
git status -sb
df -h /root/autodl-tmp
nvidia-smi
python scripts/recovery_hub.py --help
find datasets -maxdepth 3 -type d | head -n 50
```

下一步**先生成FailureScenes inventory，再决定最小下载量**。不得把本计划当作已实现的M0模拟器或真实回合。


## 2026-11-20：FailureScenes真实清单已获取，接下来只下载1组M0场景

AutoDL用户已在独立数据环境\`.venvs/data\`安装\`modelscope-hub==0.4.5\`，并实际运行\`scripts/recovery_hub.py inventory\`，得到固定修订\`95acf98dd41a4965018069e7e4f27aac1fc69328\`、15,864文件、0.054 GiB清单。只是远端列表，**尚未下载场景/运行模拟器或M0**。

新增纯标准库\`scripts/prepare_recovery_m0_scene.py\`，从清单里按suite和seed严格选一个完整的\`.bddl\`+\`_sim_state.npy\`配对，若\`index/selection.jsonl\`存在则一起纳入；只生成一个精确下载manifest。先用该小数据完成M0打通，**不能将这一个随机工程示例作为科学评测**。

\`\`\`bash
cd /root/autodl-tmp/vla_finetuning
git status -sb
git fetch origin
git merge --ff-only origin/feature/recovery-p0-offline-foundation-20261119

SCENE_REV=95acf98dd41a4965018069e7e4f27aac1fc69328
INV="runs/recovery_failure_scenes_inventory_$SCENE_REV.json"
MANIFEST="runs/recovery_m0_object_scene_$SCENE_REV.json"
ROOT="$PWD/datasets/recovery_failure_scenes/$SCENE_REV"

python scripts/prepare_recovery_m0_scene.py \
  --inventory "$INV" \
  --suite libero_object \
  --seed 7 \
  --out "$MANIFEST"

# 查看确切的文件条目，确认只有一个.bddl、一个_sim_state.npy及可选索引
.venvs/data/bin/python scripts/recovery_hub.py download \
  --manifest "$MANIFEST" --target "$ROOT" --max-gib 0.1

# 显式同意下载时再执行：不需要也不允许下载80GB Rollouts
PYTHONPATH=src .venvs/data/bin/python scripts/recovery_hub.py download \
  --manifest "$MANIFEST" --target "$ROOT" --max-gib 0.1 \
  --apply --receipt-id m0_object_s7
\`\`\`

注意：下载器为防IO影响其他GPU训练，仍有GPU空闲检查。上述\`download\`的大小预算为远端目录估计，不是底层流式字节上限。根据目录只有约0.054 GiB整体总量，一组文件的网络成本很小。

下载后用户下一步工作是**实际建立隔离的Recovery Python3.10模拟器**及策略桥接，不能重用有Python冲突的\`bootstrap_sim_env.sh\`。优先实现单场景真实\`env.reset\`/\`set_init_state\`、一次\`env.step\`，然后接通M0，而不是继续重复Expert小样本统计。
