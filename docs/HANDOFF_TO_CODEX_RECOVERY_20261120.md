# Codex 接手简报（2026-11-20，依据用户终端历史）
此处内容是截至最后用户日志的已知状态，不是最新服务器实时检查。更完整的交接Markdown已在当前ChatGPT会话生成：CODEX_HANDOFF_RECOVERY_20261120.md。Codex应首先检查服务器当前状态。

## 任务与代码
目标：对比M0原始SmolVLA、M1 Recovery Expert-only LoRA、M2混合正常LIBERO示教LoRA在Recovery失败场景下成功率与普通任务保留能力。先完成真实M0仿真，继而M1正式训练；M3机制创新需后置，避免凭空发明结果。

仓库 https://github.com/30thinkdiff/vla_finetuning
分支 feature/recovery-p0-offline-foundation-20261119
本地AutoDL工作目录 /root/autodl-tmp/vla_finetuning，本地分支 recovery-p0，远端分支本次交接前HEAD=58e56fd9f65dd99978b89c63cadc01d7c4bcc0b5（新增本文件后会变化）。
训练环境 .venvs/train/bin/python（Python3.12，LeRobot0.6.1，torch2.11+cu130，已真实RTX4090验证）。
数据环境 .venvs/data/bin/python（Python3.10，modelscope-hub==0.4.5已真机安装）。
目标仿真环境 .venvs/recovery_sim/bin/python（用户最后检查尚不存在；不能当作已经安装）。
官方Recovery源码 liulin815/LIBERO-Recovery@832d54dfc43f0c9e4240bb8d92f5fda7001bb093。
原SmolVLA lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d；路径 models/smolvla_libero_contract/<SHA>。SmolVLM2 runtime/tokenizer@4cde9d208e19519c36917aa170fffc7e381aa0ad。

## 已经在AutoDL真实跑通
1. ModelScope ataier/LIBERO_Recovery_Expert@2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73：数据集v2.1、20Hz、3184 Episodes/625731帧、110任务、state8/action7、双256x256相机。只下载并校验了Ep0(629帧)、Ep15(796帧)、Ep30(647帧)，共2072帧；未下载正式训练集。
2. 真实Batch双相机 [1,3,256,256]、状态[1,8]、动作[1,50,7]、语言token[1,48]，模型原权重SHA256=9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8。
3. P5真实训练链路：scripts/smoke_recovery_optimizer_reload.py结果 REAL_SMOLVLA_LORA_ONE_OPTIMIZER_STEP_SAVE_RELOAD_PASS。LoRA rank8 alpha16、371328可训练参数、74梯度张量（37非零）；一次AdamW(lr1e-4)更新37个张量/188855个元素，保存仅Adapter，从原始权重重建+重载逐张量最大差值0.0。pre-update单样本随机loss=9.85724不可当性能结果。runs/recovery_batch/ep0_f10_one_step_adapter只作测试，不是正式M1。
4. P6统计：原Checkpoint统计下state p95|z|=[1.213,1.556,1.234,4.290,2.153,3.199,1.828,1.705]；action=[0.947,1.496,1.390,5.647,2.555,1.740,1.051]。state[6]标签虽然pad却是实际非零、与state[7]强负相关（-0.94418/-0.98059/-0.88207），支持两个夹爪关节，不应清零。轴角norm>pi可能符合官方未进行w>=0规范的实现；尚未独立证实Expert示教真实控制源的全部语义，不应凭3条示教重算Normalizer或擅自再旋转视频。
5. FailureScenes真实场景已下载：ataier/LIBERO_Recovery_FailureScenes@95acf98dd41a4965018069e7e4f27aac1fc69328。已验证 index/selection.jsonl 与 scenes/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket/episode_018/step_0080.bddl + step_0080_sim_state.npy，全部SHA VERIFIED；receipt在 datasets/recovery_failure_scenes/<SHA>/.vla_recovery_verified_m0_object_s7.json；manifest在 runs/recovery_m0_object_scene_<SHA>.json。不要重复下载。这是工程用单场景，不是正式测试集。

## 当前阻塞，最后已知服务器状态
尝试bootstrap_recovery_sim_env.sh安装官方Recovery代码：第一次GitHub HTTPS GnuTLS recv error(-110)；后续partial clone+sparse checkout仍慢，约20–27KiB/s。用户检查进程看到 git checkout --detach 832d54d... 已运行66分钟，Git lazy fetch --filter=blob:none子进程继续下载；.git约207MB、objects/pack223MB、LIBERO目录38MB；git rev-parse HEAD无输出，说明未完成checkout。是否已经中止未知。sim venv此前不存在。不要贸然覆盖运行中的源目录，优先查 ps 和 git 状态。

用户上次unzip提示 missing recovery-m0-sparse-832d54d.zip，原因是ZIP未被上传AutoDL，后续SHA/TAR失败均是连锁错误，不是ZIP损坏。

## 有可靠替代路径，避免在AutoDL慢速Git下载
GitHub Actions已真正成功打包官方Commit的稀疏精简源码：
https://github.com/30thinkdiff/vla_finetuning/actions/runs/44716466749
artifact= recovery-m0-sparse-832d54d，ID=14619737222，约98MiB ZIP。ChatGPT容器已验证完整ZIP：
ZIP SHA256=28b51a00e79ed15db3d2647bfccea34704e314fcab0adcc9c2fdd8e8a875f085
内部TAR SHA256=8b43d2b550e9772180cd7f75cca5234f26d727ae58b208e582b3ed390caff381。
但ChatGPT的/mnt/data不是AutoDL目录。请通过GitHub Actions下载或用户浏览器下载后JupyterLab/SCP上传至 /root/autodl-tmp/vla_finetuning/recovery-m0-sparse-832d54d.zip，真正看到ls显示存在才运行 scripts/install_recovery_from_bundle.sh --apply。此脚本会校验ZIP/TAR/Commit并备份旧Git源码。随后用 VLA_APPROVED_CPU_IO=YES bash scripts/bootstrap_recovery_sim_env.sh --apply 初始化独立模拟器。均尚未在AutoDL真机验证，精简包可能漏少数当前MuJoCo依赖资产，按缺失路径定向补齐。

## 后续真实交付优先级
A. 先解决源码包在AutoDL缺失问题，安全停止旧Git慢速checkout、部署官方Recovery隔离环境；不触碰已有训练/数据/Corrector环境。
B. 用scripts/bringup_recovery_sim.py加载verified单场景、set_init_state并执行一个真实env.step，必须真实拿到 REAL_RECOVERY_SIM_RESET_AND_ONE_SAFE_STEP_PASS_NO_POLICY；当前只是脚本和CPU CI通过。
C. 完整实现SmolVLA与Recovery模拟器的两进程桥接（Python3.12策略↔Python3.10仿真），从两路相机、EEF8和task提取观测，使用原preprocessor、策略select_action队列和原postprocessor反归一化后再映射7维控制动作；官方model2libero_interface.py是StarVLA客户端，不能直接作为SmolVLA入口。动作Postprocessor下载脚本 scripts/fetch_smolvla_postprocessor_assets.py 已提交，但用户未回传是否真机下载成功。一个完整M0 rollout需保存trace、录像和成功判定。
D. 冻结来源无泄漏划分和Expert训练集，开发真DataLoader/可复现M1多步LoRA（不是单步测试Adapter）；复用相同M0评测协议，测试正常LIBERO性能。M2混合数据尚未实现，当前LeRobot仅接受一个dataset.repo_id，直接传列表无效。M3以后再决定。

先读：docs/recovery_integration_gap.md（包含历史记录，不能都当作最新）、docs/recovery_control_semantics_audit_20261120.md、docs/recovery_m0_execution_plan_20261120.md、scripts/install_recovery_from_bundle.sh、scripts/bootstrap_recovery_sim_env.sh、scripts/bringup_recovery_sim.py、scripts/fetch_smolvla_postprocessor_assets.py、scripts/smoke_recovery_optimizer_reload.py。

**严禁**把CPU CI当作模拟器/模型真机PASS；不要重复P0–P6已通过的烟雾/统计；不要改动 /root/autodl-tmp/corrector；不要自动删除原始数据、权重、Adapter或仍在下载的Git目录。优先交付可复现真实M0与M1结果。
