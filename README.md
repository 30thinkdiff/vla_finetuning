# VLA Fine-tuning Research — SmolVLA

**状态：2026-11-19 · 已提供完整的预备阶段工程实现与安全门槛；尚未在AutoDL联调真实数据、完整SmolVLA训练或LIBERO-plus闭环。**

本仓库新增了**正式训练/评测编排、独立环境安装、按来源Episode选择v3实际Parquet/视频分片、Pinned模型/数据下载预算、官方Checkpoint续训计划、官方`eval_info.json`到配对CSV、Recovery子集获取脚本与LIBERO-plus资产安全检查**。参见[完整工程使用手册](docs/ENGINEERING_GUIDE.md)。这些代码已用于静态CI测试，但所有涉及依赖或外部资源的操作需要经过真实环境联调才能宣称已运行成功。

这是对已有 [robot-lab](https://github.com/30thinkdiff/robot-lab/tree/feature/robotlab-replay) 的SmolVLA、Corrector研究的独立后续：优先学习和实现**训练数据审计、Flow Matching、LoRA、Checkpoint与闭环实验协议**；研究主线是 **LIBERO-plus视觉适配**，第二阶段再接入**LIBERO-Recovery失败恢复**。

## 已经具备的能力与边界

- 标准库即可执行的本地LeRobot数据`meta/info.json`形状审计、来源分组切分、只读资源快照；
- 可复算的两数字Flow Matching演示，以及可选的Torch版Masked Velocity MSE、梯度/参数审计；
- 基于已有[SmolVLA训练工作台](https://github.com/30thinkdiff/robot-lab/blob/feature/robotlab-replay/scripts/smolvla_training_workbench.py)里验证过的CLI参数，**生成但不自动执行**官方LeRobot LoRA训练命令；额外提供只可人工触发、有空闲GPU检查的启动入口；
- CPU-only单元测试和GitHub Actions，不下载模型/数据，不依赖Corrector环境。

**尚未完成**：真实Plus样本与扰动元数据逐条审计、真实训练Batch、正式LoRA微调、Vision/Connector LoRA、Recovery数据适配、完整Checkpoint续接、配对闭环。不能以脚本存在冒充实验完成。

## 2号机安全原则

用户2026-11-19回传：RTX4090显存已由Corrector`S2_s42`训练占约9.6GB，GPU Util 77%；数据盘实际550GB/已用415GB/余136GB，系统盘剩29GB，内存足够。**不要与Corrector并行启动新GPU任务**；只在这个独立仓库准备代码、CPU测试、少量元数据。

所有数据、HF缓存、权重、Run及视频放`/root/autodl-tmp/vla_finetuning`内的被Git忽略路径。特别注意当前系统盘有29GB剩余，**不要把HF缓存默认写到根系统盘**。将来手动下载前设置：

```bash
export HF_HOME=/root/autodl-tmp/vla_finetuning/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/vla_finetuning/.cache/huggingface/datasets
```

上述命令本身不会下载任何文件。不要在此仓库复制原Corrector训练目录、旧Checkpoint或SSH私钥。

## 新增：数据获取与亲手实现练习

**请先阅读[阶段A：数据获取、v2/v3元数据和代码练习](docs/data_stage_walkthrough.md)。**

- 已实现`inspect-data-layout`：只读本地`meta/info.json`及任务/Episode索引结构。
- 已实现`hf-pin`及`fetch-metadata`：必须明确`--allow-network`，下载**仅限指定的小元数据文件**（默认单文件上限512KiB），不接触模型或训练视频。
- `exercises/student.py`的三道题专门留给你写：任务身份解析、Episode内Action Chunk/Mask、配对闭环成败统计；测试在`exercises/tests`，未完成时失败属于预期状态，独立于正式CI。
- `lora-demo`需要可选PyTorch；当前系统Python缺Torch不影响其它功能，**不为此修改Corrector的venv**。

## 已补齐的工程模块与参考实现

- `src/vla_finetuning/pipeline.py`：统一`validate-config/shard-plan/hub-plan/hub-download/train/evaluate/resume-plan/report/merge-eval`入口。**默认只打印计划**。
- `src/vla_finetuning/dataset_files.py` + `artifacts.py`：从v3原Episode索引筛选真实Shard、检查Hub文件大小与Revision，限额、单线程、显式`--apply`下载。
- `src/vla_finetuning/engine.py` + `real_batch.py`：官方LeRobot CLI、真实CPU样本与Checkpoint安全检查；实际模型前向仍待独立环境验证。
- `src/vla_finetuning/reference.py`：三个学生练习函数的完整工程参考实现，附对应单测。
- `src/vla_finetuning/reports.py` + `eval_extract.py`：读取真实官方评测结果并转为严格配对CSV，统计救回/退化和原始能力表现。
- `scripts/bootstrap_train_env.sh`、`bootstrap_sim_env.sh`、`download_recovery_subset.py`、`prepare_plus_assets.py`：资源保护的安装/获取/资源审计脚本，默认不下载、不执行。
- [工程使用手册](docs/ENGINEERING_GUIDE.md)详细说明各阶段CLI、输入输出及尚未通过真实GPU/仿真验证的边界。

## 立刻可运行（不会调用GPU）

在2号机原项目根目录：

```bash
cd /root/autodl-tmp/vla_finetuning
git pull --ff-only origin main

# 直接用标准库和本仓库源码，无需安装依赖
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m vla_finetuning preflight --data-root /root/autodl-tmp
PYTHONPATH=src python -m vla_finetuning flow-demo
# 以下学生练习测试会在你填完TODO前失败，这是预期行为：
# PYTHONPATH=src python -m unittest discover -s exercises/tests -v
# 若当前独立Python环境已经安装torch，可额外练习CPU LoRA两步梯度:
PYTHONPATH=src python -m vla_finetuning lora-demo
PYTHONPATH=src python -m vla_finetuning split --csv examples/provenance.synthetic.csv --out runs/example_split.json
PYTHONPATH=src python -m vla_finetuning plan-train --config configs/experiment.template.json
```

`plan-train`对未填写的40位模型/数据SHA以及空Episode清单必须返回`BLOCKED`，这才是预期的安全结果。运行`split`得到的演示数据纯属虚构，不能成为正式训练集。

如你已经**手动取得**少量真实数据集元信息（不必下载整套视频），可执行：

```bash
PYTHONPATH=src python -m vla_finetuning audit-metadata --info /path/to/meta/info.json
```

程序会给出相机键、状态/动作维度、帧率、轨迹数、元文件SHA及数据来源尚未验证的风险，**不会自动联网下载**。

## 参考来源与准确性边界

1. 旧项目已经实测过`lerobot/smolvla_base`上的5步LoRA、0.371M可训练参数、真实梯度与重载；详见[2026-11-05记录](https://github.com/30thinkdiff/robot-lab/blob/feature/robotlab-replay/studies/policy_generalization/training_results_20261105.md)。**不当作新项目新训练成果**。
2. LeRobot源码版本沿用旧工作台的`7e241bd630a3719a56157a497ce5d08f244784f1`，GPU实际运行前须在独立venv核查导入与真实CLI参数；不要装进Corrector的`runtime/venv`。
3. 主要数据候选：[lerobot/libero_plus](https://huggingface.co/datasets/lerobot/libero_plus)；备选[ Sylvest/libero_plus_lerobot](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot)；恢复数据[LIBERO-Recovery](https://github.com/liulin815/LIBERO-Recovery)。视觉扰动来源映射/训练数据重复未经现场核实前禁止得出泛化结论。

## 建议阅读顺序

先读[数据来源和防泄漏](docs/data_provenance_and_split.md)，再读[研究协议](docs/research_protocol.md)，最后跟着[Flow Matching/LoRA训练习题](docs/learn_by_doing.md)运行小实验。第一版训练模板见`configs/experiment.template.json`。

## 阶段化交付：工程代码与你的实现边界

- [数据获取与分类实战](docs/data_stage_walkthrough.md)：元数据只读审计、显式限额网络获取、v3任务索引。
- [训练及闭环阶段接口](docs/model_and_evaluation_stage.md)：只读模型文件检查、配对评测协议检查，不加载模型与仿真。
- [你亲手实现的三个核心函数](exercises/student.py)及[验收测试](exercises/tests/test_student.py)：任务身份、动作Chunk/Padding、同条件评测。练习测试初次失败是预期行为，不属于正式CI测试失败。
- `plan-train`在真实来源证据尚未人工审核前继续`BLOCKED`。虚构`synthetic`划分永远不能作为正式训练数据证明。

## 下一阶段进入条件

1. 审计可追溯来源的真实Plus小子集，检查相机映射、控制量、FPS、数据归一化与Episode划分；
2. 手动下载并校验固定Revision的`smolvla_libero`模型快照、数据与独立LeRobot训练环境；
3. 2号机Corrector停止并完成状态归档后，确认空闲GPU，先启动一个真实Batch/50步Smoke；检查参数冻结、Loss、Checkpoint/处理器与数据校验；
4. 通过后再开展B0对L1的同条件闭环。不要提前以MSE下降写求职简历。

**绝对禁止将此项目的训练命令写进Corrector的队列。**
