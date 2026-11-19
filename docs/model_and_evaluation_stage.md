# 后续阶段接口：模型准备、训练与LIBERO闭环

> **接口预设，不代表模型已经加载、训练或评测。** 本文件描述阶段B/C的最小验收和源码责任边界，避免一开始复制`robot-lab`全部代码。

## 阶段B：模型与真实Batch（Corrector训练期间只做静态审计）

**已实现，不加载PyTorch**：

```bash
PYTHONPATH=src python -m vla_finetuning inspect-model \
  --model-root /path/to/已经手动取得的SmolVLA快照目录
```

该命令只读取小于1MiB的根目录模型JSON配置，`stat`权重文件尺寸，**绝不反序列化模型权重**，也不验证真实模型/Processor可用。任何显示`ready_to_train=false`都是预期。

**GPU空闲并取得用户同意后才实现/验证**：在与Corrector隔离的LeRobot独立venv中读取固定`lerobot/smolvla_libero`，完整保存配置、Processor、视觉预处理、归一化与参数冻结清单，读取少量**真实**LeRobot示教，检查`observation.images.front/wrist`映射的物理语义、state8维和action7维的含义、Chunk50、padding、Flow Matching真实Loss以及LoRA目标模块。先对照仓库`robot-lab/scripts/smolvla_training_workbench.py`旧已实测代码，不重写训练核心。不能将新的CLI生成器直接认定为已在4090验证。

**你来写的核心函数**：`exercises/student.py::make_action_chunk`，配套`exercises/tests/test_student.py`。通过后再比较官方LeRobot的实际标签窗口与padding逻辑，而不是把教学实现直接替换官方训练器。

**实验数据分组**：先冻结相机来源和源示教组，再比较B0（未新增训练）、L1（Expert LoRA新视角）、L2（Vision/Connector LoRA后续）、R1（Recovery后续）。`smolvla_libero`已使用原始LIBERO训练，重复原始示范不能声称未知任务适配。

## 阶段C：闭环评测准备

**已实现，静态协议校验**：

```bash
PYTHONPATH=src python -m vla_finetuning validate-eval-plan \
  --plan configs/eval.template.json
```

模板全部是虚构Case和示意SHA，返回`PLAN_ONLY_NOT_EXECUTED`并给出预期回合数，**不启动MuJoCo，不加载SmolVLA，不创建视频**。真实评测前必须替换为经过核对的模拟器Revision、相同的物理初始状态Hash、任务语言、视角条件、策略版本和评测随机性。

**GPU空闲后才接入**：复用`robot-lab`原已验证LIBERO-plus仿真入口、H20/H50执行窗口以及Replay；第一批只选择固定任务/条件/seed，并核对动作、相机、FPS、语言和成功判定。保存每回合视频、动作序列、真实状态、推理耗时和有效评测环境配置。Replay导入应是只读，不要复制原始Corrector的实验文件到新仓库或修改它们。

**你来写的核心函数**：`exercises/student.py::paired_outcomes`，对同一`domain/condition_id/seed`下B0/L1的完整结局表报告gain/loss/both，不要只在成功案例中选一个展示提升。

## 工程原则

- 现在2号机S2_s42仍在训练：**不调用GPU、不安装Torch、不下载大数据、不运行仿真**，所有新增远程代码仅保存在GitHub，服务器只由用户主动`git pull`。
- 数据盘真实容量550GiB量级、余约135GiB。不要把“680GB最高可扩容”当成已经分配的容量；HF缓存路径必须落在数据盘。
- 所有学习函数默认是`NotImplementedError`，只在对应练习测试中调用；工程安全模块负责强制约束来源、划分和明确授权。**不把学生习题嵌进未经验证的正式生产训练路径。**
- CI仅在独立GitHub runner执行纯CPU标准库测试；真实PyTorch、Dataset、GPU/仿真检查均明确标记未运行。

下一步真正要解决的难点是：`lerobot/libero_plus`公开数据是否能映射到**具体的视角扰动**和源示教ID；如果不能，先只能研究一般数据分布适配，不能宣称严格未见视角泛化。
