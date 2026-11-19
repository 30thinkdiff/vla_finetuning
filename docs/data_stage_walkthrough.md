# 阶段A：LIBERO-Plus 数据获取、分类与来源审计

**当前状态：仅框架。未在2号机执行任何命令、下载或安装。**

## 先理解数据格式

公开 `lerobot/libero_plus` 的 `meta/info.json` 声明：v3.0、14347个Episode、2238036帧、40任务、20FPS、front/wrist双相机、state8、action7。这只是**元数据声明**；真实动作语义、相机位姿、源示教与分布仍未审查。

| 版本 | 任务元数据 | Episode索引 |
|---|---|---|
| LeRobot v2.1 | `meta/tasks.jsonl` | `meta/episodes.jsonl` |
| LeRobot v3.0 | **`meta/tasks.parquet`** | **`meta/episodes/**/*.parquet`** |

v3.0的多条Episode会共享一个数据Parquet或MP4文件；不能通过文件名猜测Episode数量、边界或来源。

## 命令1：只读检查本地元数据

如果你已经有本地 `meta/info.json`：

```bash
cd /root/autodl-tmp/vla_finetuning
PYTHONPATH=src python -m vla_finetuning inspect-data-layout --dataset-root datasets/metadata/你的数据目录
```

此命令只读取 `info.json` 并列出`tasks`及`episodes`索引存在情况。默认`--sample-rows=0`，**不会读取Parquet内容，也不会安装PyArrow或使用Torch/GPU**。

## 命令2：可选的轻量联网获取（执行前你先审计资源）

查看发布数据的当前revision，**只读取Hub API JSON**：

```bash
PYTHONPATH=src python -m vla_finetuning hf-pin \
  --repo lerobot/libero_plus --allow-network
```

复制输出的40位SHA：

```bash
REV=将刚才的40位SHA放这里
PYTHONPATH=src python -m vla_finetuning fetch-metadata \
  --repo lerobot/libero_plus \
  --revision "$REV" --file meta/info.json \
  --output-root datasets/metadata --allow-network

PYTHONPATH=src python -m vla_finetuning fetch-metadata \
  --repo lerobot/libero_plus \
  --revision "$REV" --file meta/tasks.parquet \
  --output-root datasets/metadata --allow-network
```

每条`fetch-metadata`只请求一份文件，默认限制512KiB，并拒绝覆盖。**不请求Episode、训练Parquet、视频或权重。**如果网络不可达、Hub改变返回格式、超过限额，应停止并保留信息；不得自动改为无限下载。

然后：

```bash
ROOT="datasets/metadata/lerobot--libero_plus/$REV"
PYTHONPATH=src python -m vla_finetuning inspect-data-layout --dataset-root "$ROOT"
```

若后续独立Python环境中已有PyArrow，设置`--sample-rows 3`只读取第一份**Episode元数据**Parquet最多三行；未下载`meta/episodes`时会说明缺失。无需为这个阶段安装PyTorch。

## 核心练习：只修改 exercises/student.py

运行：

```bash
PYTHONPATH=src python -m unittest discover -s exercises/tests -v
```

起初`NotImplementedError`是**预期状态**：三个函数由你亲手实现，正式CI不运行这组未完成习题。

1. **任务身份提取** `extract_task_labels(record)`：支持`tasks:["..."]`的文本任务，以及旧`task_index:3`字段。**不能把训练数据集任务索引3当成LIBERO-Plus仿真环境任务ID3。**
2. **动作Chunk与Padding** `make_action_chunk(actions,start,horizon)`：只读取同一条Episode的动作，生成`[H,7]`数据和`is_pad[H]`布尔掩码。尾部不足`H`复制最后有效动作，但标记`True`以便损失排除。
3. **配对闭环结果** `paired_outcomes(rows,B0,L1)`：相同`domain/condition_id/seed`下统计救回、恶化、都成功与都失败。严格拒绝缺对照或重复结果；不能简单用不同样本平均成功率进行因果归因。

三个任务对应面试中真实的Dataset、模型训练和评测能力。按顺序只做第一题，不需要一次完成全部。不要通过删除断言或改测试数据来“通过”。

## 数据来源不能自动猜测

`task_index`和任务描述可帮助统计任务，但`original_demo_id`、`camera_view_id`、`perturbation_type`、`failure_scene_id`等信息必须从源数据/仿真资产追溯。没有证据的字段标记为`PENDING`，不要生成伪造`source_group`，也不要拿同一个任务的随机Episode划分当作“未见相机泛化”。

这与已存在的`split`和`plan-train`安全门槛配合：先审核来源、冻结训练/验证/测试组，再进入模型阶段。

## 阶段验收

- 能解释v2/v3元数据索引差异，以及`task_index`与LIBERO-Plus仿真任务ID不相同的原因；
- 能运行只读`inspect-data-layout`，读出`info.json`的FPS、输入/动作形状与缺失元数据；
- 自己实现第一道任务解析练习，并通过对应单测；
- 能说清**哪些字段有确切来源证据、哪些仍然未知**。

阶段B将审核SmolVLA权重/Processor/真实Batch/LoRA冻结与Flow Matching，再在GPU释放且用户许可后做真实训练。阶段C将复用RobotLab已有的LIBERO-plus/Replay代码，建立配对闭环与来源可复算结果，不从头复制一套仿真器。

参考：[官方数据格式](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)、[`lerobot/libero_plus`元数据](https://huggingface.co/datasets/lerobot/libero_plus/blob/main/meta/info.json)。
