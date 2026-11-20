## 2026-11-20 ModelScope只列出master的处理方式

ModelScope `HubApi().list_repo_revisions(dataset, "dataset")` 只列出branch/tag，可能对Expert和FailureScenes都仅返回`master`。**不能把`REV=master`直接交给已启用防漂移检查的`recovery_hub.py inventory`**。

优先使用远端Git引用只读解析真实HEAD commit（不会下载大文件）：

```bash
git ls-remote https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_Expert.git refs/heads/master
git ls-remote https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_FailureScenes.git refs/heads/master
```

用各自对应的40位SHA分别执行`inventory --revision <SHA>`；不能混用两个Repo的SHA。若远端不支持该SHA查询API，**先停下**，不要通过删掉`master`检查伪称不可变版本。接下来考虑人工审计master时刻的树指纹、文件Blob/SHA和下载前后清单比对，但那是独立的降级模式，尚未实现。

另外，已依据**已安装的modelscope-hub v0.4.5**公共`HubApi.download_file` API修正旧脚本：它接受`repo_id, repo_type, file_path, revision, local_dir, expected_sha256`，但**不接受`file_size`**。对应纯CPU回归测试覆盖该参数签名。若运行2号机的分支早于这次修复，先`git fetch`并更新到最新分支，否则下载可能因TypeError失败。

# Recovery-Aware SmolVLA：集成缺口与可执行验收顺序

**状态：P0 静态实现/纯CPU合同检查。没有真实Recovery样本或GPU训练/闭环验证。**
与正在运行的Corrector训练隔离；不要修改 \`/root/autodl-tmp/corrector\`。

## 来源核查已确认：Expert优先按LeRobot v2.1准备（2026-11-19）

论文[arXiv:2609.05178v4](https://arxiv.org/html/2609.05178)附录B.5和C.2已经确认：**Expert为v2.1格式、20Hz、双相机；`meta/info.json`+`episodes.jsonl`+`tasks.jsonl`+`episodes_stats.jsonl`、每Episode Parquet和两路MP4。** 不是Plus的v3分片布局。

新增`recovery_v21.py`与`recovery_cli plan-v21-files`：在真实元文件到位后读取`info.json`中**实际**`chunks_size`、`data_path`、`video_path`和两路视频特征键，解析最多100个指定Episode的精确相对路径；默认不下载、不解析视频、不占用GPU。示例：

```bash
PYTHONPATH=src python -m vla_finetuning.recovery_cli plan-v21-files \
  --dataset-root datasets/recovery_expert/PINNED \
  --episodes 0,1,2 \
  --out runs/recovery_v21_first3_file_plan.json
```

论文没有公开真实发布快照的精确视频键/文件大小/各维状态；ModelScope源文件本次尚未下载。**不能把评测端8维EEF状态、0/1抓手协议直接宣称已经由Expert训练数据实测确认**。

固定LeRobot中有`src/lerobot/scripts/convert_dataset_v21_to_v30.py`，但**会删除/移动本地目录，默认可能上传**。禁止在不可恢复的原始数据目录直接运行；应另行设计可重复、独立复制和SHA校验的转换流水线。此门槛尚未通过。

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

## 已核实：固定LeRobot版本不能直接接收两个repo_id

在固定源码 `huggingface/lerobot@7e241bd630a3719a56157a497ce5d08f244784f1` 中，
`src/lerobot/configs/default.py` 的 `DatasetConfig.repo_id` 是单个 `str`；
`src/lerobot/configs/train.py` 的验证代码包含：

```python
if isinstance(self.dataset.repo_id, list):
    raise NotImplementedError("LeRobotMultiDataset is not currently implemented.")
```

因此 `--dataset.repo_id=[Expert,Normal]` **不能运行M2**。可行候选是：（A）把经核实的Expert和普通示教转换成一个版本化、单一LeRobot v3数据集，保存原始来源字段；（B）在固定LeRobot训练器上开发经过独立测试的多源Dataset+Sampler适配。**目前不能假设任一方式已经实现。** 优先A，目的是尽量复用上游单数据集训练、processor与Checkpoint流程。

现增加 `src/vla_finetuning/recovery_merge_plan.py`：它只在真实split和对齐证据、两份v3 `meta/info.json` 存在且维度、FPS、相机键通过审核时，生成重新编号的Episode映射；**不写Parquet/MP4，也不生成可训练数据集**。若Expert仍是v2.1，必须先完成v2→v3的实测转换，不能悄悄修改`codebase_version`。

```bash
PYTHONPATH=src python -m vla_finetuning.recovery_cli plan-materialization \
  --split runs/REAL_AUDITED_SPLIT.json \
  --alignment runs/REAL_ALIGNMENT.json \
  --out runs/RECOVERY_MIX_MAPPING_001.json
```

配置草案：`configs/recovery_alignment.template.json`，默认`verified=false`且所有语义证据为TBD，预期STOP。需要实际人工作业确认以后才允许生成上述计划。

## 2026-11-20：真实Expert 3条Episode已下载、远端SHA验证通过

实际ModelScope数据快照固定在`2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73`，真实LeRobot v2.1 info：`fps=20`、3,184 Episode、625,731帧、110任务、两路256×256 AV1视频，动作7维，状态8维。首批选择Episode 0/15/30，计划帧数629/796/647（共2,072），实际远端文件清单9个文件、2,971,391字节。用户终端回传`VERIFIED`9条并生成`.vla_recovery_verified_pilot3.json`，说明下载和远端SHA检查通过；**尚无任何Parquet实际数值/视频解码证据**。

新增`scripts/audit_recovery_payload.py`对本地已下载的文件执行read-only检查：逐文件与原ModelScope Manifest的SHA对比、Parquet数据维度/有限值/帧序号/时间间隔、七维动作及八维状态各维统计、可用`ffprobe`则实际AV1解码并检查每路视频的尺寸/帧率/帧数。输出报告明确`training_ready=false`，避免误把可解码当作动作控制、归一化或来源已证实。

在服务器的`recovery-p0`已检出且工作区干净时更新：

```bash
cd /root/autodl-tmp/vla_finetuning
git fetch origin
git switch recovery-p0
git merge --ff-only origin/feature/recovery-p0-offline-foundation-20261119
REV=2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73
DATA="$PWD/datasets/recovery_expert/$REV"
MANIFEST="runs/recovery_pilot3_manifest_$REV.json"
PYTHONPATH=src .venvs/data/bin/python scripts/audit_recovery_payload.py \
  --dataset-root "$DATA" --manifest "$MANIFEST" --episodes 0,15,30 \
  --out "runs/recovery_payload_audit_$REV.json"
```

`ffprobe`在PATH中可用时自动解码验证全部6个MP4；否则生成`VIDEO_PROBE_PENDING`状态，必须之后安装合适AV1解码器、复跑到新的报告名，不能将缺失检查视为通过。核查使用`pyarrow`，不需要Torch或CUDA。

即便3条Episode都检查通过：**动作究竟是相对控制还是绝对控制、旋转轴角的具体计算方式、抓手0/1编码、训练视频的翻转与归一化、Expert来源场景关联仍未验收**。不得直接启动M1长训。


## 2026-11-20：3条真实Parquet数值审计回传

用户提供的`audit_recovery_payload.py`服务器输出：`VIDEO_PROBE_PENDING`，`total_parquet_rows=2072`。Episode 0/15/30分别为629/796/647行，序号从0开始连续，`task_index`分别0/1/2，时间戳0～31.4、0～39.75、0～32.3秒；20Hz名义0.05秒时间间隔的偏差约在3.05e-6秒以内。全部7维Action与8维State数值均为有限值，动作第7维在三条Episode里都只出现0/1。

**不能由此认定动作物理单位/坐标系/缩放已核实**。Episode 30前三维动作范围分别约[-0.517,0.447]、[-0.812,0.659]、[-0.696,0.822]，这很可能不是直接以米表示的单步物理位移；必须根据原始遥操作转换、MuJoCo控制器逻辑和状态差分进行交叉验证。状态最后两维连续、正负值可能对应双抓手关节坐标，不能将其误当作动作开合二值。

**下一硬阻塞为六个AV1 MP4尚未实际解码**：输出两路相机均为`NOT_PROBED`。若远端机器缺少`ffprobe`，可以在原始数据不受影响的前提下安装ffmpeg/ffprobe，再以**新的**JSON报告路径重跑审计。视频通过后，还须输出相机前/中/末帧供可视检查方向及动作-观测对应。三条任务文本虽然不同，但均是cabinet top drawer相关示教，不能据此宣称跨Suite覆盖。没有实际SmolVLA处理器/训练Batch或Recovery闭环结果；仍不得启动正式长训。


## 2026-11-20：P1首批视频全帧验证完成，开始SmolVLA静态合同检查

用户在2号机执行ffprobe对Episode 0/15/30的前视、腕部六个AV1 MP4均得到PASS；对应629、796、647帧。加上之前Parquet序号/20Hz、SHA256和PNG单帧抽取，**P1首批样本的数据文件完整性验证已完成**，但仅限这三条Episode；不是整个3,184条数据的审核，也不代表训练和执行语义通过。

经核对Hugging Face模型真实公开配置（模型仓库版本仍须独立固定），\`lerobot/smolvla_libero\`的\`input_features\`期望：
- \`observation.state\` [8]；
- \`observation.images.camera1\`和\`observation.images.camera2\` [3,256,256]；
- \`action\` [7]；
- 模型默认chunk_size=50、n_action_steps=50；
- 预处理器rename映射原图像键\`observation.images.image → observation.images.camera1\`和\`observation.images.image2 → observation.images.camera2\`。

而**真实Recovery Expert**第二相机键是\`observation.images.wrist_image\`，不是\`observation.images.image2\`。需要在使用预训练Processor**前**显式进行\`observation.images.wrist_image → observation.images.image2\`（或经严格验证后修改Processor的rename配置）；不可同时做两次映射，也不可默认会自动适配。动作/状态shape虽然兼容，动作坐标和数值归一化仍未证实。

新增**只读**脚本：

\`\`\`bash
cd /root/autodl-tmp/vla_finetuning
git fetch origin
# 仅在工作区无改动且当前分支是recovery-p0时：
git merge --ff-only origin/feature/recovery-p0-offline-foundation-20261119

# 仅获取模型的两个小JSON，**不下载权重**，Hugging Face API自动固定40位SHA
python scripts/fetch_smolvla_contract_metadata.py \
  --out-root models/smolvla_libero_contract --allow-network

# 从上条命令path值填入，以实际SHA为准
MODEL_CONTRACT_DIR="models/smolvla_libero_contract/实际40位SHA"
EXPERT_REV=2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73
python scripts/audit_recovery_smolvla_contract.py \
  --expert-info "datasets/recovery_expert/$EXPERT_REV/meta/info.json" \
  --policy-config "$MODEL_CONTRACT_DIR/config.json" \
  --policy-preprocessor "$MODEL_CONTRACT_DIR/policy_preprocessor.json" \
  --out "runs/recovery_smolvla_static_contract_$EXPERT_REV.json"
\`\`\`

输出\`STATIC_FEATURE_BRIDGE_PROPOSED_NOT_TRAIN_READY\`是**正确状态**，它验证字段形状和checkpoint Processor的静态命名兼容性，并明确输出桥接建议。该脚本不会实际加载权重、Tokenizer、视频或生成真实训练Batch，更不会进行LoRA训练。下一步需要使用固定LeRobot和SmolVLA真实模型，检查Processor Pipeline、RGB通道/转置、图像方向、state/action标准化权重文件、任务Tokenization和动作chunk padding/恢复接口。

注意**预训练Checkpoint自带的Mean/Std统计来自原LIBERO训练分布**。Recovery后训练的训练归一化策略必须固定，并在M0/M1/M2评测中保证正确逆归一化。绝对不能直接将\`meta/stats_gr00t.json\`称为SmolVLA的预处理统计。静态合同检查为\`ready_to_train=false\`是预期结果。

## 2026-11-20：AutoDL请求HF时`Errno 99`网络故障处理

真实2号机使用`python scripts/fetch_smolvla_contract_metadata.py --allow-network`返回`<urlopen error [Errno 99] Cannot assign requested address>`，发生在Python urllib对`huggingface.co`联网阶段，**不能据此推断模型不存在或数据不兼容**。常见原因IPv6/代理/源地址选择，需先检测实际连接。

```bash
curl -4 -L -sS --connect-timeout 10 --max-time 20 -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/lerobot/smolvla_libero
curl -4 --noproxy '*' -L -sS --connect-timeout 10 --max-time 20 -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/lerobot/smolvla_libero
```

已添加`--transport curl-ipv4`（优先尊重当前代理）或`--transport curl-ipv4-direct`（明确绕过代理）；二者使用Curl自身连接机制，不依赖Python urllib，固定模型SHA后才下载两个小JSON，保留2MiB单文件限额与不覆盖文件原则。升级当前Recovery分支后执行：

```bash
python scripts/fetch_smolvla_contract_metadata.py \
  --out-root models/smolvla_libero_contract \
  --allow-network --transport curl-ipv4
```

如果curl仅在`--noproxy '*'`时访问成功，使用`--transport curl-ipv4-direct`。若两者皆失败，先修复网络，**不能跳过远端固定版本/来源证明**。

## 2026-11-20：HF 443双路径失败，改为受控镜像（网络降级）

用户2号机的 `getent ahostsv4 huggingface.co` 输出 `199.193.116.105`，`curl -4` 在默认代理环境和 `--noproxy '*'` 时均超时HTTP 000。外部公开DNS在2026-11月记录`3.167.112.x`，**解析不同但不能断言DNS劫持**；同时还可能是出站443网络限制。不要改写全局 `/etc/hosts` 或禁用TLS校验。

新的 `fetch_smolvla_contract_metadata.py` 支持 `--endpoint https://hf-mirror.com`，并保留 `--transport curl-ipv4` 和 `--transport curl-ipv4-direct`。先测试：

```bash
curl -4 -L -sS --connect-timeout 8 --max-time 20 -o /dev/null \
  -w 'HTTP=%{http_code} URL=%{url_effective}\n' \
  https://hf-mirror.com/api/models/lerobot/smolvla_libero
```

如HTTP 200，更新Recovery分支后仅下载两个小JSON：

```bash
python scripts/fetch_smolvla_contract_metadata.py \
  --out-root models/smolvla_libero_contract \
  --endpoint https://hf-mirror.com \
  --allow-network --transport curl-ipv4
```

**注意**：`hf-mirror.com`是第三方镜像。工具只允许官网或固定HTTPS镜像域名，保留40位SHA和逐文件哈希，以及 `source_endpoint/source_is_third_party_mirror` 字段；此SHA由镜像报告，**不能视为官方独立签名校验**。如镜像回源到被阻断的 `huggingface.co`，仍会失败。此时优先验证ModelScope上同名checkpoint的精确来源/文件哈希，不得把 `HuggingFaceVLA/smolvla_libero` 未证实的版本冒充 `lerobot/smolvla_libero`。

无需重新下载已完成验证的三条Recovery Episode。

## 2026-11-20：静态SmolVLA契约真实通过，推进CPU真实训练Batch样本

用户在AutoDL通过镜像HTTPS获取`lerobot/smolvla_libero`模型配置，镜像报告版本`a24f1f380461b665158d52acd140152c79e8b36d`，`config.json` sha256 `3ed3ed80ee077d94ff22942130642bb60e255443ee65da676bc88ab1684a7efe`，`policy_preprocessor.json` sha256 `7beef33479fc22c9a45670154b3c7b909fb6d51dac3cd6db6f23376638b4ac11`，都由用户终端校验。

真实运行`audit_recovery_smolvla_contract.py`得到`STATIC_FEATURE_BRIDGE_PROPOSED_NOT_TRAIN_READY`：Expert两路视频原始键`observation.images.image`、`observation.images.wrist_image`，模型目标`observation.images.camera1`、`observation.images.camera2`；预训练Processor要求`image2 -> camera2`，新增的临时桥接`wrist_image -> image2`只能执行一次；8D state、7D action、`chunk_size=50`、`n_action_steps=50`、`STATE/ACTION=MEAN_STD`在静态维度上匹配。镜像SHA还需官方一致性复核。

新增`src/vla_finetuning/recovery_batch.py`与`scripts/build_recovery_pilot_batch.py`：读取SHA校验后的真实Expert v2.1 Parquet＋两路AV1视频（CPU FFmpeg按帧编号精确选择），将一帧RGB转为两个CHW float32 [0,1]输入，输出8维原始state、[50,7]未归一化动作、[50]`action_is_pad`（LeRobot `DatasetReader._get_query_indices`的Episode边界clamp规则）、真实task字符串和验证收据。**不跨Episode、不删除原数据、不写Hub、不加载模型/Processor/Normalizer、不占GPU**。输出仅作样本桥接/测试，不能作为已训练Batch。

操作（先确认分支与git工作区干净，更新分支）：

```bash
cd /root/autodl-tmp/vla_finetuning
git status -sb
git fetch origin
git merge --ff-only origin/feature/recovery-p0-offline-foundation-20261119
EXPERT_REV=2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73
MODEL_REV=a24f1f380461b665158d52acd140152c79e8b36d
DATA="$PWD/datasets/recovery_expert/$EXPERT_REV"
MANIFEST="runs/recovery_pilot3_manifest_$EXPERT_REV.json"
CONTRACT="runs/recovery_smolvla_contract_${EXPERT_REV}_${MODEL_REV}.json"
PYTHONPATH=src .venvs/data/bin/python scripts/build_recovery_pilot_batch.py \
  --dataset-root "$DATA" --manifest "$MANIFEST" --contract "$CONTRACT" \
  --episode 0 --frame 10 \
  --out-npz runs/recovery_batch/ep0_f10.npz \
  --out-report runs/recovery_batch/ep0_f10.json

# Episode 0 has 629 frames; f620 starts 9 valid and 41 padded actions:
PYTHONPATH=src .venvs/data/bin/python scripts/build_recovery_pilot_batch.py \
  --dataset-root "$DATA" --manifest "$MANIFEST" --contract "$CONTRACT" \
  --episode 0 --frame 620 \
  --out-npz runs/recovery_batch/ep0_f620.npz \
  --out-report runs/recovery_batch/ep0_f620.json
```

Expected in first report: `valid_action_steps=50, action_is_pad_true_count=0`; in f620 report: `valid_action_steps=9, action_is_pad_true_count=41`. The `.npz` has keys `observation.images.image`, `observation.images.image2`, `observation.state`, `action`, `action_is_pad`, `task` and can be loaded with NumPy `allow_pickle=False`. The two camera arrays each have shape `[3,256,256]` float32 [0,1], state `[8]` and action `[50,7]`.

After this: implement/run actual pinned LeRobot+SmolVLA Processor/Tokenizer/Normalizer weight loading, action control semantic cross-check, real GPU forward/backward and LoRA optimizer; **none of those is yet validated**.


## 2026-11-20：首次真实样本已生成；下一阶段仅离线Processor smoke

用户已使用真实Episode 0第10帧执行新脚本并得到`REAL_RECOVERY_PILOT_SAMPLE_BUILT_NO_TRAINING`。原始样本：两路RGB `[3,256,256]` float32 `[0,1]`、状态`[8]`、动作`[50,7]`、`action_is_pad`全False、任务`put the black bowl in the top drawer of the cabinet`、来源文件哈希与NPZ SHA已记录；NPZ SHA256 `8558bbe07553c8e7fc51dbe8533aab9ca90a897425ce9af1afd3136ac9c890bf`。先前缺少NumPy，已由用户在数据venv安装`numpy==1.26.4`成功修复。

**注意实际LeRobot源码细节：** `AddBatchDimensionActionStep`仅为1D action自动补Batch维度，单样本`[50,7]`不会自动变成`[1,50,7]`。新`smoke_recovery_processor.py`会在调用预训练Processor前显式对动作和`action_is_pad`增加B维，而保持相机CHW和状态8D未批处理，交给官方`AddBatchDimensionObservationStep`操作。检查双相机映射、真实Tokenizer、真实预处理Normalize状态，不允许缺失统计文件时静默跳过。

**此处对外网络只需很小的辅助资产，训练Checkpoint权重仍不下载：**

```bash
cd /root/autodl-tmp/vla_finetuning
git status -sb
git fetch origin
git merge --ff-only origin/feature/recovery-p0-offline-foundation-20261119
MODEL_REV=a24f1f380461b665158d52acd140152c79e8b36d
MODEL_DIR="models/smolvla_libero_contract/$MODEL_REV"

# 仅下载该SmolVLA版本的preprocessor normalizer safetensors文件
python scripts/fetch_smolvla_processor_state.py \
  --model-dir "$MODEL_DIR" \
  --endpoint https://hf-mirror.com \
  --transport curl-ipv4 --allow-network

# 仅下载SmolVLM2 tokenizer相关文件，额外用Model Hub SHA固定版本；不下载模型权重
python scripts/fetch_smolvla_tokenizer.py \
  --out-root models/smolvlm2_tokenizer \
  --endpoint https://hf-mirror.com \
  --transport curl-ipv4 --allow-network
```

第二个命令输出本地`path`，例如`models/smolvlm2_tokenizer/<actual_40_char_sha>`；务必使用**实际输出SHA**，不要假设它等于SmolVLA策略SHA。

同时完成之前承诺的Episode 0尾部Padding测试（第620帧仅9步有效、41步补齐）以证明不串Episode。两个例子都属于CPU读取样本，可在`.venvs/data`执行。

只有在`.venvs/train`已安装固定版LeRobot、Torch、Transformers和Tokenizer依赖后才执行：

```bash
PYTHONPATH=src .venvs/train/bin/python scripts/smoke_recovery_processor.py \
  --sample runs/recovery_batch/ep0_f10.npz \
  --sample-report runs/recovery_batch/ep0_f10.json \
  --model-dir "$MODEL_DIR" \
  --tokenizer-dir models/smolvlm2_tokenizer/<actual_40_char_sha> \
  --out runs/recovery_batch/ep0_f10_processor_smoke.json
```

预期状态`REAL_PREPROCESSOR_SMOKE_PASS_NO_MODEL_WEIGHTS`，输出`camera1/camera2 [1,3,256,256]`、state `[1,8]`、action `[1,50,7]`、mask `[1,50]`、批量语言token。真实状态/动作已经应用**对应预训练checkpoint的**均值标准差，但这只验证旧统计可加载，不代表其适合Recovery训练分布。后续LoRA需明确是否采用训练集统计重算策略并同步保存预处理/反归一化。

**脚本必须离线、必须有完整来源收据：** `config.json`、`policy_preprocessor.json`、normalizer safetensors及receipt、Tokenizer文件和receipt。缺失时STOP，不会自动下载任何缺失项或加载/更新网络模型。无法通过ModelProcessor的计算不允许标记模型已训练。

## 2026-11-20：真实Episode末尾Padding/Normalizer/Tokenizer完成，训练环境Python门槛

用户真实AutoDL终端Episode 0/frame620的`REAL_RECOVERY_PILOT_SAMPLE_BUILT_NO_TRAINING`已通过：帧620–628共9步有效，剩余41步Action Chunk clamp至628并正确`action_is_pad=True`；输出NPZ SHA256为`ff84b655a4637088f716564b33600c769949ab5be2e38f008db4f6e469380dac`。

固定SmolVLA模型SHA `a24f1f380461b665158d52acd140152c79e8b36d`的Normalizer统计Safetensors已通过受限hf-mirror工具下载（5676 bytes, SHA256 `b0cdde6e8a6f49a8e19eefb376728e47c09d3b3cc20ce3a97c45619fe7a732d9`）。VLM Tokenizer `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`自身镜像SHA `4cde9d208e19519c36917aa170fffc7e381aa0ad`的7个配置与词表文件也下载成功。都保留来源、哈希收据，**尚未由真实LeRobot Processor加载**。

用户的`.venvs/train`无Python可执行文件，显示`TRAIN_ENV_NOT_READY`；这只是未安装，不表示上面数据失败。**源码修正（重要）**：固定LeRobot提交`7e241bd630a3719a56157a497ce5d08f244784f1`的真实`pyproject.toml`是**LeRobot 0.6.1、Python>=3.12、numpy>=2,<2.3**。此前`.venvs/data`安装过`cp310`的NumPy1.26.4，它必须保留作纯数据环境，不能拿来安装此固定LeRobot，也不能使用Corrector环境。

修复`scripts/bootstrap_train_env.sh`：`--apply`的最前面检查候选Python>=3.12，不达标会在任何目录写入/安装前STOP；允许`VLA_TRAIN_PYTHON`显式选择Python3.12/3.13解释器；原来的`[peft]`安装项改为固定源的`[smolvla,peft,dataset]`，从而包含SmolVLA必需的Transformers/Num2words/Accelerate、LeRobot dataset、PyArrow、PyAV等依赖。**注意安装会下载大量PyTorch/CUDA依赖，并保留 >=50GiB可用空间及GPU无占用等门槛；不可在没有查看硬件/环境的情况下直接执行`--apply`**。

新增无副作用诊断：

```bash
cd /root/autodl-tmp/vla_finetuning
git status -sb
git fetch origin
git merge --ff-only origin/feature/recovery-p0-offline-foundation-20261119
python scripts/preflight_recovery_train_env.py
bash scripts/bootstrap_train_env.sh
```

若有Python3.12候选，按`bootstrap`提示使用选定解释器及显式CPU/IO同意；若不存在，独立新建`conda` Python3.12环境，例`/root/miniconda3/bin/conda create -p /root/autodl-tmp/vla_finetuning/.venvs/train python=3.12 pip -y`（此动作需要安装授权、下载和磁盘），再将`VLA_TRAIN_PYTHON`指定为`/root/autodl-tmp/vla_finetuning/.venvs/train/bin/python`。不得把3.10的data venv迁移到训练目录。

安装完成并通过`pip check`、`import torch, lerobot, transformers, peft, pyarrow, av`后再运行`scripts/smoke_recovery_processor.py`，并指定已下载的本地Normalizer/Tokenizer真实目录。未运行此smoke与模型权重前向/反向之前，仍处于`ready_to_train=false`。
