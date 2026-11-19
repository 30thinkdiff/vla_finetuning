# 训练面试必会技能：跟着代码学，而不是背名词

学习顺序：每次先自己解释，再运行验证，再回头读固定LeRobot源码。

## 01. Flow Matching（可立即练，无GPU/torch）

运行`python -m vla_finetuning flow-demo`，得到两维动作示例：

- 专家动作`a=(0.2,0.8)`，噪声`eps=(0.9,-0.4)`，随机时间`t=0.25`
- 插值`x_t=t*eps+(1-t)*a=(0.375,0.5)`
- 正确目标**速度**`u=eps-a=(0.7,-1.2)`
- 若模型预测`(0.6,-1.1)`，逐元素平方误差均值`0.01`

打开`src/vla_finetuning/flow_math.py`，解释：为什么预测速度而非直接监督动作？为什么`t=0`是clean而`t=1`是noise？为什么末端padding不该计入loss？

若已有PyTorch，可以运行`PYTHONPATH=src python -m unittest tests.test_torch_optional -v`查看有效位置非零梯度、padding零梯度。此例是教学用的等价数学表达，不能冒充模型真实训练结果。

## 02. VLA真实数据与时间

打开固定LeRobot数据加载器，对一个真实Episode追踪：前、腕相机原始HWC→处理后CHW/观测轴；state 8维；action chunk`[B,50,7]`；padding mask；任务语言token。重点解释：为什么相机相同分辨率不保证相同语义？为什么10Hz→20Hz必须审计？

## 03. LoRA与梯度

手算`W0 + (alpha/r)B A`的参数数量。查`audit_trainable_parameters`检查实际模型中`requires_grad=True`的名字/数量；默认应验证可训练参数是否真是预期LoRA，不直接假设所有`q_proj`都属于Expert。给出一次`zero_grad→forward→backward→clip→optimizer.step`的计算图，解释冻结上游权重不等于对全部forward使用`torch.no_grad()`。

## 04. 实验能证明什么

对比训练Loss下降和闭环成功率改善：为何可能相反？设计一套组别独立的源示范/失败源场景分组，指出从`smolvla_libero`出发再用原始LIBERO示范为什么不能宣称新数据泛化。解释LoRA-only和Expert+Vision两种实验如何固定同一训练数据和计算预算。

## 面试检查题（先口头作答，再查代码）

1. 已知`a=2`、`eps=-2`、`t=0.75`，求`x_t`和目标速度；若网络输出`-3`，MSE是多少？
2. 第50步动作块后49步是padding，怎样才能保证它们对Loss贡献恰为0？
3. 为什么已看到`optimizer.step()`不能证明LoRA权重真的更新？
4. 为什么两个来源的`task_index=3`不能直接认定是同一真实任务/扰动？
5. 当视角微调后验证Loss更低、但Plus成功率更差时，你会首先检查哪些接口和数据分布？
