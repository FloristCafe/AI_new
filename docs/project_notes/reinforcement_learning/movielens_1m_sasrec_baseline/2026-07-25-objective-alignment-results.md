# 2026-07-25 Objective Alignment 实验结果

## 实验目的

本轮实验的目标不是继续扩大结构搜索，而是回答一个更核心的问题：

- 在相同模型结构下
- 把训练目标从 `CrossEntropy` 切换到 `BCE + negative sampling`
- 是否会让当前的 MovieLens-1M SASRec baseline 更强

本轮对比固定同一套结构参数：

- `max_seq_len = 50`
- `embedding_dim = 128`
- `num_heads = 2`
- `num_blocks = 3`
- `dropout = 0.2`

对比对象为：

- `seq50_dim128_blocks3_drop02_ce`
- `seq50_dim128_blocks3_drop02_bce_ns1`

## 实验结果概览

### CE baseline

- `best valid NDCG@10 = 0.11382445640333486`
- `test HR@10 = 0.18658940397350993`
- `test NDCG@10 = 0.10535078244322423`
- `best_epoch = 18`
- `epochs_completed = 20`

### BCE + negative sampling (ns=1)

- `best valid NDCG@10 = 0.009431989823338588`
- `test HR@10 = 0.02119205298013245`
- `test NDCG@10 = 0.009746045923646767`
- `best_epoch = 7`
- `epochs_completed = 10`
- `stopped_early = true`

## 结论

本轮实验结论非常明确：

- 当前 `CE baseline` 明显强于 `BCE + negative sampling(ns=1)`
- 这不是轻微差距，而是数量级上的差距

如果只看测试集：

- `CE test NDCG@10 = 0.10535`
- `BCE test NDCG@10 = 0.00975`

说明当前这版 `BCE` 训练方案还不能作为主线替代 `CE baseline`。

因此截至 2026-07-25，项目中的可靠主基线仍然是：

- `seq50_dim128_blocks3_drop02_ce`

## 结果应该如何理解

需要特别注意：

- `BCE` 的 `train_accuracy` 很高
- 但这并不代表它真的学会了高质量推荐排序

当前 `BCE` 训练日志中的 `train_accuracy` 定义是：

- 正样本 logit 是否高于采样负样本 logit

这只是一个局部判别指标，不等于最终的全量排序能力。

当前实验里：

- `BCE` 能较快把“正样本 > 1 个随机负样本”这件事学出来
- 但它没有把真实下一个 item 在全量 item 中稳定排到前列

而评估阶段看的是：

- 在全部候选 item 中进行 Top-K 排序
- 指标是 `HR@10` 和 `NDCG@10`

所以会出现一种典型现象：

- 训练指标看似不错
- 但验证/测试排序指标极差

## 为什么当前这版 BCE 很可能会弱

当前最可疑、也最值得优先怀疑的点有以下几项：

### 1. `num_negative_samples = 1` 太弱

这是最核心的问题。

只采一个负样本时，训练目标变成：

- 让正样本分数高于一个随机负样本

这个任务过于简单，不能给模型足够强的排序压力。

### 2. BCE 直接沿用了 CE 的训练超参数

当前 BCE 仍然使用了与 CE 接近的：

- `learning_rate = 1e-3`
- `early_stop_patience = 3`
- `epochs_requested = 20`

这未必适合 BCE 路线。

### 3. 训练目标与评估目标之间仍存在落差

当前 BCE 优化的是：

- 局部正负样本分离

而最终评估要求的是：

- 全量 item 排序质量

如果负采样过弱，这种落差就会非常明显。

## 这轮实验对项目意味着什么

这轮实验并不说明：

- `BCE` 路线完全错误

它说明的是：

- 当前这个实现版本里的 `BCE + ns=1` 不够强

也就是说，项目已经获得一个非常有价值的阶段性结论：

- `CE baseline` 已经稳定、可靠、可复用
- `BCE` 路线不能直接用最弱配置替代它

因此项目下一步不应是“继续大改模型结构”，而应是：

- 精准修正 BCE 实验设计

## 项目下一步建议

下一轮实验建议不要铺太大，而是聚焦在 `BCE` 的关键修正上。

优先级建议如下：

### 第一优先级

- 把 `num_negative_samples` 从 `1` 提高到 `5` 和 `10`

这是最值得先做的修正，因为当前问题最像“负样本监督太弱”。

### 第二优先级

- 下调学习率，例如从 `1e-3` 调到 `5e-4` 或 `1e-4`

### 第三优先级

- 增加训练轮数，例如 `20 -> 30`
- 提高 early stopping patience，例如 `3 -> 5`

## 当前阶段的工作判断

截至 2026-07-25，本项目的状态可以概括为：

- `CE baseline` 已经打稳
- `BCE` 目标对齐已经具备代码实现
- 但第一轮 `BCE ns=1` 实验结果失败
- 因此项目正式进入“修正 BCE 训练策略”的阶段
