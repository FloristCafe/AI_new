# 2026-07-26 BCE 修正 Round 1

## 本次修正目的

上一轮 `objective_alignment` 实验表明：

- `CE baseline` 明显强于 `BCE + negative sampling(ns=1)`
- 当前 `BCE` 版本不能直接作为主线替代 `CE`

因此本次不是继续扩大结构搜索，而是对 `BCE` 路线做一次更针对性的修正。

## 本次代码修正

### 1. 修正负采样的语义

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\sasrec_utils.py`

之前的负采样逻辑有一个关键问题：

- 对某个位置 `t` 采负样本时
- 它会排除整个输入窗口中出现过的 item
- 包括 `t` 之后才出现的 future item

这不符合序列建模的前缀语义。

现在已经修正为：

- 只排除当前位置及其之前已经见过的 item
- 同时排除当前位置对应的正样本 item

也就是说，负采样约束从“整窗 seen items”变成了“前缀 seen items”。

这一步更符合：

- 因果序列建模
- SASRec 的按位置预测逻辑

### 2. 新增 BCE 修正实验 preset

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\run_sasrec_experiments.py`

新增 preset：

- `bce_repair_round1`

其中包括：

- `seq50_dim128_blocks3_drop02_ce_reference`
- `seq50_dim128_blocks3_drop02_bce_ns5_lr5e4`
- `seq50_dim128_blocks3_drop02_bce_ns10_lr5e4`
- `seq50_dim128_blocks3_drop02_bce_ns10_lr1e4`

## 这轮修正背后的思路

上一轮 `BCE` 失败的最可疑原因有两类：

### 1. 负样本监督过弱

上一轮只用了：

- `num_negative_samples = 1`

这意味着模型只需要学会：

- 正样本分数高于 1 个随机负样本

这个目标太容易，不足以逼近最终的全量排序任务。

### 2. BCE 沿用了 CE 的训练节奏

上一轮 BCE 使用的训练配置与 CE 接近：

- `learning_rate = 1e-3`
- `epochs = 20`
- `early_stop_patience = 3`

这未必适合 BCE。

所以这一轮修正的策略是：

- 提高负样本数：`1 -> 5 -> 10`
- 降低学习率：`1e-3 -> 5e-4 / 1e-4`
- 延长训练轮数：`20 -> 30`
- 放宽 early stopping：`3 -> 5`

## 下一步建议执行

直接运行：

```bash
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\run_sasrec_experiments.py" --preset bce_repair_round1 --device cuda
```

## 下一轮最关注的问题

这一轮实验最关键不是看训练 loss 漂不漂亮，而是看：

- `best valid NDCG@10`
- `test NDCG@10`
- `BCE` 是否开始接近或明显追上 `CE reference`

如果仍然明显落后，说明下一步应当继续怀疑：

- BCE 训练目标本身的细节
- 负采样策略是否还不够强
- 是否需要进一步向更论文化的实现靠近
