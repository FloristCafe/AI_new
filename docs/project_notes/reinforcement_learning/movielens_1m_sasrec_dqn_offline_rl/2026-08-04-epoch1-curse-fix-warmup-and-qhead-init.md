# 2026-08-04 打破 Epoch 1 魔咒：Encoder Warmup 与 Q-Head 小尺度初始化

## 这次改动解决的是什么问题

在 SASRec-DQN 早期实验里，曾经多次出现：

- `best_epoch = 1`
- 后续 epoch 的 `valid_ndcg_at_10` 持续回落

这不是一个健康的训练现象。

它通常意味着模型在刚开始时仍然保留了预训练 SASRec 的排序能力，但随着离线 RL 的目标持续反传，排序结构被快速破坏，而新的价值结构又没有及时建立起来。

## 本次加入的两个机制

### 1. Encoder warmup freeze

训练脚本现在支持：

- `--encoder-warmup-epochs`

作用是：

- 前 `N` 个 epoch 冻结 SASRec encoder
- 只训练新接上的 `Q-head`
- warmup 结束后，再自动解冻 encoder，并恢复 `encoder_learning_rate`

这个机制的物理意义是：

- 先让 Q-head 在稳定的预训练表征上建立初始 Q 值语义
- 避免一开始就让 noisy RL gradient 直接冲刷 Transformer 表征

### 2. Q-head 小尺度初始化

模型现在支持：

- `--q-head-init-std`
- `--q-head-init-mean`

默认做法是：

- 将 `Q-head` 权重按零均值、小方差重新初始化
- 将 bias 置零

它的目的不是提升容量，而是控制初始 Q 值尺度，让训练一开始不要因为头部随机输出过大而把 TD target 和 `max Q` 迅速拉歪。

## 代码层做了什么

### 模型侧

在 `sasrec_dqn_model.py` 中新增：

- `reset_q_head_parameters()`

并给优化器参数组加上了显式名字：

- `q_head`
- `encoder`

这样训练器就能在 warmup 期间把 encoder 参数组的学习率降到 `0.0`，在解冻时再恢复。

### 训练侧

在 `train_sasrec_dqn.py` 中新增：

- `--encoder-warmup-epochs`
- `--q-head-init-std`
- `--q-head-init-mean`

并加入：

- `set_encoder_optimization_state()`

它会在 epoch 边界自动切换：

- encoder 的 `requires_grad`
- optimizer 中 encoder 参数组的学习率

同时训练日志里新增：

- `train_encoder_learning_rate`

这样可以直接从 `training_metrics.json` 看出每个 epoch 时 encoder 是否处于 warmup 冻结状态。

## 这次改动的实验意义

这不是普通的小调参，而是对训练动力学的直接干预。

它检验的是一个更底层的问题：

- 之前的 `best_epoch = 1`，到底是因为 RL 目标本身不适合，还是因为训练早期缺少过渡机制

如果加入 warmup 后：

- `best_epoch` 明显后移
- `valid_ndcg_at_10` 不再一开始最好
- 训练曲线更平滑

那就说明之前的一个核心问题，确实是“预训练排序表征在第 1 轮之后被过快破坏”。

## 建议的下一步对照

最合理的对照不是一次改很多，而是固定 strongest baseline 其他设置，只比较：

1. 无 warmup
2. `warmup = 3`
3. `warmup = 5`

可以先继续保持：

- `gamma = 0.9`
- `binary reward`
- `valid_ndcg_at_10` 选模
- `CE regularization`

如果这条线有效，再继续叠加到后面的 `BPR` 线或更高级的 offline RL 方法线上。
