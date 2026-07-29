# 2026-07-29 valid NDCG 选模修复与结果解读

## 本次记录的目标

这篇笔记记录一轮非常关键但容易被误判的修正：

- 不是继续改模型结构
- 也不是继续改 reward 公式
- 而是先修正 `best checkpoint` 的选择机制

上一轮实验已经证明：

- `Huber Loss + gamma=0.7 + binary reward` 可以把 Q-value 爆炸压住
- 但如果仍然按 `valid_total_loss` 选模型，最终测试效果会非常差

因此本轮工作的核心不是“重新训练一个完全不同的模型”，而是：

- 保留当前训练配置
- 在每个 epoch 计算 `valid HR@10` 和 `valid NDCG@10`
- 把 `selection_metric` 从 `valid_total_loss` 改为 `valid_ndcg_at_10`

## 这次代码层面改了什么

修改文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\train_sasrec_dqn.py`

主要改动：

1. 每个 epoch 结束时，除了原有的 loss 验证，还会额外计算：
   - `valid_hr_at_10`
   - `valid_ndcg_at_10`
2. 新增 `--ranking-topk` 参数，默认 `10`
3. `--selection-metric` 默认值改为：
   - `valid_ndcg_at_10`
4. 选 best checkpoint 的逻辑扩展为：
   - 对 loss 类指标，越小越好
   - 对 ranking 类指标，越大越好

这一步的本质是把“训练代理目标”和“最终业务目标”拉近。

## 为什么这一步是必要的

上一轮已经出现了一个非常典型的离线 RL 现象：

- `valid_total_loss` 持续变好
- 但测试集 `HR@10 / NDCG@10` 明显变差

这说明：

- 当前阶段的 `loss` 改善，并不等价于推荐排序质量改善

也就是说，如果我们继续让 checkpoint 选择依赖：

- `valid_total_loss`

那么系统就会偏向保存一个“更会拟合当前 TD/CQL 目标”的模型，而不是一个“更会推荐下一个物品”的模型。

## 本轮实验配置

实验目录：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma07_binary_reward_valid_ndcg`

主要配置保持为：

- `gamma = 0.7`
- `td_loss_type = smooth_l1_huber`
- `reward = binary`
- `target_tau = 0.005`
- `adaptive_cql_alpha = true`
- `selection_metric = valid_ndcg_at_10`
- `ranking_topk = 10`

## 训练结果怎么解读

训练指标文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma07_binary_reward_valid_ndcg\metrics\training_metrics.json`

本轮训练的关键信息如下：

- `best_epoch = 1`
- `best_valid_ndcg_at_10 = 0.05879283157268582`
- `cql_alpha_final = 0.05`
- `train_mean_q_value` 基本稳定在 `2.5` 左右
- `valid_max_q_value` 基本稳定在 `5.5` 左右

这说明两件事：

### 1. 训练稳定性依旧是成立的

相比最早期的爆炸版本，这轮依旧没有出现：

- Q 值疯狂抬升
- 验证损失失控
- 梯度大幅爆炸

因此，之前做的 `Huber + lower gamma + binary reward`，在“压住训练动力学”这个层面依然是有效的。

### 2. 排序能力依旧在第 1 轮最强

虽然 `valid_total_loss` 后面继续下降，但：

- `valid_ndcg_at_10` 在 `epoch 1` 就达到峰值

之后整体下降：

- epoch 1: `0.05879`
- epoch 2: `0.03331`
- epoch 3: `0.02650`
- epoch 8: `0.03089`
- epoch 10: `0.02683`

这说明：

- 当前训练目标继续优化下去，会让模型越来越偏离真正的排序目标

## 对比上一轮：选模修复到底有没有效果

这里必须做一个非常明确的对比。

### 上一轮同配置，但按 `valid_total_loss` 选模

实验目录：

- `cql_huber_gamma07_binary_reward`

测试集结果：

- `HR@10 = 0.05447019867549669`
- `NDCG@10 = 0.022712753417678085`
- `top1_exact_hit_rate = 0.002317880794701987`

### 这一轮按 `valid_ndcg_at_10` 选模

实验目录：

- `cql_huber_gamma07_binary_reward_valid_ndcg`

测试集结果：

- `HR@10 = 0.11374172185430463`
- `NDCG@10 = 0.052707259069488546`
- `top1_exact_hit_rate = 0.011754966887417218`

### 结论

这说明：

1. 当前模型并不是“完全学废了”。
2. 上一轮最大的问题之一，确实是 checkpoint 选错了。
3. 只修正选模逻辑，不改训练配置，测试集效果就出现了明显恢复。

从数量级上看：

- `HR@10` 大约翻倍
- `NDCG@10` 大约翻倍
- `top1 exact hit` 也提升明显

因此，本轮实验可以明确记为：

- `valid ranking` 选模修复成功

## 但为什么它仍然不是当前主线最优版本

虽然这轮比“按 loss 选模”的上一轮好很多，但它仍然没有回到更早的 DQN 版本水平。

对比更早的结果：

### `cql_baseline_full_2026_07_27`

- `test HR@10 ≈ 0.1649`
- `test NDCG@10 ≈ 0.0905`

### `cql_stabilized_full_2026_07_28`

- `test HR@10 ≈ 0.1520`
- `test NDCG@10 ≈ 0.0831`

### 当前版本 `cql_huber_gamma07_binary_reward_valid_ndcg`

- `test HR@10 ≈ 0.1137`
- `test NDCG@10 ≈ 0.0527`

所以当前最准确的判断是：

- 选模逻辑修对了
- 训练稳定性也保住了
- 但训练目标本身仍然过于保守，导致排序能力被压低

## 这轮实验暴露出的新认识

本轮结果帮助我们把问题进一步分层。

### 第一层：选模问题

这个问题现在已经基本确认：

- 在当前项目阶段，`valid_total_loss` 不是可靠的 checkpoint 选择指标
- `valid_ndcg_at_10` 更贴近最终目标

这部分已经修复。

### 第二层：训练目标问题

即使选模修对以后，结果仍然比更早版本差，这说明：

- `Huber + gamma=0.7 + binary reward`

虽然修复了 value explosion，但也可能把策略推得太保守了。

更具体地说，可能发生了：

- Q 值区分度被压低
- 长期价值链条被砍得过短
- 排序所需的相对偏好结构没有被充分学出来

这就是一个典型的：

- 从“高估崩坏”走向“保守塌缩”

的案例。

## 当前对项目路线的影响

这轮实验之后，项目路线有两个结论应当固定下来。

### 1. 保留 valid ranking 选模

这个改动不应回退。

后续实验默认都应当：

- 每个 epoch 计算 `valid HR@10 / NDCG@10`
- 默认按 `valid_ndcg_at_10` 选 best checkpoint

### 2. 下一步应该调训练目标，而不是回退选模逻辑

当前真正需要继续探索的是：

- `gamma` 是否过低
- `binary reward` 是否过于稀疏
- `adaptive alpha` 是否把系统压得太保守

其中最自然的下一步是：

- 保持 `valid_ndcg_at_10` 选模不变
- 把 `gamma` 从 `0.7` 适度调回 `0.85` 一类的中间值

也就是说，下一步不该再问：

- “要不要继续按 ranking 选模？”

而应该问：

- “在 ranking 选模已经修好的前提下，怎样把训练目标从过度保守拉回合理区间？”

## 当前阶段总结

截至 2026-07-29，这轮实验的最重要意义不是刷新了最终最优成绩，而是完成了一次关键的工程认知修复：

- 训练 loss 最优，不等于推荐效果最优
- 当前项目必须按 `valid ranking metric` 选模

本轮实验应当被正式标记为：

- `选模逻辑修复成功`
- `训练目标仍然偏保守`
- `不是当前主线 best model`

它的价值在于，把我们从“错误的评估闭环”里拉了出来，为下一轮真正调训练目标打下了更可靠的基线。
