# 2026-08-04 CE 权重扫描收口与 strongest baseline 封板

## 本次记录的目标

这篇笔记用于正式收束当前阶段最重要的一轮精修实验：

- 在 `gamma=0.9`
- `Huber TD`
- `binary reward`
- `valid_ndcg_at_10` 选模
- `RL + CE` 联合损失

这条主线下，对 `ce_regularization_weight` 做窄范围扫描，目标是找出当前 strongest baseline 的最优点。

本轮重点比较了三个点：

- `ce = 0.3`
- `ce = 0.4`
- `ce = 0.5`

## 为什么这一轮扫描重要

在前一阶段，我们已经确认：

- 单纯 `RL-only` 容易丢掉排序感
- 加回 `CE regularization` 是有效的
- `ce = 0.1` 已经明显优于无 CE 版本

但那时还没有解决一个关键问题：

- `CE` 到底应该加多强？

如果 `CE` 太弱，排序边界守不住；
如果 `CE` 太强，又可能把系统重新推回“近似监督学习”，让 RL 目标失去价值。

所以这一轮的意义，不是普通调参，而是：

- 为当前 strongest baseline 找到一个更可信的训练目标平衡点

## 三轮实验的定位

### 1. `ce = 0.3`

对应实验目录：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg03`

它的特点是：

- 相比 `ce = 0.1`，继续提升了验证排序指标
- `all` 轨迹层面表现很强
- 是第一个明确把 `RL + CE` 主线再往前推的版本

### 2. `ce = 0.4`

对应实验目录：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg04`

它的特点是：

- 比 `ce = 0.3` 的单步 `test` 更强
- 但在 `all` 轨迹层面反而没有更优
- 呈现出一个中间态，而不是最优平衡点

### 3. `ce = 0.5`

对应实验目录：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05`

它的特点是：

- 当前 `test HR@10 / test NDCG@10` 最高
- `best_valid_ndcg_at_10` 也是当前最高
- `best_epoch` 不再卡死在第 1 轮，而是推进到了 `epoch 10`

这说明 `CE` 强度继续上调后，不只是排序结果更好，训练过程本身也比以前更健康。

## 三轮结果并排对比

### test split

`ce = 0.3`

- `HR@10 = 0.16142384105960264`
- `NDCG@10 = 0.08980818238890449`
- `top1_exact_hit_rate = 0.03741721854304636`

`ce = 0.4`

- `HR@10 = 0.1849337748344371`
- `NDCG@10 = 0.09816852838509364`
- `top1_exact_hit_rate = 0.034271523178807946`

`ce = 0.5`

- `HR@10 = 0.19039735099337748`
- `NDCG@10 = 0.1021792974824788`
- `top1_exact_hit_rate = 0.035927152317880795`

### all split

`ce = 0.3`

- `HR@10 = 0.2359417764987643`
- `NDCG@10 = 0.127011027764501`
- `top1_exact_hit_rate = 0.04638446783192797`
- `mean_cumulative_reward_per_user = 7.63476821192053`

`ce = 0.4`

- `HR@10 = 0.22594347641095228`
- `NDCG@10 = 0.1160737573016917`
- `top1_exact_hit_rate = 0.03725322354649964`
- `mean_cumulative_reward_per_user = 6.131788079470199`

`ce = 0.5`

- `HR@10 = 0.24074176523307406`
- `NDCG@10 = 0.12533210207531637`
- `top1_exact_hit_rate = 0.04175648204681498`
- `mean_cumulative_reward_per_user = 6.873013245033112`

## 这组扫描怎么解读

这三轮结果揭示了一个很明确的关系。

### 1. `ce = 0.4` 不是最优平衡点

它虽然比 `0.3` 在单步 `test` 上更强，但：

- `all HR@10`
- `all NDCG@10`
- `mean cumulative reward`

都没有优于 `0.3` 或 `0.5`。

也就是说，`0.4` 处在一个中间带，但不是甜点。

### 2. `ce = 0.3` 更偏全轨迹质量

和 `0.5` 比：

- `all NDCG@10` 更高
- `mean cumulative reward` 更高

这说明 `0.3` 更偏向：

- 维持整条轨迹上的长期一致性

### 3. `ce = 0.5` 更偏 test 排序最优

`0.5` 的突出优点是：

- `test HR@10` 最高
- `test NDCG@10` 最高
- `best_valid_ndcg_at_10` 最高
- `all HR@10` 也最高

它虽然没有在所有 `all` 指标上绝对统治，但从当前项目最核心的评估目标来看：

- 单步 next-item 排序能力

它已经是最强的。

## 训练侧的结构性变化

这一轮扫描不仅比较了最终分数，还暴露了一个重要的训练动力学变化。

在更早的很多实验里，我们频繁看到：

- `best_epoch = 1`

这说明训练目标和排序目标之间存在明显张力，模型一开始最好，后面越训越偏。

但在 `ce = 0.5` 这轮里：

- `best_epoch = 10`
- `best_valid_ndcg_at_10 = 0.11202340464208771`

这意味着更强的 `CE regularization` 并不只是把 test 分数往上推了一点，它还在改变训练过程本身：

- 训练不再只在第一轮有价值
- 排序目标在后续 epoch 中可以继续受益

这是一种非常重要的结构性改进。

## strongest baseline 应该封板为哪一版

综合当前所有已跑结果，最合理的封板结论是：

- 当前 strongest baseline 记为  
  `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05`

理由如下：

### 1. 它在最核心的 test 指标上最好

- `test HR@10` 最高
- `test NDCG@10` 最高

### 2. 它的 valid ranking 指标也最好

- `best_valid_ndcg_at_10` 最高

这说明不是 test 偶然好，而是验证集选模本身就在支持它。

### 3. 它让训练过程比以前更健康

- `best_epoch` 不再固定死在第 1 轮

这意味着当前训练目标终于开始具备“可持续优化”的迹象。

## 但为什么仍然保留 ce=0.3 的价值

虽然 strongest baseline 暂时封板给 `ce=0.5`，但 `ce=0.3` 仍然非常重要，不应被简单遗忘。

因为它在：

- `all NDCG@10`
- `mean cumulative reward per user`

上依然表现很强，说明它更像一个：

- 偏轨迹一致性 / 偏长期质量

的版本。

后续如果项目需要强调：

- 长轨迹回放质量
- 离线累计收益

那么 `ce=0.3` 仍然是非常有价值的对照。

## 当前阶段正式结论

截至 **2026 年 8 月 4 日**，本轮 `CE` 权重扫描可以正式收口为：

- `ce = 0.4` 不是最优点
- `ce = 0.3` 偏全轨迹质量
- `ce = 0.5` 偏 test 排序最优，且训练更健康

因此当前 strongest baseline 封板为：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05`

## 项目下一步应该做什么

既然 strongest baseline 已经基本收口，后续就不应该继续盲目扫 `ce` 了。

下一步最合理的是两件事：

### 1. 多 seed 稳定性验证

至少补：

- `seed = 7`
- `seed = 42`
- `seed = 2026`

这样才能判断当前 strongest baseline 是不是稳定有效，而不是单 seed 偶然结果。

### 2. 正式进入方法升级阶段

后续最值得做的创新方向，不再是单纯调参，而是：

- `BPR / pairwise ranking loss`
- `CE weight` 动态调度
- `behavior-constrained offline RL`，如 `IQL / AWAC`

也就是说，这轮扫描的作用不是无穷无尽地继续扫参，而是：

- 把 strongest baseline 封板
- 为后续方法创新清理地基
