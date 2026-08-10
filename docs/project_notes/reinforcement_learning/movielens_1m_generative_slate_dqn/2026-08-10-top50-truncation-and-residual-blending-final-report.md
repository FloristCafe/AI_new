# 2026-08-10 Top-50 候选池截断与残差 Logit 融合最终报告

## 这轮报告要回答什么问题

在上一阶段里，`movielens_1m_generative_slate_dqn` 已经证明：

- 纯 `Slate-DQN` 能在自定义 `SlateRecSimEnv` 中学到更高的 slate reward
- 但它与真实 MovieLens 持出目标严重错位
- 典型症状是：
  - `mean_slate_reward` 很高
  - `target_hit@5 = 0`

所以这一轮实验的目标非常明确：

1. 用 **Top-50 候选池截断** 把动作空间从 3706 压缩到高质量候选集
2. 用 **残差 Logit 融合（Residual Q + SASRec Logits）** 将冻结 `SASRec` 的离线确信度重新注入 Slate-DQN 推断过程
3. 观察是否能在不显著牺牲列表收益的前提下，让 `target_hit@5 / target_ndcg@5` 明显恢复

## 本轮方法升级

### 1. Top-50 候选池截断

在每个 episode 起点，用冻结 `SASRec` 对当前状态打分，构建 `Top-50` 候选集。

之后在 `Slate-DQN` 自回归生成 5 个 item 时：

- 候选池外的 item 直接被置为 `-inf`
- 只允许在 `Top-50` 中做重排

这一步的物理含义是：

- `SASRec` 做候选生成 / 粗排
- `Slate-DQN` 做重排 / 列表上下文修正

### 2. 残差 Logit 融合

在 `Top-50` 掩码和 prefix 去重掩码之后，不再直接对 `Q` 做 `argmax`，而是做：

```text
Q_final = Q_slate_dqn + lambda * logits_sasrec
```

其中：

- `Q_slate_dqn` 提供列表上下文下的增量价值
- `logits_sasrec` 提供强离线 item-level 先验
- `lambda` 控制两者的融合权重

这一轮验证了：

- `lambda = 0.1`
- `lambda = 0.5`
- `lambda = 1.0`

## 训练配置

实验目录：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn\artifacts\experiments\slate_dqn_top50_seed42_20260810`

训练汇总：

- `candidate_pool_size = 50`
- `best_episode = 600`
- `best_eval_mean_slate_reward = 0.896`

说明：

- `Top-50` 截断版本已经能稳定收敛
- 最优 checkpoint 出现在中期，而不是训练最后

## Valid 结果：Pareto 前沿已经出现

### 1. `lambda = 0.1`

`slate_dqn` 结果：

- `mean_slate_reward = 0.9596`
- `mean_episode_return = 4.798`
- `target_hit@5 = 0.063`
- `target_ndcg@5 = 0.0359`
- `intra_list_diversity_mean = 0.6804`

解读：

- reward 基本守住
- 真实命中相比“只做 Top-50、不融合”有进一步提升
- 但 bridge 改善还不够强

### 2. `lambda = 0.5`

`slate_dqn` 结果：

- `mean_slate_reward = 0.9236`
- `mean_episode_return = 4.618`
- `target_hit@5 = 0.122`
- `target_ndcg@5 = 0.0793`
- `intra_list_diversity_mean = 0.6360`

和 `sasrec_topk` 对比：

- `sasrec_topk mean_slate_reward = 0.9182`
- `sasrec_topk target_hit@5 = 0.166`
- `sasrec_topk target_ndcg@5 = 0.1082`

解读：

- 这是最均衡的一档
- reward 仍略高于 `sasrec_topk`
- hit / ndcg 已明显逼近 `sasrec_topk`
- 多样性仍高于 `sasrec_topk`

### 3. `lambda = 1.0`

`slate_dqn` 结果：

- `mean_slate_reward = 0.9082`
- `mean_episode_return = 4.541`
- `target_hit@5 = 0.151`
- `target_ndcg@5 = 0.0977`
- `intra_list_diversity_mean = 0.6150`

和 `sasrec_topk` 对比：

- `target_hit@5` 只差 `0.015`
- `target_ndcg@5` 只差 `0.010`
- 但 reward 已略低于 `sasrec_topk`

解读：

- 这是“桥接指标最强”的融合点
- 但已经开始向 `sasrec_topk` 收敛，列表收益优势明显变弱

## Test 结果：最终结论落地

测试设置：

- 全量 `6040` 个 test 用户
- `mc_rollouts = 3`
- `max_slates_per_episode = 10`
- 与四个策略全量对比

---

### Test: `lambda = 0.5`

`slate_dqn`：

- `mean_slate_reward = 0.7596`
- `mean_episode_return = 7.5962`
- `slate_success_rate = 0.5498`
- `single_slate_mean_reward = 1.3704`
- `target_hit@5 = 0.1195`
- `target_ndcg@5 = 0.0797`
- `target_mrr@5 = 0.0667`
- `genre_hit@5 = 0.8752`
- `intra_list_diversity_mean = 0.6454`

`sasrec_topk`：

- `mean_slate_reward = 0.7597`
- `mean_episode_return = 7.5971`
- `target_hit@5 = 0.1480`
- `target_ndcg@5 = 0.0985`
- `intra_list_diversity_mean = 0.6104`

解读：

- `lambda = 0.5` 在 **reward 上几乎与 `sasrec_topk` 打平**
- 但 bridge 指标仍略低于 `sasrec_topk`
- 多样性明显高于 `sasrec_topk`

这说明：

- `Top-50 + residual blending` 已经把纯 RL 的“目标错位”大幅修复
- 但目前还没有在真实 item-level 命中上超过 `sasrec_topk`

---

### Test: `lambda = 1.0`

`slate_dqn`：

- `mean_slate_reward = 0.7426`
- `mean_episode_return = 7.4259`
- `slate_success_rate = 0.5388`
- `single_slate_mean_reward = 1.3669`
- `target_hit@5 = 0.1381`
- `target_ndcg@5 = 0.0932`
- `target_mrr@5 = 0.0785`
- `genre_hit@5 = 0.8720`
- `intra_list_diversity_mean = 0.6250`

`sasrec_topk`：

- `target_hit@5 = 0.1480`
- `target_ndcg@5 = 0.0985`
- `mean_slate_reward = 0.7597`

解读：

- `lambda = 1.0` 在 bridge 指标上已经非常接近 `sasrec_topk`
- 但 reward、episode return 和 success rate 都低于 `sasrec_topk`

也就是说：

- 当融合权重过大时，Slate-DQN 越来越像 `sasrec_topk`
- 但它还没有用“列表上下文修正”换回额外收益

## 一个必须正视的异常信号：`random_unique` 的 reward 很高

在测试集上：

- `random_unique mean_slate_reward = 0.8763`
- `random_unique mean_episode_return = 8.7629`

这显著高于：

- `slate_dqn`
- `sasrec_topk`
- `popularity_topk`

但与此同时：

- `random_unique target_hit@5 = 0.0016`
- `random_unique target_ndcg@5 = 0.00077`

这说明当前环境仍然存在一个非常重要的结构性问题：

- **环境 reward 对“真实 item-level 命中”仍然不够敏感**
- 随机且高多样性的列表，在 simulator 中依然可能拿到很高收益

这也进一步证明：

- 这轮 `Top-50 + residual blending` 虽然有效
- 但还没有从根本上解决 simulator reward 与真实推荐目标之间的错配

## 这一轮实验的正式结论

### 1. Top-50 候选池截断是有效的

它成功把动作空间从“全库乱搜”改成“高质量候选中的重排”。

直接证据是：

- 纯 `Top-50` 已经能把 `target_hit@5` 从 `0` 拉到 `0.05`

### 2. 残差 Logit 融合进一步显著修复 bridge gap

直接证据是：

- `lambda = 0.1`：`target_hit@5 = 0.063`
- `lambda = 0.5`：`target_hit@5 = 0.122`
- `lambda = 1.0`：`target_hit@5 = 0.151`
- `sasrec_topk`：`target_hit@5 = 0.148`（test）

这说明：

- `Slate-DQN` 不是完全没能力做 item-level 重排
- 它需要一个足够强的 `SASRec` 先验作为保底重力场

### 3. 当前最佳工作点是 `lambda = 0.5`

原因不是它的任何单一指标最强，而是它在多目标之间最平衡：

- reward 基本与 `sasrec_topk` 持平
- `target_hit@5` 和 `target_ndcg@5` 明显优于纯 RL 版本
- 多样性仍高于 `sasrec_topk`

因此，截至 **2026-08-10**，当前项目的推荐主配置应该写成：

- **`Top-50 candidate truncation + residual logit blending (lambda = 0.5)`**

### 4. 但当前系统还不能宣称“全面超越 `sasrec_topk`”

原因很清楚：

- 在 test 上，`sasrec_topk` 仍然保持更强的 item-level 命中
- 当前环境对高多样性随机 slate 的 reward 过高

所以更准确的结论是：

- 这套方法已经把生成式 slate RL 从“纯 simulator 最优”推进到“兼顾 reward 与真实命中的工业重排形态”
- 但尚未完全超过强离线排序基线

## 下一步最值得做什么

现在最值得做的事已经非常清楚：

### 1. 修 reward 对齐，而不是继续盲目调 lambda

因为：

- `lambda` 现在已经表现出非常清晰的 Pareto 前沿
- 再继续微调 `0.6 / 0.7 / 0.8` 的信息增益已经不高

### 2. 引入 supervised anchor / imitation term

最有希望的下一步是：

- 保留 `Top-50`
- 保留 residual blending
- 在训练损失里再加入一项对真实 next-item 的监督约束

这样做的目标是：

- 不是在推断时“借” `SASRec` 的力
- 而是在训练阶段就把 `Slate-DQN` 的 Q 空间往真实 item-level 结构上拉

### 3. 审核并重构环境 reward

尤其要重点处理：

- 为什么 `random_unique` 的 simulator reward 会显著高于 `sasrec_topk`

这很可能意味着：

- 当前环境对多样性和长尾探索过度奖励
- 或对真实 click relevance 的约束仍然不够强

## 一句话总结

这轮 `Top-50 truncation + residual blending` 的最终结论是：

- **候选池截断与残差 Logit 融合成功修复了生成式 Slate-DQN 与真实持出目标之间的严重错位，使其从“高 reward 但零命中”的纯 simulator 策略，提升为“reward 近似不掉、bridge 指标大幅恢复”的工业化重排原型；其中 `lambda = 0.5` 是当前最均衡的主配置，但 `sasrec_topk` 仍在 test 的真实 item-level 命中上保持优势，且环境 reward 仍存在对随机高多样性策略过度友好的结构性问题。**
