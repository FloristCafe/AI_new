# 2026-07-28 CQL 稳定化实验对比与下一步路线

## 本次记录的背景

本次对 `movielens_1m_sasrec_dqn_offline_rl` 做了一轮更稳健的离线强化学习训练，实验目录为：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_stabilized_full_2026_07_28`

这轮实验的目标不是引入新结构，而是先处理上一版 `CQL + Double DQN` 在训练中出现的 Q-value 快速抬升与验证损失恶化问题。

对应对照组为：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_baseline_full_2026_07_27`

## 这次稳定化具体想解决什么

上一版基线已经证明：

- 工程链路是通的
- `SASRec encoder + Q-head + Double DQN + CQL` 可以正常训练
- offline evaluation 也可以稳定产出 `HR@10`、`NDCG@10` 和 reward 指标

但上一版的核心问题也很明显：

- Q 值抬升过快
- `valid_total_loss` 恶化很快
- 最优 epoch 非常靠前
- 后续训练更像是在放大 value estimation 偏差，而不是继续提升推荐质量

因此这轮工作的重点是：

- 降低 Q-head 学习率
- 冻结 encoder
- 加入梯度裁剪
- 监控 `mean_q_value` / `max_q_value`
- 给训练过程增加更明确的数值稳定保护

换句话说，这轮不是“追求更高分”，而是先把离线 RL 的动力学压稳。

## 本次稳定化实验配置层面的关键信息

从 `training_metrics.json` 可以看到，这轮实验的代表性设置包括：

- `q_head_learning_rate = 0.0003`
- `encoder_learning_rate = 0.0`
- `encoder_frozen = true`
- `cql_alpha = 1.0`
- `target_update_interval = 100`
- `grad_clip_norm = 1.0`

这说明当前策略是：

- 保持上游 `SASRec` 表征不动
- 只训练 DQN 的决策头
- 先把问题尽可能压缩为一个更稳定的离线 value learning 问题

## 训练侧结果怎么解读

训练指标文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_stabilized_full_2026_07_28\metrics\training_metrics.json`

从结果上看，这轮实验确实比上一版更稳，但还没有达到“训练越久越好”的状态。

关键现象如下：

- 共完成 `5` 个 epoch
- `best_epoch = 1`
- `selection_metric = valid_total_loss`
- `valid_total_loss` 从 `35.78` 持续升到 `653.81`
- `train_mean_q_value` 从 `0.35` 上升到 `7.51`
- `train_max_q_value` 从 `7.33` 上升到 `45.23`

这说明：

1. 数值稳定性明显好于上一版爆炸式增长的状态。
2. 但验证侧仍然是第一轮最好，后续训练仍在把 Q 值越推越高。
3. 当前训练目标还没有真正把“更稳定”转化成“更好的排序效果”。

所以这轮实验的正确定位是：

- 它是一次有效的稳定性改进实验
- 但还不是新的主线 best run

## 离线评估结果对比

本次稳定化实验评估文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_stabilized_full_2026_07_28\predictions\test_sasrec_dqn_metrics.json`
- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_stabilized_full_2026_07_28\predictions\all_sasrec_dqn_metrics.json`

上一版 baseline 评估文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_baseline_full_2026_07_27\predictions\test_sasrec_dqn_metrics.json`
- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_baseline_full_2026_07_27\predictions\all_sasrec_dqn_metrics.json`

### test split 对比

稳定版：

- `HR@10 = 0.1519867549668874`
- `NDCG@10 = 0.08307095184650086`
- `top1_average_reward = 0.05359271523178849`
- `top1_exact_hit_rate = 0.03360927152317881`

旧版 baseline：

- `HR@10 = 0.16490066225165562`
- `NDCG@10 = 0.09050467639138128`
- `top1_average_reward = 0.05668874172185497`
- `top1_exact_hit_rate = 0.03609271523178808`

test 结论：

- 稳定版在主排序指标上退步
- 稳定版在 top1 reward 上也略差
- 说明“压稳训练”暂时没有带来更强的 next-item 推荐质量

### all split 对比

稳定版：

- `HR@10 = 0.23001421287527574`
- `NDCG@10 = 0.12420156449425275`
- `top1_average_reward = 0.08796039707566615`
- `mean_cumulative_reward_per_user = 14.478062913907255`
- `top1_genre_match_rate = 0.6869445738098855`

旧版 baseline：

- `HR@10 = 0.24660897694456374`
- `NDCG@10 = 0.1307234396137235`
- `top1_average_reward = 0.0872404993520168`
- `mean_cumulative_reward_per_user = 14.359569536423797`
- `top1_genre_match_rate = 0.684202585274737`

all 结论：

- 稳定版的 `HR@10` 与 `NDCG@10` 仍然低于旧版 baseline
- 但 `top1_average_reward`、`mean_cumulative_reward_per_user` 和 `genre_match_rate` 略有提升

这说明当前策略出现了一个很典型的离线 RL 现象：

- 它在 reward 语言下有轻微改善
- 但在推荐排序语言下没有同步改善

也就是说，当前训练目标和最终关注的 recommendation ranking objective 之间，仍然没有完全对齐。

## 这次实验最值得保留的结论

本次实验最重要的价值不是“刷新了最优成绩”，而是明确了下面三件事。

### 1. 稳定化方向是有意义的

这轮训练的 Q 值规模明显比之前温和，说明：

- 冻结 encoder
- 降低学习率
- 梯度裁剪
- Q diagnostics

这些动作不是白做的。

### 2. 仅靠稳定化还不够

虽然 Q 没那么容易炸了，但排序指标还是不够好，说明真正的问题不只是数值不稳定，还包括：

- reward 设计和排序目标不完全一致
- buffer 中的 logged reward 过于退化
- OOD action 高估问题虽然被压住一部分，但没有被根治

### 3. 当前版本不应替代旧 baseline

当前最合理的项目管理结论是：

- `cql_stabilized_full_2026_07_28` 保留为稳定性改进实验
- 不把它升级为新的主线 best experiment
- 当前主线仍应以“继续修正训练目标与离线监督信号”为主

## 项目下一步应该怎么推进

下一步不建议立刻上更多复杂结构，而是先做训练目标和离线经验构造的修正。

优先级最高的方向有三个。

### 1. 先修正 reward / action 监督的退化问题

当前 offline buffer 中的动作本质上仍然是：

- 日志里真实发生过的 next-click action

这导致训练时的即时奖励信息高度单一，容易把问题退化成：

- “把 logged action 的 Q 学高”

而不是：

- “在更完整的动作空间里学会分辨什么动作更值得选”

因此后续应优先考虑：

- 更明确的负动作构造
- 行为约束
- 更强的 action discrimination 信号

### 2. 重新审视 CQL 与 TD 项的力量对比

这轮虽然更稳，但可能也把策略压得偏保守了。

下一轮实验需要重点检查：

- `cql_alpha` 是否过强
- `target_update_interval` 是否还需要更慢
- `gamma` 是否让目标 Q 累积得太激进
- `valid_total_loss` 作为选择指标是否足够贴近最终排序质量

### 3. 把“reward 进步”和“ranking 进步”区分开管理

从这轮结果看，reward 指标和 ranking 指标并没有完全同步。

所以后续实验记录应明确分成两套语言：

- 推荐排序语言：`HR@10`、`NDCG@10`
- 离线 RL 语言：`top1_average_reward`、`mean_cumulative_reward_per_user`

只有当两类指标一起改善时，才说明这个 `SASRec-DQN` 版本真的更强。

## 当前阶段总结

截至 2026-07-28，这个项目已经不再是“能不能跑通”的问题，而是正式进入：

- 离线 RL 目标设计修正
- 保守价值学习稳定化
- reward 与 ranking 指标对齐

的优化阶段。

这轮 `cql_stabilized_full_2026_07_28` 的定位应当是：

- 它证明了稳定化是必要的
- 它没有证明当前训练目标已经足够好
- 它为下一轮更有针对性的离线 RL 修正提供了可靠参照
