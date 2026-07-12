# Micro-RecSim 项目终版报告

## 1. 项目定位

`recommender_mdp_gymnasium` 是一个极简推荐强化学习教学沙盒，目标不是追求工业级效果，而是把推荐系统中的长期决策问题压缩成一个可控、可解释、可复现实验链条。

这个项目最终完成了三件事：

1. 用一个最小环境把推荐问题明确抽象成 MDP。
2. 用 baseline、tabular 和 DQN 系列实验把“为什么需要函数逼近”讲清楚。
3. 用多 seed 实验验证：在当前连续状态环境里，DQN 系列方法可以稳定超过人工启发式基线。

## 2. 环境定义

核心环境文件：

- `projects/recommendation/recommender_mdp_gymnasium/micro_recsim_env.py`

当前环境为 fully observable 教学版本，观测为 11 维：

- 前 5 维：`fatigue`
- 中间 5 维：`preference`
- 最后 1 维：`patience`

动作空间为 5 个离散推荐类目。

点击概率定义为：

\[
p(a \mid s) = preference[a] \cdot (1 - fatigue[a])
\]

奖励规则：

- 点击：`+1`
- 未点击：`0`
- 用户退出：`-10`

这套设计保留了推荐 RL 最核心的长期依赖：

- 当前推荐会抬高对应类目的疲劳度
- 疲劳度会压低未来点击率
- 未点击会持续消耗耐心
- 因此策略必须在“眼前点击”和“未来疲劳”之间做权衡

## 3. Baseline 体系与结论

Baseline 文件：

- `projects/recommendation/recommender_mdp_gymnasium/run_baselines.py`

当前基线包括：

- `random`
- `always_same_0`
- `round_robin`
- `least_fatigue`
- `myopic_oracle_preference_greedy`
- `observable_click_greedy`

代表性结果：

- `random = -2.265`
- `always_same_0 = -8.235`
- `round_robin = -0.630`
- `least_fatigue = -0.725`
- `myopic_oracle_preference_greedy = -0.480`
- `observable_click_greedy = -0.480`

这一阶段的关键判断有三条：

1. `always_same_0` 最差，说明 fatigue 机制是有效的，重复轰炸同类目会快速杀死 session。
2. `round_robin` 和 `least_fatigue` 已经非常强，说明在这个微型环境里，疲劳控制比精细个性化更主导。
3. `myopic_oracle_preference_greedy` 只是单步贪心上界，不是长期最优上界，因此后续 RL 超过它是合理现象。

## 4. Tabular Q-learning 分支结论

相关文件：

- `projects/recommendation/recommender_mdp_gymnasium/train_q_learning.py`

Tabular 分支的主要作用不是产出最优结果，而是用失败证明方法边界。

三版 encoder 的结论如下：

- `v1`：状态压缩过度，结果接近随机，best 大约在 `-2.27`
- `v2`：加入更多疲劳与偏好排序信息后明显提升，best 大约在 `-1.696`
- `v3`：状态数从 `66825` 膨胀到 `601425`，出现状态爆炸，best 退化到大约 `-2.05`

这个分支最终说明：

- 连续状态问题中，手工离散化非常容易在“信息太少”和“状态爆炸”之间来回撞墙
- tabular Q-learning 在当前任务中是教学用失败路径，而不是主线解决方案
- 项目的真正转折点不是“把 encoder 调得更巧”，而是“承认必须 перейти 到函数逼近”

## 5. DQN 主线实验

相关文件：

- `projects/recommendation/recommender_mdp_gymnasium/dqn_agent.py`
- `projects/recommendation/recommender_mdp_gymnasium/train_dqn.py`
- `projects/recommendation/recommender_mdp_gymnasium/run_dqn_multiseed.py`

当前 Q-network 为：

- `11 -> 128 -> 128 -> 5`

训练组件包括：

- replay buffer
- target network
- Huber loss
- Adam
- epsilon-greedy

主线调参过程中的关键节点：

1. 初版 DQN 已经优于 tabular，best 大约在 `-1.320`
2. 将学习率从 `1e-3` 降到 `5e-4` 后，best 提升到大约 `-0.760`
3. `SiLU` 不优于 `ReLU`
4. `target_sync_interval=500` 不如 `200`
5. 学习率进一步降到 `1e-4` 后，单 seed best 可提升到 `-0.190` 和 `-0.104`

这说明在当前连续状态环境中：

- DQN 不只是比 tabular 更合适
- 它已经能稳定超过启发式 baseline
- 它学到的是长期 fatigue 管理策略，而不只是单步点击贪心

## 6. 标准 DQN 的多 seed 结果

在主配置：

- `learning_rate = 1e-4`
- `activation = relu`
- `target_sync_interval = 200`
- `early_stop_patience = 5`

下，多 seed 结果为：

- `seed 7`: best `-0.034`, final `-0.474`
- `seed 42`: best `-0.042`, final `-0.316`
- `seed 123`: best `-0.220`, final `-0.400`

聚合结果：

- `best_reward_mean = -0.099`
- `best_reward_std = 0.086`
- `best_clicks_mean = 9.901`
- `best_clicks_std = 0.086`
- `final_reward_mean = -0.397`
- `final_clicks_mean = 9.603`

这一阶段的结论是：

- DQN 已经证明“能学会”
- 主要问题不再是学习能力，而是 `best checkpoint` 和 `final checkpoint` 之间的明显断崖

## 7. Double DQN 与学习率退火实验

### 7.1 Double DQN

在标准 DQN 基础上加入 Double DQN 后：

- `best_reward_mean = -0.073`
- `best_reward_std = 0.042`
- `best_clicks_mean = 9.927`
- `final_reward_mean = -0.533`
- `final_clicks_mean = 9.467`

结论：

- Double DQN 显著提升了峰值表现和跨 seed 稳定性
- 但没有解决晚期退化，反而让 `best -> final` 的落差仍然很明显

也就是说，Double DQN 只解决了“Q 值高估”的一部分问题，没有消除训练后期的数据同质化与策略漂移。

### 7.2 Double DQN + late-stage LR decay

进一步加入学习率退火，配置为：

- `learning_rate = 1e-4`
- `learning_rate_end = 1e-5`
- `lr_decay_start_fraction = 0.7`

多 seed 结果：

- `seed 7`: best `-0.008`, final `-0.290`
- `seed 42`: best `-0.020`, final `-0.442`
- `seed 123`: best `-0.178`, final `-0.592`

聚合结果：

- `best_reward_mean = -0.069`
- `best_reward_std = 0.077`
- `best_clicks_mean = 9.931`
- `final_reward_mean = -0.441`
- `final_clicks_mean = 9.559`

结论：

- 相比“只有 Double DQN”，学习率退火明显改善了 final checkpoint
- `best -> final` gap 被压缩，说明后期崩塌被部分缓解
- 但问题没有被完全根治，尤其 `seed 123` 仍有明显退化

### 7.3 另一组退火消融

尝试更晚开始、衰减更浅的配置：

- `learning_rate_end = 3e-5`
- `lr_decay_start_fraction = 0.8`

多 seed 结果：

- `seed 7`: best `-0.148`, final `-0.182`
- `seed 42`: best `-0.784`, final `-0.850`
- `seed 123`: best `-0.048`, final `-0.258`

聚合结果：

- `best_reward_mean = -0.327`
- `final_reward_mean = -0.430`

结论：

- 虽然 final 均值略有改善空间，但 best 性能显著下滑
- 并且出现了 seed 42 明显掉队的情况
- 因此该配置不如 `0.7 -> 1e-5`，不应作为主线

## 8. 项目最终结论

到项目结束时，可以比较稳健地给出以下结论：

1. `Micro-RecSim` 成功把推荐问题压缩成一个最小可讲清楚的 MDP 教学沙盒。
2. baseline 结果证明：在当前环境里，疲劳控制是比单步个性化更主导的因素。
3. tabular Q-learning 分支系统性失败，充分说明连续状态推荐场景下离散化方法的局限。
4. DQN 主线已经稳定超过当前人工启发式 baseline，说明 RL 在该沙盒中确实学到了更优的长期策略。
5. Double DQN 提升了峰值策略质量与跨 seed 稳定性。
6. late-stage learning-rate decay 在 Double DQN 基础上进一步缓解了后期漂移，但没有完全消除 seed-sensitive instability。

用一句话概括：

**这个项目已经完成了从“环境搭建”到“RL 优于启发式基线”的完整闭环，并且清楚暴露了深度强化学习训练中 best checkpoint 与 final checkpoint 脱钩的核心工程问题。**

## 9. 最终推荐配置

如果把本项目作为阶段终版结果，推荐保留以下训练协议：

- `double_dqn = true`
- `learning_rate = 1e-4`
- `learning_rate_end = 1e-5`
- `lr_decay_start_fraction = 0.7`
- `activation = relu`
- `target_sync_interval = 200`
- `early_stop_patience = 5`

同时在汇报中应明确：

- 正式结果优先报告 `best checkpoint`
- `final checkpoint` 只用于说明训练后期仍存在稳定性问题

## 10. 项目收尾定位

这个项目到此可以视为阶段性完成，原因不是“所有问题都解决了”，而是：

- 环境、baseline、失败路径、成功路径都已经讲清楚
- 关键实验现象有足够证据支撑
- 继续微调超参的边际收益已经明显下降

因此，当前最合理的收尾方式不是继续盲目调参，而是把它作为一个完整的 RL & RecSys 教学项目归档。

