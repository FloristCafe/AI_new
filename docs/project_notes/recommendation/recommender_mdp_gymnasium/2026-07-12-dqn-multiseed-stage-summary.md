# Micro-RecSim DQN 多 Seed 阶段总结

## 这份总结记录什么

这份笔记用于总结 `recommender_mdp_gymnasium` 当前阶段最重要的结果：

- 最小推荐 MDP 环境已经搭好
- baseline 已经形成稳定对照系
- tabular Q-learning 已验证为过渡性失败路线
- DQN 已经在多 seed 下稳定优于 heuristic baseline

这意味着项目已经从“做出一个能跑的 RL 沙盒”推进到“拿到一个可汇报的 RL 结果”。

## 项目当前目标的变化

项目最初目标是：

- 搭建一个极简推荐强化学习沙盒
- 让环境具备 fatigue、preference、patience 和 reward 的基本动力学
- 验证推荐问题能否被自然抽象成 MDP

当前目标已经变成：

- 证明在这个连续状态推荐环境中，DQN 能否稳定学出优于人工启发式的长期策略

目前这个目标已经基本达成。

## 当前环境定义回顾

当前环境文件：

- `projects/recommendation/recommender_mdp_gymnasium/micro_recsim_env.py`

当前 fully observable observation 为 11 维：

- 前 5 维：fatigue
- 中间 5 维：preference
- 最后 1 维：patience

动作空间为 5 个推荐类目。

点击概率定义为：

\[
p(a \mid s) = preference[a] \cdot (1 - fatigue[a])
\]

奖励规则：

- 点击 `+1`
- 未点击 `0`
- 用户退出 `-10`

这个环境的关键难点不在高维输入，而在于：

- 当前动作会改变未来 fatigue
- fatigue 会影响未来点击率
- 因此这是一个标准的长期序列决策问题，而不是单步点击预测问题

## Baseline 体系已经完成

当前 baseline 文件：

- `projects/recommendation/recommender_mdp_gymnasium/run_baselines.py`

已完成的 baseline 包括：

- `random`
- `always_same_0`
- `round_robin`
- `least_fatigue`
- `myopic_oracle_preference_greedy`
- `observable_click_greedy`

代表性结果为：

- `random = -2.265`
- `round_robin = -0.630`
- `least_fatigue = -0.725`
- `myopic_oracle_preference_greedy = -0.480`
- `observable_click_greedy = -0.480`

这里需要强调：

- 当前 oracle 不是长期最优上界
- 它只是一步即时点击率贪心
- 因此后续 RL 超过它是合理现象，不是计算错误

## Tabular Q-learning 分支的结论

当前 tabular 尝试文件：

- `projects/recommendation/recommender_mdp_gymnasium/train_q_learning.py`

这个分支的主要价值已经不是继续优化，而是提供了一条很有教育意义的失败证据链。

### encoder v1

状态压缩过强，结果几乎接近 random：

- best 大约在 `-2.27`

说明状态信息严重不足。

### encoder v2

保留了更多疲劳结构和偏好排序信息：

- best 大约在 `-1.696`

说明 agent 开始学到一点东西，但仍然明显弱于 heuristic baseline。

### encoder v3

继续增加 preference 强度与 gap 后：

- best 反而退化到大约 `-2.05`

说明状态空间爆炸导致覆盖不足。

### tabular 分支当前结论

在这个连续状态推荐环境里：

- 信息过少会学不到
- 信息过多会状态爆炸
- tabular Q-learning 对离散化设计过于敏感

因此 tabular 分支当前应视为：

- 一个必要的过渡实验
- 一个说明“为什么需要函数逼近”的证据

而不应再作为主线继续投入。

## DQN 为什么成为主线

当前 DQN 相关文件：

- `projects/recommendation/recommender_mdp_gymnasium/dqn_agent.py`
- `projects/recommendation/recommender_mdp_gymnasium/train_dqn.py`
- `projects/recommendation/recommender_mdp_gymnasium/run_dqn_multiseed.py`

选择 DQN 的原因很明确：

- observation 是连续浮点向量
- heuristic baseline 直接利用连续值
- Q-table 需要人为离散化，容易发生 state aliasing
- DQN 可以直接在连续空间中做函数逼近

当前网络结构：

- 输入维度：11
- 隐藏层：`128 -> 128`
- 输出维度：5
- 激活函数：当前主线为 `ReLU`

并配套：

- replay buffer
- target network
- Huber loss
- Adam
- epsilon-greedy

## DQN 调参与结果演化

### 初版 DQN

较早一版已优于 tabular：

- best 大约在 `-1.320`

说明连续状态函数逼近方向正确。

### 调整到 `learning_rate=5e-4`

表现显著改善：

- best 大约在 `-0.760`

已经逼近 `least_fatigue = -0.725`。

### 尝试 `SiLU`

没有优于 ReLU：

- best 大约在 `-0.796`

当前不作为主线。

### 尝试更慢的 `target_sync_interval=500`

结果变差：

- best 大约在 `-0.968`

说明在这个小环境里，target 更新过慢会损害性能。

### 调整到 `learning_rate=1e-4`

这是当前最关键的突破点。

代表性单次结果：

- `seed 42`：best 大约 `-0.190`
- `seed 7`：best 大约 `-0.104`

说明 DQN 已经不只是接近 heuristic，而是开始明显超过 heuristic。

## 多 Seed 稳定性验证结果

当前多 seed 汇总脚本：

- `projects/recommendation/recommender_mdp_gymnasium/run_dqn_multiseed.py`

当前主配置：

- `learning_rate = 1e-4`
- `activation = relu`
- `target_sync_interval = 200`
- `eval_every = 100`
- `early_stop_patience = 5`

多 seed 汇总结果为：

- `seed 7`
  - best reward `-0.034`
  - best clicks `9.966`

- `seed 42`
  - best reward `-0.042`
  - best clicks `9.958`

- `seed 123`
  - best reward `-0.220`
  - best clicks `9.780`

聚合统计：

- `best_reward_mean = -0.099`
- `best_reward_std = 0.086`
- `best_clicks_mean = 9.901`
- `best_clicks_std = 0.086`

同时 final checkpoint 仍然显著弱于 best checkpoint：

- `final_reward_mean = -0.397`
- `final_clicks_mean = 9.603`

## 当前最重要的项目结论

到这一阶段，可以比较稳定地给出以下结论：

1. 在这个连续状态推荐环境里，DQN 明显优于 tabular Q-learning
2. DQN 已经稳定优于当前人工 heuristic baseline
3. baseline 中的 myopic oracle 不是长期策略上界，DQN 超过它是合理现象
4. 当前核心问题不再是“学不会”，而是“如何保住最好策略”

也就是说，项目主矛盾已经从：

- RL 能否在这个沙盒里起作用

切换成：

- 如何让 DQN 的 best policy 更稳定地被保留和复现

## 当前最优主配置

当前建议固定的主配置为：

- `learning_rate = 1e-4`
- `activation = relu`
- `target_sync_interval = 200`
- `eval_every = 100`
- `early_stop_patience = 5`

并且当前阶段一律应：

- 汇报 `best checkpoint`
- 不以 final checkpoint 作为主结果

## 当前阶段项目定位

当前项目已经具备以下阶段性成果：

- 环境搭建完成
- baseline 体系完成
- tabular 失败原因已明确
- DQN 主线验证成功
- 多 seed 稳定性验证完成

这意味着它已经不是一个“练手脚本”，而是一个结构完整、可讲清问题、可复现实验结果的小型 RL & RecSys 教学项目。

## 当前阶段最自然的下一步

下一步不建议大幅换模型，而建议按以下方向推进：

1. 固化训练协议
   - 继续保留 best checkpoint
   - 继续使用多 seed 汇总

2. 训练稳定性优化
   - 可以尝试更系统的 early stopping 策略
   - 可以考虑 Double DQN 作为下一步增强

3. 环境复杂度提升
   - 把当前 fully observable 教学环境扩展回更真实的部分可观测设定
   - 或加入更复杂的用户动态偏好变化

## 当前阶段一句话总结

当前 `Micro-RecSim` 项目已经完成了一个重要转折：

- 从“做出一个最小推荐 MDP”
- 走到了“用 DQN 稳定学出优于启发式的长期推荐策略”

这是一个可以被视为阶段性成功的强化学习项目结果。
