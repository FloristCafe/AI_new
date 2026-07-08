# Micro-RecSim DQN 转向与当前策略笔记

## 当前这份笔记记录什么

这份笔记用于记录 `recommender_mdp_gymnasium` 在完成环境和 baseline 后，如何从 `tabular Q-learning` 的离散化死胡同，转向 `DQN` 并得到当前最强结果。

它不是环境设计说明，也不是最终总结，而是一个阶段性策略笔记，回答三个问题：

- 当前到底试了什么
- 哪些方向已经被排除
- 现在主线该怎么继续推进

## 当前项目状态

当前项目已经不再停留在“构造最小推荐环境”的阶段，而是进入了第一个真正的 RL agent 验证阶段。

当前代码主线包括：

- `micro_recsim_env.py`
  - fully observable 的推荐环境
  - 11 维 observation：`fatigue(5) + preference(5) + patience(1)`

- `run_baselines.py`
  - baseline 评估脚手架

- `train_q_learning.py`
  - tabular Q-learning 尝试线

- `dqn_agent.py`
  - Q-network + replay buffer + target network

- `train_dqn.py`
  - DQN 训练与评估脚手架

## Baseline 结论回顾

当前 baseline 的代表性结果大致为：

- `random = -2.265`
- `round_robin = -0.630`
- `least_fatigue = -0.725`
- `oracle_preference_greedy = -0.480`
- `observable_click_greedy = -0.480`

这里需要明确一件事：

- 当前这个 `oracle_preference_greedy` 不是全局理论最优
- 它只是“一步即时点击率贪心”的 myopic oracle

因此后续 RL 超过它并不矛盾，反而说明 RL 学会了长期权衡。

## 为什么 Tabular Q-learning 被放弃

### 1. encoder v1：信息太少

第一版状态压缩过强，Q-learning 基本只能达到接近 `random` 的水平。

代表性结果：

- best 大约在 `-2.27` 附近

这说明状态表示严重缺信息。

### 2. encoder v2：有提升，但仍明显弱于 heuristic

第二版状态编码保留了：

- 5 个 fatigue bucket
- `best_preference_action`
- `second_preference_action`
- `patience_bucket`

代表性结果：

- best 大约在 `-1.696`

这说明：

- 状态信息增加后，Q-learning 确实学到了一些东西
- 但仍明显打不过 `round_robin / least_fatigue`

### 3. encoder v3：状态空间爆炸

第三版又加入了：

- `best_preference_value_bucket`
- `preference_gap_bucket`

结果反而变差：

- best 大约在 `-2.05`

主要原因不是新信息本身无效，而是状态空间从 `66825` 膨胀到 `601425`，导致 tabular 方法覆盖不足。

### 4. 当前对 tabular 线的结论

当前结论已经比较明确：

- 状态信息过少：学不到
- 状态信息过多：状态爆炸
- 表格法对连续状态的离散化过于敏感

所以当前项目已经不再继续深挖 `Q-table`，而是将 tabular 分支视为：

- 一个重要的失败实验
- 一个说明“为什么要用函数逼近”的证据链

## DQN 为什么成为当前主线

转向 DQN 的原因很直接：

- observation 是连续浮点向量
- heuristic baseline 直接使用连续值做运算
- tabular 方法必须手工离散化，容易产生 state aliasing
- DQN 可以直接对连续状态做函数逼近

当前 DQN 采用的第一版结构是：

- 输入维度：11
- 输出动作数：5
- 网络：`11 -> 128 -> 128 -> 5`
- 激活函数：`ReLU` 或 `SiLU`
- replay buffer
- target network
- Huber loss
- Adam

## 当前 DQN 实验结果

### 1. 初版 DQN

较早一版 DQN 已经优于 tabular：

- best 大约在 `-1.320`

这证明：

- 连续状态 + 函数逼近 的方向是对的

### 2. 学习率降到 `5e-4`

将学习率从 `1e-3` 降到 `5e-4` 后，结果明显改善：

- best 大约达到 `-0.760`

这已经逼近 `least_fatigue = -0.725`。

### 3. `SiLU` 没有带来提升

在相同主配置下试 `SiLU`，表现不如 `ReLU`：

- best 大约在 `-0.796`

因此当前不把 `SiLU` 作为主线。

### 4. `target_sync_interval=500` 不如 `200`

尝试更慢的 target 同步后，表现下降：

- best 大约在 `-0.968`

说明在这个小环境中，target 更新过慢会让 bootstrap 目标过旧。

### 5. 学习率进一步降到 `1e-4`

这一步带来了真正的突破。

一组结果：

- best 大约在 `-0.190`

另一组结果：

- best 大约在 `-0.104`

这两个结果都明显优于：

- `least_fatigue = -0.725`
- `round_robin = -0.630`
- `observable_click_greedy = -0.480`

这说明：

- 当前 DQN 不只是优于 tabular
- 也不只是接近 heuristic
- 而是已经学出了比一步贪心和简单疲劳控制更强的长期策略

## 当前最重要的认识更新

当前必须修正一个概念：

- baseline 中的 oracle 只是 myopic oracle
- 它并不是长期 reward 的真正上界

所以 DQN 超过它，说明的是：

- DQN 没有被一步即时点击率困住
- 它在做更好的 fatigue 管理和长期 reward 权衡

这正是强化学习在推荐序列决策中的核心价值。

## 当前主要问题

虽然 DQN 已经有效，但当前仍有一个非常明显的问题：

- `best checkpoint` 和 `final checkpoint` 差距很大

例如一轮强结果中：

- best 可到 `-0.104`
- final 可能退化到 `-0.614` 或更差

这说明：

- 当前主问题已经不再是“学不会”
- 而是“继续训练会把已学到的好策略冲坏”

也就是说，当前瓶颈更像是：

- 训练稳定性
- 早停策略
- 多 seed 可复现性

而不是网络表达能力本身。

## 当前主配置

到目前为止，最优主线配置应暂定为：

- `learning_rate = 1e-4`
- `activation = relu`
- `target_sync_interval = 200`

此外，当前经验也表明：

- `best checkpoint` 必须保留
- 结果汇报时应优先使用 `best_eval_reward`
- 不应只看最终一次 eval

## 当前项目策略

当前项目不建议再做这些事：

- 回到 tabular 分支继续抠离散化
- 继续无目的扩状态编码器
- 随意切换更复杂网络

当前最合理的策略是：

1. 固定 DQN 主结构
2. 以 `1e-4 + ReLU + sync200` 作为主配置
3. 用更密的 eval 监控训练曲线
4. 以 `best checkpoint` 作为当前结果主体
5. 做多 seed 稳定性验证

## 当前最自然的下一步

当前最值得继续做的不是大改模型，而是把这条已经成功的 DQN 主线做稳：

1. 多 seed 验证
   - 至少比较 `seed=7 / 42 / 123`

2. 引入 early stopping
   - 连续若干次 eval 未提升则提前停止

3. 做一个多 seed 汇总脚本
   - 汇总每个 seed 的 `best reward / best episode / final reward`

## 当前阶段一句话结论

当前 `Micro-RecSim` 项目已经完成了一个关键转折：

- `Q-table` 证明了离散化在连续推荐状态下的局限
- `DQN` 成功学出了超过启发式 baseline 的长期策略

因此，项目主线已经明确转向：

- 从“证明 RL 能否工作”
- 进入“验证 DQN 结果是否稳定可复现”的阶段
