# 2026-07-27 离线评估脚本落地与当前状态

## 本次新增内容

本次在 `SASRec-DQN` 项目中补齐了离线评估脚本：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\evaluate_sasrec_dqn.py`

到这一刻为止，这个项目已经不再只有：

- offline buffer
- 模型骨架
- 训练循环

而是已经具备了：

- 经验池构建
- `SASRec-DQN` 训练
- 离线回测评估

三条主链路。

## 当前评估脚本支持什么

当前评估脚本已经支持：

- 读取 `train / valid / test / all` 任意 replay buffer split
- 加载训练好的 `SASRec-DQN` checkpoint
- 对全量 `3706` 个动作输出 Q-value
- 对历史中已经见过的 item 做屏蔽
- 计算 top-k 排序指标：
  - `HR@k`
  - `NDCG@k`
- 计算 `top1` 推荐动作对应的 reward
- 按用户汇总累计 reward

也就是说，当前评估已经同时覆盖了两类结果语言：

- 推荐系统语言：
  - `HR@10`
  - `NDCG@10`
- 强化学习语言：
  - `top1 average reward`
  - `mean cumulative reward per user`

## 当前 reward 回测是如何做的

评估阶段并不是继续沿用 buffer 里 logged action 自带的 `+1`，而是：

- 模型先在当前状态上输出全量 Q-values
- 取 `top1` 作为当前策略真正会执行的动作
- 再拿这个动作与真实下一点击比较
- 按 reward 规则重新计算：
  - 精确命中：`+1.0`
  - genre 匹配：`+0.1`
  - genre 不匹配：`-0.1`

这很关键，因为它说明：

- buffer 中退化的 logged reward
- 与评估时 agent 自己选动作后得到的 reward

不是同一回事。

训练阶段当前仍然受 logged buffer 限制，但评估阶段已经开始接近“策略本身质量”的检查。

## smoke evaluation 已完成

本次已经基于 smoke training 产物完成了两种小规模评估：

### 1. `test` split

这是每个用户最后一个时间步的单步评估。

当前 smoke 结果显示：

- `HR@10 = 0.000000`
- `NDCG@10 = 0.000000`
- `top1 average reward = 0.020000`
- `mean cumulative reward per user = 0.020000`

这个结果说明：

- 当前 smoke 训练出来的策略几乎还没有形成有效的 top-k 命中能力
- 但 top1 动作偶尔能拿到 genre 匹配的微小奖励

### 2. `all` split

这是更接近整条离线轨迹回放的评估方式。

当前 smoke 结果显示：

- `HR@10 = 0.029083`
- `NDCG@10 = 0.015477`
- `top1 average reward = -0.011633`
- `mean cumulative reward per user = -1.040000`

这个结果说明：

- 当前策略在全轨迹上仍然非常弱
- 平均累计 reward 已经是负值
- 说明 agent 的 top1 动作大多数时候仍然偏离真实点击与 genre 匹配方向

## 这些 smoke 结果该如何解读

这里最重要的是：

- smoke 结果的意义不是“模型已经好”
- 而是“训练与评估代码已经形成闭环”

当前这些数字更多用于证明：

1. 模型能从训练脚本输出 checkpoint
2. 评估脚本能读取 checkpoint
3. 排序指标和 reward 指标都能稳定落盘

因此，目前不应对 smoke 指标本身做业务级解读，而应把它理解为：

- 工程链路验证成功

## 当前项目已经形成的能力闭环

截至 2026-07-27，当前项目已经拥有：

### 1. 数据闭环

- `MovieLens-1M` 原始交互
- 上游 `SASRec baseline` 映射与序列资产
- 下游 offline replay buffer

### 2. 模型闭环

- `SASRec encoder`
- `Q-head`
- `SASRec-DQN`

### 3. 训练闭环

- offline replay sampling
- `Double DQN`
- `CQL penalty`
- checkpoint 保存
- metrics 保存

### 4. 评估闭环

- top-k ranking 指标
- top1 reward
- per-user 累计 reward

这意味着项目已经从“立项阶段”进入“可迭代优化阶段”。

## 当前仍然最重要的风险

虽然评估脚本已经落地，但项目最核心的问题仍然没有消失：

- 当前 replay buffer 的 logged reward 是退化的
- 训练阶段仍然是在这种退化信号上做 `Q-learning`

所以当前项目的下一步，不是简单地“多跑几轮训练”，而是必须继续思考：

### 1. 当前 reward 设计如何真正进入训练目标

尤其是：

- genre reward 目前更多只在评估时体现
- 还没有真正变成高信息量的训练监督

### 2. 当前 offline DQN 的 value 学习是否仍然过于脆弱

虽然已经加了 `CQL`，但仍然要密切观察：

- `mean_q_value`
- `max_q_value`
- `td_loss`
- `cql_penalty`

### 3. 当前评估还不是在线环境评估

它本质上仍然是：

- 在静态日志状态上做 policy replay

因此，它更接近：

- offline policy scoring

而不是完整的在线交互式环境验证。

## 当前阶段总结

截至 2026-07-27，`movielens_1m_sasrec_dqn_offline_rl` 已经完成：

1. 项目骨架与 README
2. 上游 SASRec 资产复用梳理
3. offline buffer 构建
4. `SASRec-DQN` 模型拼接
5. `Double DQN + CQL` 训练循环
6. top-k 与累计 reward 的离线评估脚本

项目现在已经具备完整工程闭环，下一步将进入：

- 正式训练配置整理
- 正式实验运行
- 结果解读与训练目标修正
