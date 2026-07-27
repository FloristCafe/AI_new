# 2026-07-27 CQL 训练循环启动

## 本次推进内容

本次真正进入了 `SASRec-DQN` 项目的训练主链路实现阶段，完成了两块核心内容：

- `SASRec-DQN` 模型拼接
- `Double DQN + CQL` 离线训练循环落地

对应代码文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\sasrec_dqn_model.py`
- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\train_sasrec_dqn.py`

## 模型层已经落地的内容

当前 `sasrec_dqn_model.py` 已经支持：

- `SASRecEncoder`
- 取最后一个隐藏状态作为当前状态 embedding
- `Linear(128, 3706)` 的 `Q-head`
- 加载上游强基线 `SASRec` checkpoint
- 冻结 encoder 或开放 encoder 微调
- 输出全量动作 Q-value
- 根据给定动作 id 抽取当前动作 Q 值

这意味着当前项目已经完成了从“监督学习推荐编码器”到“离线 RL 状态编码器”的结构升级。

## 训练循环已经落地的内容

当前 `train_sasrec_dqn.py` 已经支持：

- 读取离线 replay buffer
- 构建 train / valid dataloader
- 初始化 online network 与 target network
- 计算 `Double DQN` 目标值
- 计算 `TD loss`
- 计算 `CQL penalty`
- 融合为 `total_loss = td_loss + alpha * cql_penalty`
- 记录训练和验证阶段的：
  - `total_loss`
  - `td_loss`
  - `cql_penalty`
  - `mean_q_value`
  - `max_q_value`
  - `mean_current_q`
  - `mean_target_q`
- 保存 best checkpoint 和 final checkpoint
- 落盘 `training_metrics.json`

## 本次真正焊进去的核心公式

本次不是只把 `DQN` 空壳搭起来，而是已经把离线 RL 最核心的保守项焊到了训练图里：

- 当前动作 Q：
  - `current_q = Q(s, a_logged)`
- Double DQN target：
  - 由 online network 在 `s'` 上选动作
  - 再由 target network 在该动作上评估目标值
- CQL 惩罚项：
  - `logsumexp(all_q_values) - current_q`
- 总损失：
  - `td_loss + alpha * cql_penalty`

这一步的意义非常关键：

- 它不是普通的 offline DQN
- 而是已经开始对 OOD 动作高估做保守抑制

## 本次踩到的关键工程问题

在小规模 smoke training 时发现了一个非常隐蔽但关键的问题：

- 模型一旦切到 `eval()` 路径
- `MultiheadAttention` 的推理 fastpath 会在当前这类左侧 padding 的序列 batch 上产生 `NaN`

这会直接摧毁：

- target network 前向
- validation 前向
- 后续所有 `Double DQN` 目标值计算

最终定位到的症状是：

- `online_model.train()` 下的 `Q-values` 是正常的
- `target_model.eval()` 下的 `Q-values` 直接变成 `NaN`

为了解决这个问题，已经在模型文件里加入全局保护：

- 关闭 `torch.backends.mha` 的推理 fastpath

这一步不是“代码风格修补”，而是训练能否稳定运行的必要条件。

## 当前验证状态

在修复 `eval()` 路径的数值问题后，已经完成了一次 smoke training 验证：

- 能跑完 1 个 epoch
- 能打印 step 级别的 `mean_q` 与 `max_q`
- 能输出 train / valid loss
- 能保存 checkpoint
- 能保存 metrics

说明这条训练主链路已经从“占位脚本”升级成了“可执行训练骨架”。

## 当前阶段仍然需要记住的事实

虽然训练主链已经落地，但当前 offline buffer 仍有一个结构性事实：

- buffer 中存的是 logged next-click actions
- 因而即时 reward 基本全为 `+1.0`

这意味着当前项目虽然已经有：

- replay buffer
- Double DQN
- CQL regularization

但后续仍然必须继续面对一个核心问题：

- 当前 logged-action buffer 的奖励分布是退化的

也就是说，后续真正决定项目质量的，不只是“训练代码能不能跑”，而是：

- 如何让离线目标与推荐决策价值更有辨识度

## 当前阶段结论

截至 2026-07-27，本项目已经完成：

1. `SASRec-DQN` 模型骨架落地
2. `Double DQN + CQL` 训练循环落地
3. `mean/max Q-value` 日志监控接入
4. `eval()` 下注意力 fastpath 数值问题定位并修复

下一步将进入：

- 整理正式训练命令
- 进一步完善评估脚本
- 决定如何处理退化 reward 与更可信的离线评估策略
