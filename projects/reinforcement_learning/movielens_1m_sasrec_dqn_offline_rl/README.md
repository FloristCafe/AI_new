# MovieLens 1M SASRec DQN Offline RL

这是一个真实离线强化学习项目，目标是在已经跑通的 `MovieLens-1M SASRec baseline` 之上，构建一个 `SASRec + DQN` 的工业级离线推荐决策引擎。

## 项目目标

- 复用 `MovieLens-1M` 的用户电影交互序列
- 用强化学习视角重构离线马尔可夫轨迹
- 用 `SASRec` 作为状态编码器提取用户当下意图
- 在编码器顶部拼接 `DQN Q-head`
- 使用离线经验池训练 `Double DQN`
- 用 `HR@10`、`NDCG@10` 和轨迹累计奖励做离线回测

## 状态、动作、奖励定义

- `State S_t`
  - 用户最近的 50 个交互电影 ID 序列
  - 长度不足时做零填充
- `Action A_t`
  - 在全部候选电影中推荐 1 个电影 ID
- `Reward R_t`
  - 精确命中真实下一个点击：`+1.0`
  - 未命中但与真实下一个电影同类型：`+0.1`
  - 类型完全不匹配：`-0.1`
- `Done`
  - 到达该用户真实历史的最后一个物品

## 目录结构

- `src/build_offline_buffer.py`
  - 把 `MovieLens-1M` 历史序列转成离线五元组
  - 产出 `(state, action, reward, next_state, done)` 经验池
- `src/ml_1m_genre_utils.py`
  - 读取电影类型信息
  - 提供 genre 匹配奖励所需的辅助函数
- `src/sasrec_dqn_model.py`
  - 加载或复用 `SASRec` 编码器
  - 抽取最后一个隐藏状态
  - 拼接 `Q-head`
- `src/train_sasrec_dqn.py`
  - 从离线经验池采样 batch
  - 训练 `Double DQN`
  - 管理 target network、checkpoint 和指标保存
- `src/evaluate_sasrec_dqn.py`
  - 在测试轨迹上做离线回测
  - 输出 `HR@10`、`NDCG@10` 和累计奖励

## 计划复用的上游项目

- 上游项目：
  - `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline`
- 上游笔记：
  - `D:\Python\Artificial Intelligence\docs\project_notes\reinforcement_learning\movielens_1m_sasrec_baseline`

## 计划产物

- `artifacts/offline_buffer/`
  - 固化后的离线经验池
- `artifacts/checkpoints/`
  - `SASRec-DQN` 模型权重
- `artifacts/metrics/`
  - 训练过程指标
- `artifacts/predictions/`
  - 离线回测结果
- `artifacts/debug/`
  - 中间检查产物

## 当前阶段

当前先完成：

1. 新项目目录落地
2. 上游 SASRec 可复用资产梳理
3. 离线经验池构建脚本设计

随后再进入：

1. 轨迹构建
2. `SASRec-DQN` 模型拼接
3. 离线 `Double DQN` 训练
4. 测试集回测评估
