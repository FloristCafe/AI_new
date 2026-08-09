# 2026-08-09 Generative Slate-DQN 启动落地与 Valid 基线差距结论

## 今天完成了什么

今天把 `movielens_1m_generative_slate_dqn` 从概念图纸推进到了可训练、可验证、可出报告的完整工程闭环。

已经落地的核心组件有：

- 新真实项目目录：
  - `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn`
- 新环境子类：
  - 在 `D:\Python\Artificial Intelligence\projects\recommendation\recommender_mdp_gymnasium\micro_recsim_env.py` 中新增 `SlateRecSimEnv`
- 新模型：
  - `src/slate_dqn_model.py`
- 新在线训练脚本：
  - `src\train_generative_slate_dqn.py`
- 新评估脚本：
  - `src\evaluate_generative_slate_dqn.py`

这一版系统遵循的主逻辑是：

- 冻结上游 `SASRec` 作为用户状态编码器
- 用 `GRU` 编码已生成的 slate prefix
- 自回归生成长度为 `5` 的推荐列表
- 将完整 slate 丢入 `SlateRecSimEnv`
- 用整条 slate 的总点击数 `R_total` 更新 5 个 micro-actions

## 环境与训练侧的关键实现

### 1. Slate 环境已经变成真实可用版本

`SlateRecSimEnv` 已经支持：

- `step()` 接收长度为 `5` 的整型 slate
- 基于 MovieLens genre 映射 item type
- 位置曝光概率：
  - `1 / log2(k + 1)`
- 同类目重复惩罚：
  - 同类目后续位置点击率打五折
- 被点击类别 fatigue 上升，未点击类别随时间衰减
- 以整条 slate 的总点击数作为 reward

### 2. 训练主循环已经打通

在线训练已经完整跑通，能够产出：

- `slate_dqn_best.pt`
- `slate_dqn_final.pt`
- `training_metrics.json`

训练过程中还修掉了一个关键 bug：

- `generate_slate()` 临时把模型切到 `eval()` 后没有恢复 `train()`，导致 CUDA 下 `GRU backward` 报错
- 现已改为：
  - rollout 时临时 `eval()`
  - 结束后恢复原模式
  - 真正做梯度更新前显式 `online_model.train()`

### 3. 训练生成阶段做了第一轮提速

已经完成的提速项：

- 去掉训练时每个位置的 `q_values.cpu()`
- 将同一 slate 内的 action masking 与 `argmax` 保留在 GPU
- 让 prefix tensor 常驻 GPU，避免反复 `numpy -> torch`

这一步不改变算法语义，只减少训练中的设备同步开销。

## 评估协议已经落成代码

`evaluate_generative_slate_dqn.py` 已支持：

- `slate_dqn`
- `sasrec_topk`
- `popularity_topk`
- `random_unique`

并输出统一 JSON 报告，覆盖：

- `reward_metrics`
- `position_metrics`
- `diversity_metrics`
- `offline_bridge_metrics`
- `policy_ranking`

同时，评估脚本已经按日常工作流做了 split-aware 默认值：

### Valid 选模阶段默认

- `mc_rollouts = 1`
- `max_eval_users = 1000`
- `max_slates_per_episode = 5`
- 可用 `--skip-baselines` 只测 `slate_dqn`

### Test 最终报告阶段建议

- 全量用户
- `mc_rollouts = 3 ~ 5`
- `max_slates_per_episode = 10`
- 开启四个策略全量对比

## 本轮正式实验结果

实验目录：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn\artifacts\experiments\slate_dqn_online_seed42_20260809`

### 1. 训练结果

训练汇总文件：

- `metrics\training_metrics.json`

关键结果：

- `best_episode = 900`
- `best_eval_mean_slate_reward = 0.939`
- `last_episode = 1000`
- `last_eval_mean_slate_reward = 0.824`

解读：

- `best.pt` 的选模是有效的
- 后期已经出现一定回落
- 不能直接拿 `final.pt` 代表最优模型

### 2. Valid 上只测 `slate_dqn` 的结果

关键结果：

- `mean_slate_reward = 0.987`
- `mean_episode_return = 4.935`
- `slate_success_rate = 0.6708`
- `intra_list_diversity_mean = 0.8083`
- `click_rate_by_position = [0.3434, 0.2494, 0.1582, 0.1276, 0.1084]`

这说明：

- 模型在 simulator 中确实学到了有效的 slate 排序
- 点击率随位置单调下降，符合环境物理规律
- 列表内部没有明显塌缩到极端同质化

## 最重要的 Valid 基线对比结论

本轮对比设置：

- `1000` 个 valid 用户
- `mc_rollouts = 1`
- `max_slates_per_episode = 5`
- 对比四个策略：
  - `slate_dqn`
  - `sasrec_topk`
  - `popularity_topk`
  - `random_unique`

### 各策略关键结果

#### `slate_dqn`

- `mean_slate_reward = 0.987`
- `mean_episode_return = 4.935`
- `target_hit_at_5 = 0.0`
- `target_ndcg_at_5 = 0.0`
- `genre_hit_at_5 = 0.816`
- `intra_list_diversity_mean = 0.8083`

#### `sasrec_topk`

- `mean_slate_reward = 0.8938`
- `mean_episode_return = 4.469`
- `target_hit_at_5 = 0.136`
- `target_ndcg_at_5 = 0.0820`
- `genre_hit_at_5 = 0.873`
- `intra_list_diversity_mean = 0.5707`

#### `popularity_topk`

- `mean_slate_reward = 0.9362`
- `mean_episode_return = 4.681`
- `target_hit_at_5 = 0.012`
- `target_ndcg_at_5 = 0.0086`

#### `random_unique`

- `mean_slate_reward = 0.9624`
- `mean_episode_return = 4.812`
- `target_hit_at_5 = 0.001`
- `target_ndcg_at_5 = 0.00039`

### 正式结论

这一轮结果非常明确：

- `slate_dqn` 在 **simulator reward** 上是当前最强
- 但 `slate_dqn` 在 **真实持出目标命中** 上是失败的
- `sasrec_topk` 才是当前 `target_hit@5 / target_ndcg@5` 上最强的真实桥接基线

换句话说：

- 当前 `Slate-DQN` 已经学会了在手写环境中赚点击
- 但它学到的策略与真实 MovieLens 下一物品预测并不对齐

## 如何解释这次结果

当前最合理的研究解释是：

### 1. 在线 simulator reward 与真实离线持出标签存在 objective mismatch

环境当前更偏好：

- genre 对齐
- 位置收益
- 多样性收益
- fatigue 管理

而不直接奖励：

- 真实下一部电影 item 的命中

### 2. 模型可能学到了“推对类别”，但没有学到“推对具体 item”

支持这个判断的证据是：

- `slate_dqn` 的 `genre_hit_at_5 = 0.816`
- 但 `target_hit_at_5 = 0.0`

这意味着：

- 它经常覆盖到了正确的大类
- 但没有把真实目标 item 放进 slate

### 3. 当前结果不能直接写成“优于真实推荐基线”

现在可以成立的说法是：

- 这套 `Generative Slate-RL` 工程闭环已经打通
- `SlateRecSimEnv + Slate-DQN` 能在自定义环境中稳定优化 slate reward

现在不能直接成立的说法是：

- 它已经优于 `SASRec` 的真实推荐质量

## 今天顺手确认的风险点

### 1. 验证脚本没有死循环

之前怀疑 `evaluate_generative_slate_dqn.py` 死循环，后面确认：

- 不是死循环
- 是默认评估规模过大
- 再加上中间没有进度日志，视觉上像卡死

### 2. 当前还有一个环境 correctness 风险没有处理

`SlateRecSimEnv.step()` 目前还没有在环境层做“同一 slate 内 item 不可重复”的硬校验。

虽然：

- 训练策略与评估策略层都做了无重复约束

但环境本身还应补一层防线，避免未来某个新策略静默传入重复 slate。

### 3. 当前还没有屏蔽用户历史中已经看过的 item

训练和评估目前都只屏蔽了：

- 同一 slate prefix 内重复

还没有屏蔽：

- 用户历史序列里已经消费过的 item

这可能也是 bridge 指标偏差的一部分来源。

## 截至今天的项目阶段判断

截至 **2026-08-09**，这个项目已经完成了：

1. `Generative Slate-RL` 工程启动
2. 在线训练闭环打通
3. Valid 报告生成
4. 与 `SASRec / Popularity / Random` 的首轮有效对比

当前最准确的阶段定位是：

- **工程上已经可用**
- **研究上已经出现第一个有价值的负结果**

这个负结果不是坏事，反而说明下一步方向已经很清楚：

- 问题不再是“能不能把 Slate-RL 跑起来”
- 而是“如何让 simulator reward 和真实推荐目标重新对齐”

## 下一步最值得做什么

优先级最高的不是立刻跑 full test，而是先修正训练目标对齐。

当前最值得推进的方向：

### 1. 给环境 reward 加入 item-level 对齐信号

不要只让环境奖励：

- 点击数
- genre 覆盖
- 多样性

还应该让 reward 对真实目标 item 更敏感。

### 2. 在训练损失中加入 supervised anchor

例如让 `Slate-DQN` 在在线 TD 更新之外，再保留一定程度的：

- item-level imitation
- 或与 `SASRec` 排序分布的对齐项

### 3. 补环境侧 duplicate hard-check

确保 future baseline 或新策略不可能静默生成非法 slate。

## 一句话总结

今天这轮 `movielens_1m_generative_slate_dqn` 的正式阶段结论是：

- **Generative Slate-DQN 已经在工程上完整落地，并且能在手写 SlateRecSimEnv 中学到高 reward、高多样性的列表策略；但当前 reward 设计与真实 MovieLens 下一物品命中目标明显错位，导致模型在线收益最强，却在 `target_hit@5 / target_ndcg@5` 上显著落后于 `sasrec_topk`。**
