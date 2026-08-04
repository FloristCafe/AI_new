# 2026-08-04 代码改动与实验记录总表

## 这篇笔记的用途

这篇笔记用于把当前阶段已经完成的核心代码改动与关键实验记录集中整理到一处。

它不是替代之前的分主题笔记，而是作为总览索引，回答两个问题：

- 这个项目到现在代码层究竟改了什么
- 这些改动分别对应了哪些实验现象与结论

## 一、项目主线已经完成的代码改动

### 1. 项目骨架与上游复用

项目 `movielens_1m_sasrec_dqn_offline_rl` 已经从零搭建完成，并明确复用了上游项目 `movielens_1m_sasrec_baseline` 的 SASRec 表征能力。

已经落地的核心文件包括：

- `src/build_offline_buffer.py`
- `src/ml_1m_genre_utils.py`
- `src/sasrec_dqn_model.py`
- `src/train_sasrec_dqn.py`
- `src/evaluate_sasrec_dqn.py`

这一阶段完成的不是单个脚本，而是一条完整工程闭环：

- 数据转离线轨迹
- 预训练 SASRec encoder 复用
- DQN 训练
- test / all 双评估脚本

### 2. Offline buffer 构建完成

ML-1M 原始序列已经被重塑为离线 RL 五元组：

- `(S_t, A_t, R_t, S_{t+1}, Done)`

状态定义为最近 50 个 item 的序列，动作是下一个真实点击 item，经验被固化到离线 buffer 中，供后续训练与评估统一复用。

这一步的重要意义是：

- 推荐问题不再只是监督学习样本对
- 项目正式获得了离线 MDP 训练接口

### 3. SASRec-DQN 主体网络完成

当前模型结构已经固定为：

- 感知模块：SASRec encoder
- 状态抽取：最后一个非 padding 位置的隐藏状态
- 决策模块：线性 `Q-head`

也就是说，当前主线不是直接输出 softmax 概率，而是输出全量 item 的 Q 值。

### 4. 训练主线从基础 DQN 升级到稳定版离线训练器

训练器已经历多轮关键改造，目前具备：

- Double DQN
- CQL conservative penalty
- Polyak soft target update
- Huber TD loss
- gradient clipping
- Q-value 监控
- adaptive CQL alpha

这意味着当前训练器已经不是最早期的原型，而是明确围绕“离线训练稳定性”做过多次增强的版本。

### 5. 奖励体系已切换为 binary reward

早期尝试过带 genre 微奖励的 dense reward。

后续已经正式切换为：

- 命中真实下一个 item：`reward = 1.0`
- 未命中：`reward = 0.0`

这是一个非常关键的修正，因为它让环境目标从“模糊相似性奖励”回到了“精确 next-item 命中”。

### 6. 选模逻辑从 loss 切换为 ranking metric

这是整个项目最重要的工程修复之一。

best checkpoint 现在不再按：

- `valid_total_loss`

而是按：

- `valid_ndcg_at_10`

来选择。

这一步真正修正了“训练数值目标”和“推荐排序目标”错位的问题。

### 7. CE regularization 已加入主线

在 `RL loss` 之外，训练器已经加入监督式排序锚点：

- `ce_loss`

联合目标变为：

- `RL loss + lambda * CE loss`

它的作用不是简单再做一次监督学习，而是把 RL 的长期价值学习和 next-item 排序边界约束焊接到同一张图里。

### 8. BPR / pairwise regularization 入口已加好

为了进入真正的方法升级阶段，训练器已经支持：

- `--supervised-regularizer ce`
- `--supervised-regularizer bpr`
- `--supervised-regularizer none`

并新增：

- `--bpr-negative-count`

这意味着当前训练器已经具备从 CE 线切到 pairwise ranking 线的能力。

### 9. 针对 “best_epoch = 1” 问题的新修复已加入

为处理早期反复出现的 “epoch 1 最好，后续迅速恶化” 现象，代码又新增了两项训练动力学修复：

- `encoder warmup freeze`
- `Q-head` 小尺度初始化

对应新参数包括：

- `--encoder-warmup-epochs`
- `--q-head-init-std`
- `--q-head-init-mean`

并且训练日志新增：

- `train_encoder_learning_rate`

这让我们后续可以正式验证：

- 问题到底是预训练表征被过早破坏
- 还是 RL 目标本身不足以支撑排序能力持续提升

## 二、关键实验记录与结论

### 1. 最早期实验暴露出明显的 value explosion 与选模错位

在早期不稳定版本中，出现了非常典型的坏现象：

- `best_epoch = 1`
- 后续 epoch 的 `valid_ndcg_at_10` 持续下降
- Q 值随训练快速抬升

一个典型坏实验是：

- `cql_softupdate_unfreeze_adaptive_main`

其关键现象包括：

- `gamma = 0.99`
- `selection_metric = valid_total_loss`
- `best_epoch = 1`
- `valid_max_q_value` 最终升到 `129.621182`

这说明当时的问题不只是分数差，而是训练动力学本身就不健康。

### 2. Huber + 降 gamma + binary reward 是一条真正的稳定化主线

在后续修正中，我们逐步把主线收束到：

- `Huber TD`
- `binary reward`
- `soft target update`
- `valid_ndcg_at_10` 选模

这条主线的意义不是一步到位提分，而是先把系统从“容易失控”拉回到“可以认真比较实验”的状态。

### 3. gamma 扫描结论：0.9 优于 0.7 / 0.85

当前阶段已经得到明确结论：

- `gamma = 0.7` 过短视
- `gamma = 0.85` 明显恢复
- `gamma = 0.9` 成为当前更优 horizon

也就是说，在 MovieLens-1M 这个离线序列推荐问题上，过度短视会伤害排序能力，但极高 `gamma` 又会放大目标漂移。

### 4. CE regularization 明显改善了排序质量

在引入 CE regularization 之后，训练目标开始更好地对齐 Top-K 排序能力。

尤其重要的是，它不只是抬升 test 指标，还修复了一个更深层的问题：

- `best_epoch` 不再总是死在 `1`

这说明训练不再只是“初始预训练表征最好，之后越训越坏”，而是真正出现了可持续优化。

### 5. CE 权重扫描收口结果

在当前 strongest baseline 主线上，已经完成：

- `ce = 0.3`
- `ce = 0.4`
- `ce = 0.5`

的对照扫描。

#### `ce = 0.3`

- `test HR@10 = 0.16142384105960264`
- `test NDCG@10 = 0.08980818238890449`
- `all HR@10 = 0.2359417764987643`
- `all NDCG@10 = 0.127011027764501`
- `mean_cumulative_reward_per_user = 7.63476821192053`

#### `ce = 0.4`

- `test HR@10 = 0.1849337748344371`
- `test NDCG@10 = 0.09816852838509364`
- `all HR@10 = 0.22594347641095228`
- `all NDCG@10 = 0.1160737573016917`
- `mean_cumulative_reward_per_user = 6.131788079470199`

#### `ce = 0.5`

- `test HR@10 = 0.19039735099337748`
- `test NDCG@10 = 0.1021792974824788`
- `all HR@10 = 0.24074176523307406`
- `all NDCG@10 = 0.12533210207531637`
- `mean_cumulative_reward_per_user = 6.873013245033112`

这轮扫描的正式结论是：

- `ce = 0.4` 不是最优平衡点
- `ce = 0.3` 更偏全轨迹质量
- `ce = 0.5` 是当前 next-item 排序最强版本

### 6. strongest baseline 已正式封板

截至当前阶段，正式 strongest baseline 为：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05`

它的意义不只是分数最好，更重要的是：

- valid ranking 指标支持它
- test 指标支持它
- 训练过程本身比早期版本健康

### 7. multi-seed 验证已经完成

为防止 strongest baseline 只是单 seed 偶然结果，已经完成：

- `seed = 7`
- `seed = 42`
- `seed = 2026`

多 seed 汇总。

聚合结果如下：

- `best_valid_ndcg_at_10 mean = 0.11165426968574889`
- `best_valid_ndcg_at_10 std = 0.00030601067141307315`
- `test_hr_at_10 mean = 0.19188741721854305`
- `test_hr_at_10 std = 0.001216633978203567`
- `test_ndcg_at_10 mean = 0.10336486618167529`
- `test_ndcg_at_10 std = 0.0008383258209104889`
- `all_hr_at_10 mean = 0.24270621996863714`
- `all_hr_at_10 std = 0.0014317630498752831`
- `all_ndcg_at_10 mean = 0.12632784969970204`
- `all_ndcg_at_10 std = 0.0007515397971753838`
- `all_mean_cumulative_reward_per_user mean = 6.913962472406181`
- `all_mean_cumulative_reward_per_user std = 0.04338492541299075`

这一步非常重要，因为它说明 strongest baseline 已经不再是偶然样本，而是稳定复现的结果。

### 8. 当前关于 “best_epoch = 1” 的新认识

随着实验推进，我们对这个现象的理解已经更具体：

- 预训练 SASRec encoder 初始排序能力很强
- 新接入的随机 Q-head 在训练初期可能给出噪声极大的 Q 值
- RL 目标和 Top-K 排序目标本身存在错位
- 如果缺少足够强的监督锚点，模型会出现“第 1 轮最好，后面越训越偏”

因此，新加入的 warmup 与小尺度初始化，不是普通微调，而是正式针对训练动力学问题的修复。

## 三、当前阶段的整体结论

到现在为止，这个项目已经不再是“能不能跑通”的问题，而是：

- 已经有一条稳定的 strongest baseline
- 已经把早期数值失控和选模错位问题基本理顺
- 已经具备进入真正方法升级阶段的工程基础

换句话说，当前阶段已经完成了三件最关键的事情：

- 把离线序列推荐 RL 系统搭起来
- 把 strongest baseline 稳下来
- 把下一步创新接口预留出来

## 四、接下来最合理的实验方向

在这份总表对应的时间点之后，最值得继续推进的不是普通参数扫描，而是两条更有研究意义的线：

### 1. 训练动力学修复验证

继续验证：

- `encoder warmup epochs = 0 / 3 / 5`

观察：

- `best_epoch`
- `valid_ndcg_at_10`
- `train / valid Q-value`

是否明显变得更健康。

### 2. 真正的方法升级

在 strongest baseline 稳住后，继续进入：

- `BPR / pairwise ranking`
- 更进一步的 `IQL / AWAC`

而不是继续围绕单个普通参数反复扫值。
