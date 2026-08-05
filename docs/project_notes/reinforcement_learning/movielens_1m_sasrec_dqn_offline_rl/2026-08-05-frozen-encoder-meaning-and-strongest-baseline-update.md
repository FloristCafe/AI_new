# 2026-08-05 Frozen Encoder 的含义与 strongest baseline 更新

## 本次记录的核心问题

在完成：

- `warmup5`
- `encoder_frozen`

两组实验之后，一个非常关键的问题被正式提了出来：

- 如果 encoder 冻结了，是否意味着 RL 在这个项目里没有学到东西？

结论是：

- **不意味着 RL 没学到东西**
- 只意味着 **RL 目前没有有效改写底层状态表征**
- 但它已经明确学会了 **在固定表征上重塑动作价值与排序边界**

## 一、冻结 encoder 到底意味着什么

当前 SASRec-DQN 的结构可以粗略拆成两段：

### 1. 表征层

- `SASRec encoder`

它负责把用户历史序列编码成状态向量，也就是把：

- 历史 item 序列

压缩成：

- 用户当前意图的稠密表示

### 2. 决策层

- `Q-head`

它负责在这个状态向量上输出所有 item 的 Q 值，进而完成：

- 动作价值建模
- Top-K 排序
- 下一个 item 的推荐决策

因此，冻结 encoder 并不等于整个模型不学习，而只是意味着：

- 表征层不更新
- 决策层继续学习

## 二、RL 在 frozen encoder 设置下学到了什么

当前实验已经证明，在 fixed representation 上，RL 仍然能学到非常实质的东西。

它至少学到了三件事：

### 1. 学到了新的 Q-value 结构

即使 encoder 不变，Q-head 仍然在持续学习：

- 哪些状态下哪些 item 更有长期价值
- 哪些动作应该被压低
- 哪些动作应该被推到前列

### 2. 学到了新的排序边界

当前项目的评估不是看一个抽象损失，而是看：

- `HR@10`
- `NDCG@10`

而 frozen encoder 版本能显著提升这些指标，说明：

- 模型在固定状态表示上，确实学到了更好的 next-item 排序方式

### 3. 学到了比原 strongest baseline 更强的策略头

这次 frozen encoder / warmup5 阶段的 best checkpoint 对应结果为：

- `test HR@10 = 0.2099337748344371`
- `test NDCG@10 = 0.1166359124280432`
- `all HR@10 = 0.2737029619712544`
- `all NDCG@10 = 0.14818211838997059`
- `all mean_cumulative_reward_per_user = 9.029304635761589`

而此前 strongest baseline `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05` 的结果为：

- `test HR@10 = 0.19039735099337748`
- `test NDCG@10 = 0.1021792974824788`
- `all HR@10 = 0.24074176523307406`
- `all NDCG@10 = 0.12533210207531637`
- `all mean_cumulative_reward_per_user = 6.873013245033112`

这说明 frozen encoder 并不是“没学到”，而是：

- **在不动底层表征的前提下，学到了一个更强的决策头**

## 三、当前实验告诉了我们什么

### 1. RL head-only learning 是成立的

至少在当前这个 MovieLens-1M 离线序列推荐设置下，以下判断已经有明确证据支持：

- `RL + CE` 对 `Q-head` 是有效的

换句话说：

- RL 不是完全无用
- 它在固定表征上可以学到可转化为排序提升的价值结构

### 2. 当前没有证据证明 RL fine-tune encoder 有效

`warmup5` 的关键现象是：

- `best_epoch = 5`
- 最优点正好出现在 encoder 仍然冻结的阶段
- 从 `epoch 6` 开始解冻 encoder 后，`valid_ndcg_at_10` 立即回落

这说明在当前配置下：

- RL 微调 encoder 没有带来额外收益
- 反而更像是在破坏原有的预训练排序表征

所以当前更精确的结论不是：

- RL 没学到东西

而是：

- **RL 学会了决策头，但还没有学会安全地改写 encoder**

## 四、为什么 warmup5 与 encoder_frozen 得到完全相同结果

这是因为两次实验的最佳点都落在：

- `epoch 5`

而 `warmup5` 在前 5 个 epoch 中本来就保持 encoder 冻结。

因此：

- `warmup5` 的前 5 个 epoch
- `encoder_frozen` 的前 5 个 epoch

在训练动力学上本质是一条相同的轨迹。

这也是为什么两组实验的：

- best checkpoint
- test 指标
- all 指标

会完全一致。

这个现象本身就是非常强的证据，说明：

- 当前最优结果来自 **frozen encoder 阶段**
- 而不是来自后续的 encoder 微调阶段

## 五、strongest baseline 的阶段性更新

截至 **2026-08-05**，项目的 strongest baseline 应作如下更新：

旧 strongest baseline：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05`

新的阶段性 strongest baseline：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05_encoder_frozen`

它的核心定义是：

- `gamma = 0.9`
- `binary reward`
- `Huber TD`
- `valid_ndcg_at_10` 选模
- `RL + CE` 联合目标
- `Q-head` 小尺度初始化
- encoder 全程冻结

## 六、这个更新对项目方向意味着什么

这次 strongest baseline 更新改变了我们对项目下一阶段的理解。

之前的理解更偏向：

- 想办法让 RL 逐步接管整个网络

而现在更合理的理解是：

- 先承认预训练 SASRec 表征非常强
- 在当前阶段，把它当成稳定状态编码器
- 让 offline RL 主要学习更强的价值头与排序头

也就是说，项目当前最可靠的路线不是：

- 强行让 RL 改写整个 encoder

而是：

- 先把 **fixed representation + stronger policy/value head** 这条线做扎实

## 七、当前最准确的阶段结论

当前这个项目到这一步，最准确的一句话总结是：

- **RL 不是没学到东西，而是它目前主要学到了“如何在固定 SASRec 表征上做更好的决策”，还没有学会“如何安全地微调表征本身”。**

这不是坏消息，反而是很清晰的结构性发现。

因为它告诉我们：

- 问题不在于 offline RL 完全无效
- 问题在于 encoder fine-tuning 这一步还没有被正确解决

这会直接决定后续真正有研究价值的方向：

- 保持 frozen encoder，继续升级 ranking-aware head/objective
- 或者在未来单独研究更安全的 encoder adaptation 机制
