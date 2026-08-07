# 2026-08-06 Frozen-Encoder Baseline 之后的方法升级路线

## 当前起点

截至 **2026-08-06**，项目的 strongest baseline 已经明确收束为：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05_encoder_frozen`

这条线已经说明：

- `SASRec encoder` 作为固定状态编码器是强的
- `RL + CE` 在固定表征上训练 `Q-head` 是有效的
- 当前没有证据表明 RL 微调 encoder 会进一步提升效果

因此，下一阶段的主问题不再是：

- 要不要继续解冻 encoder

而是：

- 在 **fixed representation** 前提下，如何让决策头和训练目标更贴近排序本质

## 总体路线判断

接下来不建议把主要精力继续投入在：

- 解冻策略来回试
- encoder 学习率小数点后再调一位
- 普通超参数穷举

更值得做的是沿着一条更清晰的研究主线推进：

- **Frozen encoder + stronger ranking-aware objective / policy head**

## 第一优先级：BPR / pairwise ranking 正则

这是当前最自然、工程风险最低、信息增益最高的一步。

原因有三点：

### 1. 它直接对齐“排序”而不是只对齐“分类”

当前 CE regularization 的本质是：

- 让真实 item 的 logit 变大

而 BPR 的本质是：

- 让真实 item 的分数直接压过负样本分数

这更接近推荐系统真正关心的 pairwise ordering。

### 2. 它和 frozen encoder 的结构天然兼容

当前 encoder 已经不打算继续改写，那么最值得动手的地方就是：

- Q-head 的训练目标

BPR 正好属于这一层的升级，而不要求重构整个框架。

### 3. 它能直接检验“当前瓶颈是不是排序边界”

如果 BPR 比 CE 更强，说明当前系统的下一步核心增益点，确实在：

- 排序目标表达

如果 BPR 不强，甚至退化，那就说明问题不只是排序损失，还可能需要进入更深层的离线 RL 方法升级。

## 第二优先级：更强的监督锚，而不是更强的 encoder 微调

如果 BPR 有效，后面可以继续沿着“更强监督锚点”走，而不是回头继续折腾 encoder。

可能的方向包括：

- `CE + BPR` 混合正则
- margin ranking loss
- 更精细的 hard negative sampling
- 基于候选集的 ranking loss

这一整条线的共同思想是：

- 先承认 encoder 足够强
- 把主要创新集中在动作排序头与损失函数

## 第三优先级：真正的 offline RL 方法升级

只有在 ranking-aware 头部升级之后，才值得正式切到更重的方法线，例如：

- `IQL`
- `AWAC`

原因很简单：

- 如果当前系统的主要瓶颈其实还在排序边界
- 那么过早切到更复杂的 offline RL 算法，可能只是把噪声复杂化

更合理的顺序应该是：

1. 先确认 frozen encoder 下，head-level ranking upgrade 能走多远
2. 再判断是否需要 policy-constrained offline RL

## 下一步实验的最合理定义

当前最值得做的，不是三五条分散小试，而是一轮非常明确的对照：

### A. baseline

- frozen encoder
- `RL + CE`

### B. method upgrade

- frozen encoder
- `RL + BPR`

保持其他关键设置不变：

- `gamma = 0.9`
- `binary reward`
- `Huber TD`
- `valid_ndcg_at_10` 选模
- `Q-head` 小尺度初始化

首轮只需要比较少量负采样规模即可，例如：

- `negative_count = 1`
- `negative_count = 3`

## 评估重点

下一阶段不要只看一个 test 指标，而要同时看三层结果：

### 1. valid 选模层

- `best_epoch`
- `best_valid_ndcg_at_10`

### 2. test 推荐层

- `test HR@10`
- `test NDCG@10`
- `top1_exact_hit_rate`

### 3. all 轨迹层

- `all HR@10`
- `all NDCG@10`
- `mean_cumulative_reward_per_user`

如果某个方法只提升了单步指标，却明显伤害整条轨迹收益，就不能轻易视为下一阶段主线。

## 当前阶段的简化决策

如果只用一句话概括现在最应该做什么，那就是：

- **不要再围绕 encoder 解冻来回打转，直接把 frozen encoder 视为稳定底座，优先验证 BPR 这种真正贴近排序本质的方法升级。**

这才是从“把 baseline 做稳”走向“开始做方法”的最自然下一步。
