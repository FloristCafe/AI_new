# 2026-08-08 BPR Frozen-Encoder 对照实验结论

## 本次实验想回答什么问题

在 strongest baseline 已经更新为 frozen encoder 版本之后，项目进入了第一条真正的方法升级线：

- 保持 `SASRec encoder` 冻结
- 将监督正则从 `CE` 切换到 `BPR`

这轮实验的核心问题是：

- 在 fixed representation 前提下，pairwise ranking loss 是否能比当前 `CE` 正则更好地提升离线序列推荐效果？

为此，我们做了两组最直接的对照：

- `BPR ns=1`
- `BPR ns=3`

其中：

- `ns` 表示 `negative sampling`
- `ns=1` 表示每个正样本配 1 个负样本
- `ns=3` 表示每个正样本配 3 个负样本

## 对照基线是谁

这轮对照的参照物不是早期不稳定版本，而是当前 strongest baseline：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05_encoder_frozen`

它对应的关键结果为：

- `best_valid_ndcg_at_10 = 0.12605225919047516`
- `test HR@10 = 0.2099337748344371`
- `test NDCG@10 = 0.1166359124280432`
- `all HR@10 = 0.2737029619712544`
- `all NDCG@10 = 0.14818211838997059`
- `all mean_cumulative_reward_per_user = 9.029304635761589`

## 实验结果

### 1. BPR `ns=1`

实验目录：

- `cql_huber_gamma09_binary_reward_valid_ndcg_bpr_ns1_encoder_frozen`

结果：

- `best_epoch = 3`
- `best_valid_ndcg_at_10 = 0.1180249373168396`
- `test HR@10 = 0.19834437086092715`
- `test NDCG@10 = 0.10869828663512052`
- `all HR@10 = 0.2601841336835085`
- `all NDCG@10 = 0.13865339132929885`
- `all mean_cumulative_reward_per_user = 8.102649006622517`

### 2. BPR `ns=3`

实验目录：

- `cql_huber_gamma09_binary_reward_valid_ndcg_bpr_ns3_encoder_frozen`

结果：

- `best_epoch = 3`
- `best_valid_ndcg_at_10 = 0.11757435422942454`
- `test HR@10 = 0.19751655629139073`
- `test NDCG@10 = 0.1088414074103438`
- `all HR@10 = 0.2596580661839184`
- `all NDCG@10 = 0.13845286285528144`
- `all mean_cumulative_reward_per_user = 8.102317880794702`

## 结果怎么解读

这次结论非常清楚，没有暧昧空间。

### 1. 纯 BPR 没有超过 CE frozen baseline

无论是 `ns=1` 还是 `ns=3`，都在关键指标上落后于当前 strongest baseline。

也就是说：

- 纯 `BPR` 不能替代当前的 `CE` 主锚

这不是单一指标上的偶然回落，而是：

- `valid`
- `test`
- `all`

三层指标一起回落。

### 2. `ns=1` 和 `ns=3` 差别很小

两组 BPR 的结果几乎贴在一起，说明当前瓶颈并不在于：

- 负样本数量不够

也就是说，问题不是“BPR 方向对，但 `ns` 还没调好”，而更像是：

- 在当前任务设定下，纯 pairwise ranking objective 本身不如 CE 稳定有效

### 3. BPR 的最优点更早，后续优化潜力更弱

两组 BPR 的：

- `best_epoch = 3`

而 strongest baseline 的 frozen-encoder CE 版本：

- `best_epoch = 5`

这说明纯 BPR 线不只是最终分数低，而且训练过程本身也更早见顶。

换句话说：

- 它没有展现出比 CE 更好的持续优化能力

## 为什么会这样

当前最合理的解释不是“BPR 完全错误”，而是：

### 1. CE 提供的是更强的全局排序锚

在当前这个 next-item 推荐问题里，`CE` 的作用是：

- 直接在全量 item 空间里拉高真实 item 的 logit

这会形成非常强的全局分类边界。

而 `BPR` 更像是：

- 让正样本只需压过若干个采样负样本

这是一种局部 pairwise 约束。

当前实验说明，在这个项目的现阶段：

- 全局分类式约束比局部 pairwise 约束更强

### 2. Frozen encoder 下，Q-head 可能更依赖稳定的大范围分类边界

因为 encoder 已被冻结，不再继续学习表征，那么 head 的训练质量就更依赖监督目标本身是否足够强。

当前结果显示：

- `CE` 更适合在固定表征上训练出稳定有效的决策头
- `BPR` 单独使用时约束不够强

## 本轮实验的正式结论

截至 **2026-08-08**，可以正式收口为：

- 纯 `BPR` 不能替代当前的 `CE frozen baseline`
- `BPR ns=1` 与 `BPR ns=3` 都落后于 `CE frozen baseline`
- 当前 strongest baseline 仍然保持为  
  `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05_encoder_frozen`

## 项目下一步该做什么

这轮实验之后，不建议继续围绕：

- `BPR ns=5`
- `BPR ns=10`
- 更细碎的纯 BPR 扫描

继续投入时间。

因为当前证据已经足够说明：

- 纯 BPR 不是这条线的最优主方向

下一步更合理的升级方向是：

- 保留 `CE` 作为主锚
- 将 `BPR` 作为补充项加入

也就是进入：

- **`CE + BPR` 混合正则**

这会比继续做“纯 BPR 替代 CE”的小修小补更有信息增益。

## 一句话总结

这轮 frozen-encoder BPR 对照实验的最终结论是：

- **在当前 MovieLens-1M 离线序列推荐项目里，纯 BPR 不如 CE；下一步不该继续问“BPR 能不能替代 CE”，而该问“BPR 能不能作为第二个排序锚，与 CE 形成互补”。**
