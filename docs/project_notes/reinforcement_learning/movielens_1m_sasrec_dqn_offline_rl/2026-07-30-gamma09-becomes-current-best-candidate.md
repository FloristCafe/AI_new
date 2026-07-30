# 2026-07-30 gamma=0.9 成为当前主线候选

## 本次记录的目标

这篇笔记记录 `Huber + binary reward + valid NDCG 选模` 这条路线下，`gamma` 调参实验的最新结论。

在前一阶段，我们已经确认了两件事：

- `valid_ndcg_at_10` 选模是必须保留的
- `gamma=0.7` 虽然能压住训练，但会把模型推得过于保守

因此后续实验的核心目标变成：

- 在不回退训练稳定性的前提下
- 逐步把 `gamma` 从 `0.7` 往上调
- 找到排序质量和数值稳定性之间更合理的平衡点

本轮记录的是：

- `gamma=0.9`

对应实验目录为：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg`

## 本轮实验配置

本轮核心配置保持不变，只改一个关键变量：

- `gamma = 0.9`

其余保持为：

- `td_loss_type = smooth_l1_huber`
- `reward = binary`
- `selection_metric = valid_ndcg_at_10`
- `ranking_topk = 10`
- `target_tau = 0.005`
- `adaptive_cql_alpha = true`
- `encoder_learning_rate = 1e-6`
- `q_head_learning_rate = 3e-4`

这意味着本轮实验是一个非常干净的对照：

- 不再混入多余改动
- 专门回答“把 horizon 从 0.85 再放长到 0.9，会不会继续改善排序能力”

## 训练侧结果

训练指标文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg\metrics\training_metrics.json`

关键信息如下：

- `best_epoch = 1`
- `best_valid_ndcg_at_10 = 0.09092615071867788`
- `cql_alpha_final = 0.06973711168284089`
- `valid_max_q_value` 大体在 `12 ~ 23` 区间
- 训练完整跑完 `10` 个 epoch，没有早停，也没有非数值异常

### 这说明什么

首先，`gamma=0.9` 并没有把系统重新推回最早那种 value explosion 状态。

虽然它的 Q 值规模相比 `gamma=0.85` 更高，但目前仍处于可控区间：

- 没有出现 `max_q_value` 几十到上百持续无界抬升
- `adaptive alpha` 也没有完全失效
- 训练过程可稳定完成

换句话说：

- 我们在放长 horizon 的同时
- 还没有丢掉此前通过 `Huber + binary reward + soft update` 获得的稳定性收益

## 验证集 ranking 结果

这一轮最关键的验证指标是：

- `best_valid_ndcg_at_10 = 0.09092615071867788`

这比上一轮 `gamma=0.85` 的：

- `0.08824547411332986`

更高。

也就是说，在当前训练框架下，单从验证集 NDCG 看：

- `gamma=0.9` 已经优于 `gamma=0.85`

同时它依然延续了这条路线的一个现象：

- 最强 ranking 能力仍然出现在 `epoch 1`

这再次说明：

- 目前训练目标继续往后优化，仍然会逐渐偏离真正的推荐排序目标
- `valid_ndcg_at_10` 选模必须继续保留

## 测试集结果

测试集评估文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg\predictions\test_sasrec_dqn_metrics.json`

结果为：

- `HR@10 = 0.1468543046357616`
- `NDCG@10 = 0.07904506965050907`
- `top1_exact_hit_rate = 0.02947019867549669`

### 与 gamma=0.85 对比

`gamma=0.85` 的测试集结果为：

- `HR@10 = 0.15149006622516556`
- `NDCG@10 = 0.07718725614916837`
- `top1_exact_hit_rate = 0.024172185430463577`

对比可见：

- `HR@10` 略降
- `NDCG@10` 小幅上升
- `top1 exact hit rate` 上升

这说明 `gamma=0.9` 更偏向：

- 优化排序质量
- 改善命中项在 top-k 中的位置分布

而不是单纯追求更高的 top-k 命中率。

## 全轨迹 all split 结果

全量回放评估文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg\predictions\all_sasrec_dqn_metrics.json`

结果为：

- `HR@10 = 0.22281925909981098`
- `NDCG@10 = 0.1170736908403192`
- `top1_exact_hit_rate = 0.040108874849245954`
- `mean_cumulative_reward_per_user = 6.60182119205298`

### 与 gamma=0.85 对比

`gamma=0.85` 的 all split 结果为：

- `HR@10 = 0.20845852163968098`
- `NDCG@10 = 0.10598234940855593`
- `top1_exact_hit_rate = 0.03354057509336944`
- `mean_cumulative_reward_per_user = 5.520695364238411`

这说明在整条轨迹层面，`gamma=0.9` 的改善更加明确：

- `HR@10` 更高
- `NDCG@10` 更高
- `top1 exact hit rate` 更高
- 累计 reward 也更高

也就是说，如果不只盯住单步 test，而是从更完整的轨迹角度看：

- `gamma=0.9` 明显强于 `gamma=0.85`

## 与 gamma=0.7 的关系

`gamma=0.7` 的那一版虽然在训练稳定性上很好，但排序能力太弱：

- `test HR@10 ≈ 0.1137`
- `test NDCG@10 ≈ 0.0527`

而 `gamma=0.9` 已经提升到：

- `test HR@10 ≈ 0.1469`
- `test NDCG@10 ≈ 0.0790`

这进一步确认：

- `0.7` 的 horizon 对当前问题来说确实过短
- 过强的短视会把推荐排序能力压掉

## 为什么 gamma=0.9 可以暂时定为当前主线候选

当前把 `gamma=0.9` 标为这条路线的主线候选，理由主要有四个。

### 1. 它的 valid NDCG 最好

在已经跑过的 `0.7 / 0.85 / 0.9` 三轮中：

- `0.9` 的 `best_valid_ndcg_at_10` 最高

### 2. 它的 test NDCG 最好

虽然 `test HR@10` 不如 `0.85` 略高，但：

- `test NDCG@10` 更高

而在 sequential recommendation 里，NDCG 更能反映：

- 真实命中项在推荐列表中的排序质量

### 3. 它的 all split 整体表现最强

从全轨迹角度看，`0.9` 比 `0.85` 更强，这对当前离线 RL 项目尤其重要，因为它说明：

- 模型不只是单个 test step 更像样
- 在更长回放范围内也更有稳定收益

### 4. 它还没有重新爆炸

这是最重要的前提。

如果 `0.9` 把 Q 值重新推回明显失控区间，那么即便 ranking 指标略有提升，也不能作为主线候选。

但当前情况是：

- Q 值上去了
- 但还没有炸
- 指标也同步更好

所以它处于一个更合理的平衡点。

## 但为什么它还不能直接宣布为最终最佳版本

虽然本轮可以标记为“当前主线候选”，但还不能草率宣告为最终最佳版本，原因有两个。

### 1. test HR@10 略低于 gamma=0.85

这意味着：

- `0.9` 并不是在所有指标上都绝对占优

它更像是：

- 用更好的排序质量
- 换来略低一点的纯命中率

### 2. valid/test 最优仍然都在 epoch 1

这说明当前训练目标仍然有一个根本问题没有彻底解决：

- 继续训练下去，模型会逐渐偏离排序目标

也就是说，虽然 `0.9` 是当前更好的参数点，但训练目标本身依然没有被完全修好。

## 当前阶段结论

截至 2026-07-30，在 `Huber + binary reward + valid NDCG 选模` 这条路线下：

- `gamma=0.7` 过于保守
- `gamma=0.85` 明显恢复排序能力
- `gamma=0.9` 进一步提升了 ranking 质量与全轨迹表现

因此，本轮实验应正式标记为：

- `gamma=0.9 成为当前主线候选`

它的定位不是“已经终局”，而是：

- 当前这条路线下最值得继续沿着往前推进的版本

## 对项目下一步的意义

这轮实验之后，项目下一步不应该再回退到：

- `gamma=0.7`

也不应再反复争论：

- 要不要继续用 valid ranking 选模

因为这两件事已经基本有结论了。

当前真正值得继续做的是：

1. 以 `gamma=0.9` 作为当前主线候选继续记录。
2. 如果还要继续扫参，下一步只试更小范围微调，例如：
   - `gamma=0.92`
   - `gamma=0.95`
3. 同时继续盯住：
   - `valid_ndcg_at_10`
   - `test_ndcg_at_10`
   - `valid_max_q_value`
   - `all split` 指标

也就是说，项目现在已经从“先把系统救活”阶段，进入：

- 围绕一个已可用主线候选做窄范围精修

的阶段。
