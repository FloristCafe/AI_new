# 2026-07-30 CE Regularization 推动当前主线刷新

## 本次记录的目标

这篇笔记记录一个非常关键的训练目标升级：

- 不再只用 `TD Loss + CQL`
- 而是在同一张计算图里，把监督学习的 `CrossEntropy` 重新加回来

本轮实验的核心思想是：

- `RL loss` 负责长期价值
- `CE loss` 负责维持排序边界

也就是说，我们不再让 `Q-head` 只学：

- “哪个动作长期值更高”

而是同时强制它继续保留：

- “真实下一点击在全量候选里的排序锐度”

对应实验目录为：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg01`

## 本轮代码改动是什么

修改文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\train_sasrec_dqn.py`

本轮把训练损失扩展成三部分：

1. `td_loss`
2. `cql_penalty`
3. `ce_loss`

具体形式为：

- `rl_loss = td_loss + alpha * cql_penalty`
- `ce_loss = cross_entropy(q_values, true_action)`
- `total_loss = rl_loss + lambda * ce_loss`

其中这轮实验里：

- `lambda = ce_regularization_weight = 0.1`

同时训练日志新增了：

- `train_rl_loss`
- `train_ce_loss`
- `valid_rl_loss`
- `valid_ce_loss`

这意味着后续我们不只知道“总损失变了多少”，还可以单独观察：

- RL 目标在做什么
- CE 排序正则在做什么

## 为什么这一步很重要

在前几轮实验里，我们已经看到一个持续出现的问题：

- 训练稳定以后
- 排序能力仍然容易在第 1 轮之后快速衰减

这说明当前的 `RL-only` 优化目标有一个明显短板：

- 它会逐渐把本来还不错的 next-item 排序结构磨钝

之前的 `Huber + binary reward + valid NDCG 选模` 已经把训练拉回可控区间，但还存在一个遗留问题：

- Q 值虽然不再爆炸
- 可模型的“排序感”仍然不够锋利

因此这次把 `CE loss` 加回来，本质上是在做一件工业界非常常见的事：

- 用监督排序目标作为行为先验
- 防止离线 RL 把表征空间过度扭曲

## 本轮实验配置

本轮大框架沿用当前主线候选，只额外加入 `CE regularization`：

- `gamma = 0.9`
- `td_loss_type = smooth_l1_huber`
- `reward = binary`
- `selection_metric = valid_ndcg_at_10`
- `ranking_topk = 10`
- `target_tau = 0.005`
- `adaptive_cql_alpha = true`
- `ce_regularization_weight = 0.1`

也就是说，这轮实验回答的是一个很干净的问题：

- 在当前最好的一条主线附近
- 只加一个轻量级监督排序正则
- 是否能把排序能力再往上推

## 训练侧结果

训练指标文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg01\metrics\training_metrics.json`

关键信息如下：

- `best_epoch = 1`
- `best_valid_ndcg_at_10 = 0.09576465781626146`
- `cql_alpha_final = 0.0963628182002476`
- `valid_max_q_value` 大致在 `11.5 ~ 23.8`

### 训练侧怎么解读

首先，它没有破坏当前已经建立起来的稳定性：

- 训练完整跑完
- 没有数值崩溃
- Q 值规模虽然不低，但仍在可控范围内

其次，加入 `CE regularization` 后，验证集最优排序指标继续提高：

- 上一版 `gamma=0.9` 无 CE：`best_valid_ndcg_at_10 = 0.09092615071867788`
- 本轮有 CE：`best_valid_ndcg_at_10 = 0.09576465781626146`

这说明 `CE` 没有和 RL 目标打架，反而确实补强了排序目标。

## 测试集结果

测试集评估文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg01\predictions\test_sasrec_dqn_metrics.json`

结果为：

- `HR@10 = 0.15248344370860928`
- `NDCG@10 = 0.08411850352739135`
- `top1_exact_hit_rate = 0.03278145695364238`

### 与上一版 gamma=0.9 无 CE 对比

上一版结果：

- `HR@10 = 0.1468543046357616`
- `NDCG@10 = 0.07904506965050907`
- `top1_exact_hit_rate = 0.02947019867549669`

本轮对比提升为：

- `HR@10` 上升
- `NDCG@10` 上升
- `top1 exact hit rate` 上升

这说明：

- `CE regularization` 不是只改善了“列表里的排序位置”
- 它连纯命中率也一起拉起来了

## all split 结果

全量回放评估文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg01\predictions\all_sasrec_dqn_metrics.json`

结果为：

- `HR@10 = 0.22950625094928528`
- `NDCG@10 = 0.12210019975062049`
- `top1_exact_hit_rate = 0.04326527984678661`
- `mean_cumulative_reward_per_user = 7.1213576158940395`

### 与上一版 gamma=0.9 无 CE 对比

上一版结果：

- `HR@10 = 0.22281925909981098`
- `NDCG@10 = 0.1170736908403192`
- `top1_exact_hit_rate = 0.040108874849245954`
- `mean_cumulative_reward_per_user = 6.60182119205298`

这一轮 all split 也出现了同步改善：

- `HR@10` 更高
- `NDCG@10` 更高
- `top1 exact hit rate` 更高
- 累计 reward 更高

这非常关键，因为它说明：

- CE 正则不是只在 test 单步上“偶然有效”
- 它对更长轨迹上的策略表现也有正向帮助

## 为什么可以把这版标成当前主线最强候选

和当前几条已验证主线相比，这一版的综合表现最好。

### 对比 gamma=0.7 + valid NDCG 选模

- 排序能力明显更强
- 训练同样稳定

### 对比 gamma=0.85

- `test NDCG@10` 更高
- `all` 指标更强

### 对比 gamma=0.9 无 CE

- `valid NDCG@10` 更高
- `test HR@10 / NDCG@10` 都更好
- `all` 指标也更好

因此在当前所有已跑结果中，这一版最合理的定位是：

- `current best candidate`

## 这轮实验最重要的认识

这轮实验最重要的价值，不只是“分数又涨了一点”，而是它验证了一个方向：

- 离线 RL 在推荐任务里，不能完全丢掉监督排序目标

更准确地说：

- `RL loss` 负责学长期价值
- `CE loss` 负责守住局部排序结构

二者不是互斥关系，而是当前这个项目里非常自然的互补关系。

这意味着项目下一步的优化思路，已经不再是单纯围绕：

- `gamma`
- `alpha`
- `tau`

做 RL 纯参数扫描，而是进入更像工业推荐系统常见的混合训练范式：

- `RL objective + supervised ranking regularization`

## 当前阶段结论

截至 2026-07-30，这轮实验应正式记录为：

- `CE regularization 有效`
- `当前主线刷新`
- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg01 成为当前 best candidate`

当前推荐把它作为项目主线继续往前推进。

## 项目下一步建议

既然 `ce_regularization_weight = 0.1` 已经被证明有效，下一步最自然的不是推翻它，而是做一个窄范围强度对照：

- `ce_regularization_weight = 0.3`

目标不是大改结构，而是回答一个很具体的问题：

- `0.1` 是否只是刚好有效
- 还是 CE 监督还能再加一点，从而进一步提升排序质量

也就是说，项目现在最适合进入：

- 围绕当前最强主线，做小步强度扫描

的阶段。
