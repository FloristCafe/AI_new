# 2026-08-04 多 seed 验证收口与 BPR 升级入口

## 本次阶段结论

当前 strongest baseline 已经不再只是单次偶然结果，而是经过多 seed 验证后可以正式收口的稳定版本。

对应基线为：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05`

多 seed 汇总文件位于：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\summaries\ce_reg05_multi_seed_7_42_2026.json`
- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\summaries\ce_reg05_multi_seed_7_42_2026.csv`

## 多 seed 结果怎么解读

本轮验证使用：

- `seed = 7`
- `seed = 42`
- `seed = 2026`

聚合结果显示：

- `best_valid_ndcg_at_10 mean = 0.111654`
- `test_hr_at_10 mean = 0.191887`
- `test_ndcg_at_10 mean = 0.103365`
- `all_hr_at_10 mean = 0.242706`
- `all_ndcg_at_10 mean = 0.126328`

更重要的是，几个关键指标的标准差都很小。这说明当前版本不是靠某一个 seed 碰巧跑出来的，而是已经具备了可复现实验基础。

## 为什么这个节点重要

这意味着项目可以从“先把系统训稳”正式转入“真正的方法升级”。

也就是说，后续的工作重点不应该继续停留在：

- 单纯调 `gamma`
- 单纯扫 `ce_regularization_weight`
- 单纯换 seed 反复确认

而应该开始做结构层升级。

## 本次代码升级

训练脚本 `train_sasrec_dqn.py` 已经从单一的 `CE regularization` 版本扩展为支持三种监督正则模式：

- `ce`
- `bpr`
- `none`

其中：

- `ce` 表示继续使用当前 strongest baseline 的监督排序约束
- `bpr` 表示引入 pairwise ranking 约束，让正样本分数直接压过采样负样本
- `none` 表示只保留 RL + CQL 主线，不额外加监督排序项

同时新增了：

- `--supervised-regularizer`
- `--bpr-negative-count`

并在训练日志中加入了以下监控项：

- `train_supervised_loss`
- `train_bpr_loss`
- `valid_supervised_loss`
- `valid_bpr_loss`

这意味着我们已经具备了第一条真正的方法升级入口，而不需要再重构整套训练器。

## 为什么先做 BPR，而不是立刻跳 IQL / AWAC

原因很简单：

- `BPR` 和当前项目的 next-item ranking 目标天然更贴近
- 它比直接换整套 offline RL 算法的工程风险更低
- 它能直接检验一个核心问题：当前系统的瓶颈，到底更多来自价值学习，还是来自排序边界表达

如果 `BPR` 能比 `CE` 进一步提升 `test HR@10 / NDCG@10`，那么说明当前阶段继续沿“排序感增强”这条线深挖是有价值的。

如果 `BPR` 没有带来提升，反而退化，那么这也是非常有价值的负结果，因为它会指向下一阶段应该考虑真正的策略约束式 offline RL，例如：

- `IQL`
- `AWAC`

## 当前最合理的下一步

下一步不是继续扫普通参数，而是做一轮清晰的对照实验：

1. 保持 strongest baseline 其余设置不变
2. 仅把监督正则从 `ce` 切换成 `bpr`
3. 先测试较小负采样规模，例如 `1 / 3 / 5`
4. 对比：
   - `best_valid_ndcg_at_10`
   - `test_hr_at_10`
   - `test_ndcg_at_10`
   - `all_hr_at_10`
   - `all_ndcg_at_10`
   - `top1_exact_hit_rate`

只有完成这轮对照，项目才算真正迈出了“从稳固 baseline 到方法创新”的第一步。
