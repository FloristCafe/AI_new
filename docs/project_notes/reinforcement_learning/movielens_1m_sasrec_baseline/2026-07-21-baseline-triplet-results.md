# MovieLens 1M SASRec Baseline Triplet 实验结果笔记

## 1. 这轮实验要回答什么

在 `movielens_1m_sasrec_baseline` 已经跑通并具备验证闭环之后，第一轮最小系统化消融实验的目标不是直接刷分，而是回答一个更关键的问题：

- 当前瓶颈更像是历史窗口不够，还是模型容量不够。

因此这轮实验只比较三组最关键、最容易解释的配置：

- `baseline_seq50_dim64_drop02`
- `seq100_dim64_drop02`
- `seq50_dim128_drop02`

这三组配置分别代表：

- 当前正式 baseline
- 更长的历史窗口
- 更大的表示容量

## 2. 实验结果

### 2.1 Baseline

配置：

- `max_seq_len = 50`
- `embedding_dim = 64`
- `dropout = 0.2`

结果：

- `best_valid_hr_at_10 = 0.194702`
- `best_valid_ndcg_at_10 = 0.106844`
- `test_hr_at_10 = 0.175993`
- `test_ndcg_at_10 = 0.101169`
- `best_epoch = 17`

### 2.2 更长历史窗口

配置：

- `max_seq_len = 100`
- `embedding_dim = 64`
- `dropout = 0.2`

结果：

- `best_valid_hr_at_10 = 0.125000`
- `best_valid_ndcg_at_10 = 0.068591`
- `test_hr_at_10 = 0.108444`
- `test_ndcg_at_10 = 0.057762`
- `best_epoch = 19`

### 2.3 更大表示容量

配置：

- `max_seq_len = 50`
- `embedding_dim = 128`
- `dropout = 0.2`

结果：

- `best_valid_hr_at_10 = 0.197682`
- `best_valid_ndcg_at_10 = 0.108874`
- `test_hr_at_10 = 0.179470`
- `test_ndcg_at_10 = 0.102364`
- `best_epoch = 12`

## 3. 这轮实验说明了什么

### 3.1 当前更缺容量，不缺更长历史

最重要的结论是：

**当前瓶颈更像是模型容量，而不是历史长度。**

理由很直接：

- `embedding_dim 64 -> 128` 在 valid 和 test 上都带来了小幅但一致的提升
- `max_seq_len 50 -> 100` 在 valid 和 test 上都出现了明显退化

这说明对当前这版数据构造、训练目标和模型实现来说：

- 历史窗口从 50 拉到 100 没有带来有效信息增益
- 反而更可能引入了额外噪声、优化难度或无效长依赖
- 但把表示容量从 64 扩到 128，模型是能够真正利用起来的

### 3.2 `seq_len = 100` 不是“暂时没赢”，而是明显不合适

需要特别明确一点：

`seq100_dim64_drop02` 不是“差一点赢”，而是明显掉队。

它相对 baseline 的下降幅度很大：

- `valid_ndcg_at_10`: `0.106844 -> 0.068591`
- `test_ndcg_at_10`: `0.101169 -> 0.057762`

这说明当前阶段不应该继续优先往“更长历史窗口”这个方向推进，否则很可能是在错误方向上继续花时间。

### 3.3 `dim128` 目前可以视为新的主线 baseline

`seq50_dim128_drop02` 在三组里最强，而且优势在 valid 和 test 上同时出现。

虽然提升幅度不是极端大，但它是：

- 稳定的
- 可解释的
- 符合预期的

因此当前完全可以把：

- `seq50_dim128_drop02`

视为新的正式 baseline，而不再把：

- `seq50_dim64_drop02`

继续当作后续实验的默认出发点。

## 4. 关于训练行为的补充判断

还有两个训练层面的信号值得记录。

### 4.1 baseline 在第 17 轮达到最佳

这说明：

- 20 epoch 的预算是合理的
- early stopping 与 best checkpoint 机制是必要的
- 当前项目已经进入“训练协议开始影响正式结果”的阶段

### 4.2 `dim128` 在第 12 轮就达到最好

这说明：

- 更大容量模型虽然更强，但也更早到达最优点
- 后续如果继续增大容量，正则化的重要性会更高

这正好说明下一轮实验不该继续盲目加容量，而该开始检查：

- `dropout`
- 训练稳定性
- regularization 是否足够

## 5. 当前最合理的下一步

基于这轮实验，当前最合理的下一步不是：

- 继续拉长 `max_seq_len`
- 或直接跳去更复杂结构

而是：

**围绕新的 `dim128` baseline 做第二轮容量与正则化实验。**

最自然的下一组实验是：

- `baseline_seq50_dim64_drop02`
- `seq50_dim64_drop05`
- `seq50_dim128_drop02`
- `seq50_dim128_drop05`

也就是当前 runner 里的：

- `capacity_and_regularization`

这组实验将回答：

- `dropout 0.5` 是否能提升稳定性
- `dim128` 的优势是否在更强正则下仍然成立
- 当前最优配置究竟是“更大容量”，还是“更大容量 + 更强正则”

## 6. 当前阶段结论

这轮 `baseline_triplet` 已经给出了足够清晰的方向判断：

1. 当前项目不应优先往更长历史窗口推进。
2. 当前最有效的提升来自更大的表示容量。
3. `seq50_dim128_drop02` 可以被视为新的正式 baseline。
4. 下一阶段应优先进入容量与正则化实验，而不是继续探索更长序列。

一句话总结：

**这轮实验已经把“下一步该沿哪条主线继续优化”讲清楚了：先稳住 `seq_len=50`，把注意力转向 `embedding capacity + regularization`，而不是继续在长历史窗口上投入。**
