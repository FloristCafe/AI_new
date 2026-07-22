# MovieLens 1M SASRec Capacity And Regularization 实验结果笔记

## 1. 这轮实验要回答什么

在 `baseline_triplet` 已经说明：

- `max_seq_len = 100` 明显退化
- `embedding_dim = 128` 小幅稳定优于 `64`

之后，项目的下一个关键问题变成了：

- 更大容量模型是否需要更强正则才能稳定发挥
- 当前的提升到底来自容量，还是来自正则控制

因此这轮 `capacity_and_regularization` 实验聚焦四组配置：

- `baseline_seq50_dim64_drop02`
- `seq50_dim64_drop05`
- `seq50_dim128_drop02`
- `seq50_dim128_drop05`

这轮实验的核心目的不是再去试历史窗口，而是判断：

- `dropout = 0.5` 是否有帮助
- `dim128` 的优势是否在更强正则下仍然成立

## 2. 实验结果

### 2.1 `baseline_seq50_dim64_drop02`

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

### 2.2 `seq50_dim64_drop05`

配置：

- `max_seq_len = 50`
- `embedding_dim = 64`
- `dropout = 0.5`

结果：

- `best_valid_hr_at_10 = 0.154139`
- `best_valid_ndcg_at_10 = 0.080565`
- `test_hr_at_10 = 0.135596`
- `test_ndcg_at_10 = 0.072659`
- `best_epoch = 20`

### 2.3 `seq50_dim128_drop02`

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

### 2.4 `seq50_dim128_drop05`

配置：

- `max_seq_len = 50`
- `embedding_dim = 128`
- `dropout = 0.5`

结果：

- `best_valid_hr_at_10 = 0.174007`
- `best_valid_ndcg_at_10 = 0.095332`
- `test_hr_at_10 = 0.153146`
- `test_ndcg_at_10 = 0.085742`
- `best_epoch = 18`

## 3. 结果排序

按 `test_ndcg_at_10` 看，四组结果从高到低为：

1. `seq50_dim128_drop02`：`0.102364`
2. `baseline_seq50_dim64_drop02`：`0.101169`
3. `seq50_dim128_drop05`：`0.085742`
4. `seq50_dim64_drop05`：`0.072659`

按 `best_valid_ndcg_at_10` 排序，顺序也是一致的。

这说明当前 valid 与 test 指标方向一致，结论具有较高可信度。

## 4. 这轮实验说明了什么

### 4.1 更大容量的优势仍然成立

`seq50_dim128_drop02` 仍然是四组中的最优配置，而且同时赢在：

- valid
- test

这说明第一轮 `baseline_triplet` 中观察到的“容量提升有效”并不是偶然。

当前项目可以比较稳定地给出一个判断：

**在 `seq_len = 50` 的前提下，`embedding_dim = 128` 比 `64` 更合适。**

### 4.2 `dropout = 0.5` 明显过强

这轮实验最重要的新信息其实不是“哪个更强”，而是：

**`dropout = 0.5` 在当前任务上明显过强。**

无论在 `dim64` 还是 `dim128` 下，`drop05` 都明显弱于对应的 `drop02`：

- `dim64`: `0.101169 -> 0.072659`
- `dim128`: `0.102364 -> 0.085742`

这说明更强正则并没有帮助当前模型更稳，反而伤害了表达能力和最终排序效果。

因此目前不应继续把 `dropout = 0.5` 作为主线方向。

### 4.3 这不是“需要更强正则”的项目状态

如果当前项目处于明显过拟合阶段，那么理论上：

- 更强正则应当改善 valid 或 test 表现

但现在看到的是：

- 强正则配置全部退化

这更像说明：

- 当前模型并不是“正则太弱”
- 而是“强正则过度削弱了序列表示能力”

所以目前最重要的问题不是“再加大 dropout”，而是：

- 在已经证明有效的 `dim128` 主线上做更细粒度的结构或训练策略升级

## 5. 关于训练行为的补充判断

### 5.1 `dim128_drop02` 更早达到最优

`seq50_dim128_drop02` 在 `epoch 12` 达到最佳，并触发了 early stopping。

这说明：

- 更大容量模型确实学得更快
- 当前 `best checkpoint` 机制很重要
- 不能简单假设“训练越久越好”

### 5.2 `drop05` 训练更慢，但更慢不代表更好

两个 `drop05` 配置都出现了：

- `best_epoch` 更晚
- `stopped_early = false` 或更晚停

这说明强正则让训练过程变慢了，但没有带来更好的最终效果。

因此不应把“训练更慢、更晚达到峰值”误判成“更稳”或“更好”。

## 6. 当前正式 baseline 应如何更新

到这一步，当前正式 baseline 可以明确更新为：

- `seq50_dim128_drop02`

原因很清楚：

- 它在 `baseline_triplet` 中最优
- 它在 `capacity_and_regularization` 中仍然最优
- 它的 valid/test 表现一致
- 它不依赖更激进的正则假设

这意味着后续实验不应再以：

- `baseline_seq50_dim64_drop02`

作为默认起点，而应改为：

- `seq50_dim128_drop02`

## 7. 当前不该继续做什么

基于这轮结果，当前阶段有两条路已经不值得再优先投入：

1. 不继续扩大 `dropout = 0.5` 这一方向  
因为无论小模型还是大模型，强正则都明确退化。

2. 不再回到 `seq_len = 100` 方向  
上一轮已经说明长序列窗口明显掉队。

这两条都已经有充分证据说明不应作为当前主线。

## 8. 当前最合理的下一步

基于现有两轮实验结果，当前最合理的下一步不是：

- 再继续试极端正则
- 或者再回头扩历史窗口

而是：

**以 `seq50_dim128_drop02` 为新主线 baseline，做更细粒度的容量与结构实验。**

比较自然的下一层选择包括：

- 更温和的 `dropout`，例如 `0.3` 或 `0.4`
- `num_blocks = 3`
- 进一步的训练目标升级，例如 negative sampling 或 ranking loss

如果继续按“先做最小可解释实验”的路线推进，那么下一步更推荐：

- 保持 `seq_len = 50`
- 保持 `dim128`
- 试更细粒度结构/正则，而不是跳太大

## 9. 当前阶段结论

这轮 `capacity_and_regularization` 已经把当前项目的主线进一步讲清楚：

1. `embedding_dim = 128` 的优势是真实且稳定的。
2. `dropout = 0.5` 在当前任务上过强，不适合作为主线。
3. 当前正式 baseline 应更新为 `seq50_dim128_drop02`。
4. 下一步应从“极端正则尝试”转向“更细粒度容量与结构优化”。

一句话总结：

**这轮实验证明了当前项目最值得保留的是更大的表示容量，而不是更强的 dropout；后续主线应围绕 `seq50_dim128_drop02` 展开，而不是继续在长序列或重正则方向上投入。**
