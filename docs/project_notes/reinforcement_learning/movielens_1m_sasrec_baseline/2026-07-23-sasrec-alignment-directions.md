# MovieLens-1M SASRec 对齐方向整理

生成日期：2026-07-23

这份文档用于整理当前 `movielens_1m_sasrec_baseline` 与论文级 SASRec 复现之间仍需对齐的方向、缺口和优先级。

## 一、当前已经做到的部分

- 已经完成 MovieLens-1M 数据清洗、时间排序、leave-one-out 切分和固定长度序列构造。
- 已经完成 item embedding、position embedding、causal masking、自注意力编码和 next-item ranking 的完整 baseline 链路。
- 已经补上 valid 集评估、best checkpoint 选择、early stopping 和 experiment summary。
- 已经形成两轮消融结论：`seq_len=100` 退化，`embedding_dim=128` 优于 `64`，`dropout=0.5` 过强。

## 二、当前实现与论文级 SASRec 的核心差距

### 1. 训练目标还没有对齐论文

当前实现使用的是 `full-item softmax + cross-entropy`，也就是让模型对所有 item 同时打分，再通过 softmax 学习真实下一 item 的概率。

论文原味 SASRec 更接近 `point-wise BCE + negative sampling`：对每个时间步保留一个正样本，再随机采若干负样本，让模型学会把正样本打得更高。

- 这意味着当前模型虽然是 `SASRec-style`，但训练信号与论文版本并不一致。
- 这也是为什么当前实验现象不一定会和论文表格完全一致。

### 2. 训练形式还没有完全做成 shifted-sequence 多位置监督

当前项目本质上仍偏向 `history -> next item` 的样本式训练。论文版 SASRec 更强调在一个序列中对多个时间步同时做监督，也就是输入和目标整体右移一位，形成更原味的自回归训练。

- 当前实现可运行、可做实验，但还不是最原味的论文训练形态。

### 3. 模型结构还没有完全手写对齐

当前实现大量依赖 PyTorch 的 `TransformerEncoder`。它在工程上合理，但如果目标是手写复现论文，则应把 self-attention block、point-wise feed-forward、residual、layer norm、dropout 的位置显式写出来。

- 当前模型属于 `SASRec-style` 实现。
- 目标若是论文级复现，则应升级为 block-level 对齐实现。

### 4. 评估协议虽然合理，但还没有做论文向的扩展

当前项目已经采用 `leave-one-out + HR@10 / NDCG@10`，这一步是正确的。但若目标是论文级复现，后续还应补更标准的 sampled negative evaluation、多 seed 汇总，以及更完整的实验表。

### 5. 实验趋势还没有和论文经验现象对齐

当前实验里，`seq_len=100` 相比 `50` 明显退化，而论文语境下更长序列窗口通常不应直接大幅恶化。这说明当前实现中至少有一部分因素仍未与论文版本对齐。

- 问题可能来自训练目标。
- 问题可能来自结构细节。
- 问题也可能来自超参数或采样策略。

## 三、需要对齐的方向

### 方向 A：训练目标对齐

1. 把当前 `full softmax + cross-entropy` 改成更接近论文的 `BCE + negative sampling`。
2. 明确正样本与负样本的构造方式，并把采样过程写进训练循环。
3. 比较改造前后的 valid / test 指标，以及序列长度实验趋势是否更合理。

### 方向 B：训练形式对齐

1. 把当前偏样本式的 next-item 训练，升级为更原味的 shifted-sequence 多位置监督。
2. 让一个训练序列在多个时间步同时产生监督信号，而不是主要盯最后一个位置。

### 方向 C：结构实现对齐

1. 手写 SASRec attention block，而不是主要依赖 `TransformerEncoder`。
2. 显式实现 self-attention、point-wise FFN、residual、layer norm 和 dropout。
3. 让代码结构更接近论文描述，便于后续做模块消融和可视化。

### 方向 D：实验协议与论文结果对齐

1. 补多 seed 实验，避免只依赖单次结果。
2. 补 sampled negative evaluation 或与论文更接近的评估设置。
3. 复现更完整的消融：序列长度、模块作用、训练效率、注意力可视化。

## 四、当前阶段最推荐的优先级

如果目标是沿着“手写复现顶会论文 SASRec”的路线推进，当前最推荐的优先级如下：

1. 先对齐训练目标：从 `full softmax + CE` 转向 `BCE + negative sampling`。
2. 再对齐训练形式：从当前 `history -> next item` 训练，转向更原味的 shifted-sequence 多位置监督。
3. 然后对齐结构细节：手写 SASRec block。
4. 最后再做论文向实验表与趋势复现。

## 五、当前项目状态的准确定位

当前项目已经不是一个玩具脚本，而是一个完整的 Sequential RecSys baseline 工程。它已经具备：

- 可运行的数据链路
- 可比较的实验体系
- 有效的 early stopping / best checkpoint
- 两轮有结论的消融实验

但如果目标是“手写复现顶会论文 SASRec”，目前还应该把它看作一个高质量工程 baseline，而不是论文级严格复现版本。

## 六、简明结论

一句话总结：当前最需要对齐的不是再继续粗调超参数，而是训练目标、训练形式、结构细节和实验协议。只有把这些层面对齐，项目才会真正从 `SASRec-style baseline` 走向论文级 SASRec 复现。
