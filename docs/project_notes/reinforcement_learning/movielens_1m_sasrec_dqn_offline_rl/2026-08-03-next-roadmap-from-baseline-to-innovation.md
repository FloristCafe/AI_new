# 2026-08-03 后续路线图：从基线精修到方法创新

## 这份路线图要回答什么

当前项目已经有了一个相对可信的主线最优候选：

- `cql_huber_gamma09_binary_reward_valid_ndcg_ce_reg05`

所以接下来最重要的问题已经不是：

- “项目还能不能跑”

而是：

- “下一步到底该继续精修什么”
- “哪些属于必要收口”
- “哪些才是真正值得写进项目亮点的方法创新”

这份路线图按三个层级来拆：

1. 短期：把当前 strongest baseline 收口
2. 中期：做训练目标与离线约束升级
3. 长期：做更像研究型项目的方法创新

## 一、短期路线：把当前主线 baseline 收口

这一层的目标不是发明新范式，而是把当前最强主线的结论做实。

### 1. CE 权重扫描收口

当前已经跑过并确认有效的点包括：

- `ce = 0.1`
- `ce = 0.3`
- `ce = 0.5`

下一步最自然的是：

- 试 `ce = 0.4`

目的不是盲扫，而是判断最优点是否落在：

- `0.3 ~ 0.5`

之间。

这一步完成后，`RL + CE` 这条主线就可以基本定型。

### 2. 做多 seed 稳定性确认

当前所有结论几乎都还是：

- `seed = 42`

如果想把当前 best candidate 变成真正可信的 strongest baseline，至少要补：

- `seed = 7`
- `seed = 42`
- `seed = 2026`

然后比较：

- `mean/std of test HR@10`
- `mean/std of test NDCG@10`
- `mean/std of all NDCG@10`

这一步非常重要，因为它决定后面的方法创新是不是建立在稳定结论上。

### 3. 选一个正式 baseline 版本封板

在完成 `ce` 收口和多 seed 之后，应当明确冻结一个版本，作为后续所有创新的参照基线。

理想状态下，这个基线应包含：

- 固定参数
- 固定选模方式
- 固定评估脚本
- 固定命名

这一步的意义是：

- 后面的新方法必须和一个清晰、稳定、可复现实验对象比较

## 二、中期路线：训练目标与离线约束升级

这一层开始从“调强基线”进入“改变方法”的范畴。

### 1. 从 CE regularization 升级为更合适的排序监督

当前 `CE` 已经有效，但它仍然是最朴素的全量分类形式。

下一步可以考虑：

- `BPR / pairwise ranking loss`
- `sampled softmax`
- `hard negative aware` 的 ranking loss

这条线的核心目标是：

- 不只是保持排序感
- 而是让排序监督本身更贴近推荐任务结构

### 2. 引入行为约束更强的 offline RL 目标

当前主线仍然是：

- `DQN/CQL-style value learning`

这对于离线推荐当然有效，但还不是最适合这个问题的唯一选择。

下一步非常值得做的升级方向是：

- `IQL`
- `AWAC`
- `TD3+BC` 风格的离线约束

原因很明确：

- 推荐场景的动作空间大
- 离线日志里 OOD action 问题严重
- 纯 DQN 容易在 unseen action 上产生错误价值估计

这条线是真正从“参数调优”迈向“算法升级”的重要分水岭。

### 3. 让 CE 权重从常数变成调度项

当前我们用的是：

- 固定 `ce_regularization_weight`

但更合理的做法可能是：

- 训练早期强 CE，稳住排序边界
- 训练后期逐步降低 CE，让 RL 目标获得更大自由度

比如可以试：

- linear decay
- cosine decay
- warmup + plateau

这条路线的价值在于：

- 把“监督保持排序”和“RL 学长期价值”做成时间上的分工

## 三、长期路线：更像研究型项目的方法创新

这一层才是真正更像论文复现升级、项目亮点提炼、面试/简历可讲故事的部分。

### 1. Dual-head 架构

当前我们其实已经在损失层面做了双目标，但结构上仍然是：

- 一个 `Q-head`

后面完全可以升级成：

- `value/Q head`
- `policy/ranking head`

两个 head 共享 `SASRec encoder`，但各自负责不同目标。

这样做的好处是：

- ranking head 保持推荐结构
- value head 学长期回报
- 二者不需要完全挤在同一个 logit 空间里

这是非常自然、也很有项目亮点的一条线。

### 2. 更合理的状态表示读取方式

当前主线主要依赖：

- 序列最后一个 hidden state

这在工程上简洁，但未必总是最优。

后续可以做对照：

- attention pooling
- gated pooling
- last-k hidden fusion

这条线属于：

- 在不推翻主架构的前提下做结构创新

### 3. Offline buffer 的结构升级

当前 buffer 主要是：

- 单步 `(s, a, r, s')`

后面可以升级为：

- `n-step return`
- sequence chunk replay
- hard negative augmented transition
- constrained candidate action replay

这条线的创新点在于：

- 不只是改模型
- 而是从数据与经验组织方式上提高离线 RL 的信号质量

### 4. 更贴近工业推荐的候选约束

当前动作空间仍然偏“全量打分”。

但真实推荐系统里通常是：

- 先召回候选
- 再重排

所以后面可以考虑做一个更贴近工业的二阶段版本：

- SASRec / retrieval 先给候选集合
- RL 只在候选集合内做 value-aware reranking

这条线很适合写成：

- “从学术型离线 RL baseline 向工业可落地推荐架构过渡”

的项目亮点。

## 四、建议的执行顺序

为了避免路线发散，后续推进建议按下面顺序走。

### 第一阶段：收口当前 strongest baseline

顺序建议：

1. `ce = 0.4`
2. 多 seed
3. 固定 strongest baseline

这一阶段的目标是：

- 把“当前最强版本”正式钉住

### 第二阶段：升级训练目标

顺序建议：

1. `CE` 调度
2. `BPR / pairwise ranking loss`
3. 行为约束更强的 offline RL 算法

这一阶段的目标是：

- 从“有效的联合损失”进化到“更合理的联合训练范式”

### 第三阶段：结构与数据创新

顺序建议：

1. dual-head
2. state pooling 对照
3. buffer / candidate 机制升级

这一阶段的目标是：

- 提炼真正有研究与项目亮点的方法创新

## 五、哪些最值得写进后续项目亮点

如果从简历、面试、项目报告的表达角度看，后续最值得形成亮点的有三条。

### 1. 从离线 DQN 崩坏到稳定主线的系统修复

这部分亮点在于：

- 你不是只会跑模型
- 而是真正理解了离线 RL 里 value explosion、选模失真、排序目标丢失这些问题

### 2. RL + ranking supervised regularization 的联合训练范式

这是当前最有项目辨识度的一条线，因为它已经被实验结果证明有效，而且很贴近推荐系统实际。

### 3. 从 baseline 调优走向 behavior-constrained offline RL

这会是后续最值得升级成“方法创新”的方向，尤其适合和：

- 推荐系统
- 离线 RL
- 量化研究里的序列决策问题

做迁移类比。

## 当前路线一句话总结

项目接下来不该再漫无目的地继续扫参，也不该过早跳到完全新框架。

最合理的路线是：

- 先把当前 `RL + CE` strongest baseline 收口
- 再从训练目标和 offline RL 约束机制切入
- 最后再做结构与数据层面的真正创新

也就是说，项目已经正式进入：

- 从“强 baseline 工程阶段”过渡到“可讲方法创新的研究型工程阶段”

的门槛位置。
