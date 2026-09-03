# 2026-09-03 为什么 RL 在 LLM 上像能力放大器，在离线推荐上却更别扭

## 实验假设与核心矛盾

同样是“在强预训练模型之上叠加 RL”，LLM 往往能通过 RL 显著提升可验证推理能力，而当前 MovieLens-1M 离线序列推荐项目中，RL 的主要增益只体现在固定表征上的决策头学习，尚未证明对 `SASRec attention encoder` 的动态微调有效。

## 数据与防穿越边界

### 当前离线推荐项目的边界

- 数据划分采用按时间排序的用户行为序列切分。
- 状态定义为最近 `50` 个交互 item 序列。
- 标签定义为下一个真实点击 item。
- 边界条件为 `t_feature < t_label`。
- 训练、验证、测试都建立在 logged next-click action 上，而不是在线交互上。

当前离线 buffer 规模：

- `kept_user_count = 6040`
- `kept_item_count = 3706`
- `train_transition_count = 982089`
- `valid_transition_count = 6040`
- `test_transition_count = 6040`
- `all_transition_count = 994169`

当前即时奖励定义：

- `r(s_t, a_t) = 1`，当 `a_t` 等于日志中的真实下一个点击
- `r(s_t, a_t) = 0`，在评估意义上表示未命中

标签分布：

- 显式正样本数：`982089`（train）
- 显式负样本数：`0`（buffer 中未显式存储）
- 隐式负动作空间：每个状态最多对应 `3705` 个非真实动作
- `imbalance_ratio = N_negative / N_positive = [待补充: 若按训练时实际采样负样本口径统计]`

### 与 LLM-RL 的本质差异

当前推荐项目的数据是：

- 静态日志
- 单一路径观察
- 未覆盖全动作空间

而 LLM-RL 常见训练数据更接近：

- 当前策略自己生成的候选输出
- 可被规则、测试或答案校验器直接判定
- 更新分布与当前策略更接近

因此，二者虽然都叫 RL，但数据供给机制完全不同。

## 特征构建与数学表达

### 当前项目中 RL 目标的统计结构

当前推荐项目近似在学习：

```text
Q(s_t, a_t) ≈ E[r_t + γ max_a' Q(s_{t+1}, a') | s_t, a_t]
```

其中：

```text
s_t = [i_{t-49}, ..., i_t]
a_t = i_{t+1}^{logged}
r_t ∈ {0, 1}
γ = 0.9
```

但训练可见的动作只覆盖：

```text
a_t = a_t^{logged}
```

而目标网络在更新时需要对全动作空间做：

```text
max_a' Q(s_{t+1}, a')
```

这就引入了典型的离线 RL 结构性风险：

```text
OOD(a') = 1[a' not sufficiently supported by logged data]
```

如果 `Q(s, a')` 对 OOD 动作高估，则会通过：

```text
target_q = r + γ max_a' Q_target(s', a')
```

向上游回传虚假价值。

### 为什么 LLM-RL 的数学结构更友好

LLM 推理类 RL 常见目标虽然形式上仍是策略优化，但其奖励函数更接近：

```text
R(y | x) = verifier(x, y)
```

其中 `verifier` 可能是：

- 数学答案是否正确
- 代码是否通过测试
- 输出是否满足格式约束

这意味着：

```text
R(y | x)
```

比推荐系统里的点击信号更低噪声、更接近真实目标。

同时，LLM 更像在 token 轨迹空间做受约束搜索：

```text
π(y | x) -> sample -> verify -> reinforce
```

而当前推荐项目更像在静态日志上做受限反事实估计：

```text
logged(s, a, r, s') -> bootstrap -> conservative regularization
```

两者难度不在同一量级。

## 离线评估与指标对比

### 当前项目中的直接证据

当前 strongest baseline：

| 运行名 | best_valid_ndcg@10 | test_hr@10 | test_ndcg@10 | all_hr@10 | all_ndcg@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ce_reg05_encoder_frozen` | `0.126052` | `0.209934` | `0.116636` | `0.273703` | `0.148182` |

对照结果表明：

| 运行名 | best_epoch | best_valid_ndcg@10 | test_hr@10 | test_ndcg@10 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `ce_reg05_encoder_frozen` | `5` | `0.126052` | `0.209934` | `0.116636` | 当前最强 |
| `bpr_ns1_encoder_frozen` | `3` | `0.118025` | `0.198344` | `0.108698` | 低于 CE |
| `bpr_ns3_encoder_frozen` | `3` | `0.117574` | `0.197517` | `0.108841` | 低于 CE |

更关键的是，`warmup5` 与 `encoder_frozen` 得到相同最优结果，说明当前增益主要来自：

- 固定 `SASRec encoder`
- 训练 `Q-head`

而不是来自 RL 对 attention block 的继续微调。

### 结构性解释

当前证据支持以下判断：

1. `SASRec attention encoder` 作为表征器是有效的。
2. `RL + CE` 能在固定表征上学到更强的决策头。
3. 当前没有证据证明 RL 动态微调 encoder 会继续抬升离线排序指标。

### 与 LLM-RL 的对照结论

LLM-RL 能显著起效，通常同时满足：

- 奖励可验证
- 任务目标与奖励高度一致
- 采样分布贴近当前策略
- 基础模型已具备强先验能力
- 常配有 KL / reference model / verifier 等稳定约束

而当前离线推荐项目同时面临：

- 奖励稀疏且只观测 logged positive
- 大动作空间：`3706` 个动作
- OOD 动作高估风险
- `TD/CQL` 数值目标与 `HR@10/NDCG@10` 排序目标不完全对齐
- 静态日志无法提供在线纠错

所以在当前项目里，RL 更像：

- 在固定表征上训练价值头

而不是像 LLM-RL 那样：

- 全面放大 backbone 的推理能力

## 性能瓶颈与物理开销

### 当前项目的主要结构瓶颈

1. 动作空间大但日志支持稀薄。
2. 即时奖励只对 logged action 明确，反事实动作无真实回报。
3. `max_a Q(s', a)` 必然涉及未充分观测动作。
4. 排序质量最终由 `HR@10 / NDCG@10` 评估，但训练主目标包含 bootstrapped value estimation。

### 当前项目已观察到的动力学症状

- 早期版本频繁出现 `best_epoch = 1`
- 解冻 encoder 后 `valid_ndcg@10` 回落
- 纯 `BPR` 线早于 CE 见顶

这些症状与“RL 可以直接改好 attention encoder”这一假设不一致。

### 物理开销

- 训练样本规模：`982089` transitions
- 全量回测规模：`994169` transitions
- 序列长度：`50`
- 候选动作数：`3706`
- 输出工件：checkpoint、training metrics、test/all evaluation summaries
- 墙钟时间：`[待补充: 各实验 wall-clock time]`
- 显存峰值：`[待补充: GPU memory peak]`
- 峰值内存：`[待补充: host memory peak]`
- OOM 风险：当前未见直接证据，但全量动作评分与多次全量回测具有持续算力压力

## 正式结论

当前最稳健的判断不是“RL 对推荐无用”，而是：

- **RL 在 LLM 上像革命，主要因为 reward 更真、策略采样更贴近训练分布、目标与评估更一致。**
- **RL 在当前离线推荐项目里表现别扭，主要因为它必须在静态日志、稀疏奖励、大动作空间和反事实不可观测的条件下进行 bootstrap 学习。**
- **因此，当前项目中 RL 的有效角色是“固定 SASRec 表征上的价值头学习器”，而不是“自动把 attention encoder 继续训强的万能增益器”。**

## 下一步问题

- `[待补充: 若需要进一步量化“LLM-RL 与离线推荐 RL 的差异”，应补充一个 CE-only、RL+CE、RL+CE+BPR 的统一对照表]`
- `[待补充: 若需要进一步验证 encoder 是否完全不值得微调，应补充参数高效微调方案，如 LoRA/last-block-only，而非全量解冻]`
