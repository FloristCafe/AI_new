# 2026-07-29 软更新方案失效分析与 Huber / Gamma / Binary Reward 修正

## 本次记录的目的

本条日志记录两件事：

1. 对上一轮 `soft update + unfreeze encoder + adaptive CQL alpha` 实验做一次正式复盘。
2. 记录当前已经落地到代码中的下一轮修正方案：
   - `TD loss` 从 `MSE` 改为 `Huber Loss`
   - `gamma` 从长视野大折扣改为更短视野
   - reward 从稠密奖励改为二值命中奖励

对应的失败实验目录为：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\artifacts\experiments\cql_softupdate_unfreeze_adaptive_main`

## 上一轮实验改了什么

上一轮代码层面的主要改动包括：

- 废除硬更新，改为 `Polyak soft target update`
- 解除 `encoder` 冻结，允许极低学习率微调
- 引入 `adaptive CQL alpha`
- 训练日志中增加：
  - `train_raw_cql_to_td_ratio`
  - `train_effective_cql_to_td_ratio`
  - `valid_raw_cql_to_td_ratio`
  - `valid_effective_cql_to_td_ratio`
  - `cql_alpha_initial`
  - `cql_alpha_final`

这一轮的原始目标是正确的：

- 不再让 target network 每隔固定 step 暴力跳变
- 不再让 Q-head 独自扭曲冻结表征
- 不再盲目把 `alpha` 写死在一个绝对常数上

但是实验结果说明，这些动作并没有触及最核心的动力学矛盾。

## 上一轮实验结果的关键现象

从 `cql_softupdate_unfreeze_adaptive_main` 的训练摘要可以看到：

- `best_epoch = 1`
- `best_selection_metric_value = 37.70275241776018`
- `cql_alpha_final = 5.0`
- `train_mean_q_value` 从 `0.46` 上升到 `86.59`
- `train_max_q_value` 从 `6.29` 上升到 `121.88`
- `valid_max_q_value` 上升到 `129.62`
- `valid_effective_cql_to_td_ratio` 从 `0.415` 掉到 `0.0038`

这几个数字说明了一件很本质的事：

- 系统确实试图用更大的保守惩罚去压制 Q 值上升
- 但越到后期，CQL 对训练方向的真实控制力越弱
- 后面的训练已经几乎完全被 TD 目标主导

所以问题不是“有没有上 CQL”，而是：

- 在当前训练目标下，CQL 的梯度强度是否从数学上就注定压不住 TD

答案是：基本注定压不住。

## 数学尸检：为什么上一轮 CQL 失去了控制力

上一轮失败的根本原因，不只是参数没调好，而是损失项的梯度结构天生不对称。

### 1. MSE 型 TD Loss 的梯度是线性无界的

之前 TD 项使用的是：

- `F.mse_loss(current_q, target_q)`

它对 `Q` 的梯度形式可以理解为：

- `2 * (Q - target)`

这意味着：

- 当 `Q` 和 `target` 的误差变大时
- TD 项的梯度会随误差线性增大
- 误差如果膨胀到几十、上百，梯度也会跟着变成几十、上百

TD 梯度没有上界。

### 2. CQL 的 `logsumexp` 梯度本质上是 softmax 概率

CQL penalty 的主项是：

- `logsumexp(q_values) - current_q`

其中 `logsumexp` 对每个动作 logit 的梯度，本质上对应一个 softmax 权重。

这意味着：

- 单个动作在 `logsumexp` 里的梯度天然被限制在 `[0, 1]`
- 即便 `Q` 整体膨胀，单项梯度也不会像 MSE 那样无界变大

所以哪怕 `alpha` 增大，CQL 的“单步拉力”也有很强的天然上限。

### 3. 为什么把 `alpha` 顶到 5.0 仍然不够

这轮实验里：

- `cql_alpha_final = 5.0`

但这并不意味着保守项就一定足够强。

真正的问题在于：

- `MSE` 型 TD 梯度可以随着 `Q-target` 误差无限放大
- `CQL` 的梯度放大能力远没有这么快

于是就会发生一种很典型的训练后期失衡：

- TD 像洪水一样推着 `Q` 往上冲
- CQL 像一根有限张力的绳子，在后面越拉越无力

这正是为什么我们看到了：

- `valid_effective_cql_to_td_ratio` 从 `0.415` 掉到 `0.0038`

也就是说，到训练后期，保守项虽然还在数值上存在，但它对总梯度方向几乎已经没有实际话语权了。

### 4. 为什么 soft update 也没有从根上解决问题

`Polyak soft update` 解决的是：

- target network 变化过于剧烈的问题

但它没有改变下面这个核心事实：

- online Q 仍然在用 `R + gamma * max Q(next_state)` 自我放大

因此 soft update 做到的只是：

- 让错误目标传播得更平滑

而不是：

- 让错误目标本身变正确

所以它降低了跳变，却没有切断放大的链条。

## 物理坍塌：为什么 reward 结构和 horizon 也在制造幻觉

除了损失函数的梯度结构失衡，上一轮还有两个环境层面的错位。

### 1. 稠密奖励给了离线策略可钻的空子

原先的 reward 设计是：

- exact hit: `+1.0`
- genre match: `+0.1`
- mismatch: `-0.1`

这个设计在概念上看似更“聪明”，但在离线序列 RL 的冷启动阶段，实际上可能是在给策略制造幻觉。

因为模型可能学到一种伪规律：

- “只要我持续往某个类型邻域推，我就可能稳定拿到小正奖励”

如果这类局部模式再被 `max Q(next_state, a')` 自我放大，就会演化成一种错误的长期价值信念。

也就是说，模型不一定真的学会了：

- 精确预测用户下一步会点什么

它更可能只是学会了：

- 在 reward shaping 留下的空隙里刷一个看起来还行的价值估计

### 2. `gamma = 0.99` 对这个问题来说视野过长

在推荐序列中，用户意图非常脆弱。

和 Atari 这类环境不同，推荐里的“未来 100 步价值”往往并不稳定，也未必有物理意义。

当 `gamma = 0.99` 时，系统默认自己要为一个极长 horizon 负责，这会带来两个问题：

1. 未来价值项 `gamma * max Q(next_state)` 被过度放大。
2. 任何小的正向幻觉，都更容易在自举链条里被反复累积。

如果再叠加上一轮的 reward shaping，那么即便是很小的错误正反馈，也可能被无限向后传播。

所以这不是单纯的“参数不理想”，而是：

- reward 形状和 horizon 长度共同放大了错误价值信号

## 为什么上一轮“不是无效”，而是“不够有效”

这里需要区分两个层面。

### 有效的部分

上一轮并不是完全无意义。它至少证明了：

- target 硬更新确实太粗暴
- 训练日志必须显式跟踪 `TD` 和 `CQL` 的相对力量
- 只看 `loss` 数值大小，不足以理解离线 RL 的真实训练动力学

也就是说，它帮助我们更清楚地看见了病灶。

### 不够有效的部分

但它没有解决更深层的问题：

- `MSE` 的梯度放大量级过强
- `gamma = 0.99` 让 bootstrapping 视野过长
- 稠密 reward 给了价值幻觉可乘之机

因此它只能算：

- 在旧目标函数上做稳定化修补

而不是：

- 对训练目标本身做结构性纠偏

## 当前已经落地到代码的修正

在 2026-07-29 这轮修正中，代码已经完成了三项关键改动。

### 1. TD Loss 从 MSE 改为 Huber Loss

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\train_sasrec_dqn.py`

改动：

- `F.mse_loss` 替换为 `F.smooth_l1_loss`

这样做的意义是：

- 误差小时保留二次惩罚，便于精细拟合
- 误差很大时退化为一次惩罚，防止 TD 梯度无限膨胀

这不是在追求更小的表面 loss，而是在给 TD 梯度加物理刹车。

### 2. 默认 `gamma` 从 0.99 下调到 0.7

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\train_sasrec_dqn.py`

改动：

- `--gamma` 默认值从 `0.99` 调整为 `0.7`

这表示当前项目明确进入“短视离线控制”阶段：

- 先拟合近未来
- 先压住 bootstrapping 放大链
- 再讨论更长 horizon 的价值建模

### 3. reward 改成严格二值命中奖励

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\ml_1m_genre_utils.py`
- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\build_offline_buffer.py`
- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl\src\evaluate_sasrec_dqn.py`

改动：

- exact hit: `1.0`
- non-hit: `0.0`
- 不再使用 `genre match = +0.1`
- 不再使用 `mismatch = -0.1`

这意味着当前阶段我们主动放弃“聪明的 shaped reward”，转而使用：

- 更冷酷
- 更窄
- 但更不容易产生幻觉

的目标定义。

## 当前这轮修正的真正目标

这一轮不是为了立刻刷新最终指标，而是为了先验证三件事：

1. `Q` 值抬升速度能否明显下降。
2. `valid_effective_cql_to_td_ratio` 能否不再迅速塌缩到接近 0。
3. `best_epoch` 是否仍然死在第 1 轮。

如果这三件事改善了，即便 `HR@10` 还没有立刻大幅上升，这轮修正也仍然是成功的。

因为它说明：

- 我们终于开始在改“训练动力学本身”

而不是只在旧病灶上继续打补丁。

## 当前阶段总结

截至 2026-07-29，这个项目对上一轮失败的理解已经比之前更深入了一层：

- 问题不只是“Q 值变大”
- 也不只是“CQL 权重不够”
- 而是 `MSE TD`、长 horizon 和稠密 reward 一起构成了一个会自我放大的错误系统

因此，今天这轮修正的核心意义在于：

- 用 `Huber Loss` 给 TD 梯度加上限
- 用更小的 `gamma` 砍短未来价值链条
- 用二值 reward 清除虚假的局部正反馈

这三步合起来，才是真正意义上的“从训练物理层面修复离线 RL”。
