# Criteo CTR Baseline DeepFM 架构笔记

## 这个项目在学什么

这个项目不是单纯把 one-hot LR 换成一个更复杂的模型名字。

它要解决的是更本质的问题：

- 当类别特征很多、类别取值很多时，为什么不能一直靠 one-hot 展开
- 为什么 CTR 任务里经常会用 embedding
- linear、FM、deep 三部分分别在学什么
- 一个样本到底是怎么进入 DeepFM 的

## DeepFM 和 one-hot LR 的区别

在 one-hot LR 里：

- 每个类别值会被展开成稀疏的 0/1 位
- 线性模型主要学习每个类别位各自的加性贡献

在 DeepFM 里：

- 每个类别值先被映射成整数 id
- 每个 id 再去 embedding table 里查一个低维向量
- 这些向量同时被 FM 部分和 deep 部分使用

所以 one-hot 是“把类别拆成很多维”，embedding 是“把类别压成一个可学习的低维表示”。

## embedding 到底是什么

可以先把 embedding 理解成：

- 一个可学习的查表层

例如某个 sparse 特征叫 `categorical_feature_12`：

- 类别 `abc123` 先被编码成 id `17`
- 进入 embedding table 后，id `17` 会查出一个向量，例如 `[0.12, -0.31, 0.08, ...]`

这个向量不是人工写死的，而是在训练中不断更新出来的。

所以 embedding 的核心不只是“压缩维度”，而是：

- 给每个离散类别学习一个连续语义表示

这比 one-hot 的优势在于：

- 不再把每个类别当作完全孤立的位
- 模型可以通过向量空间表达相似性和交互

## 一个样本是怎么进入 DeepFM 的

以一条样本为例，它原始上仍然是一行表格：

- 13 个 dense 特征
- 26 个 sparse 特征
- 1 个 label

预处理之后：

- dense 特征保留为数值
- sparse 特征变成整数 id

进入模型时会分成两部分：

### 1. dense 输入

dense 特征直接作为连续值输入：

- `dense_x.shape = [batch_size, dense_feature_count]`

这些值既可以送入 linear 部分，也可以送入 deep 部分。

### 2. sparse 输入

sparse 特征会形成一个 id 矩阵：

- `sparse_x.shape = [batch_size, sparse_feature_count]`

每一列代表一个 field，每个位置是该样本在这个 field 上的类别 id。

然后每个 field 都有自己的 embedding table：

- `field_1 id -> embedding`
- `field_2 id -> embedding`
- ...

于是每个样本就会得到一组 embedding 向量。

## DeepFM 的三部分

### 1. linear 部分

这一部分和 LR 很像，主要负责学习一阶贡献：

- 某个 dense 值变大，是不是整体更容易点击
- 某个 sparse id 单独出现，是不是整体更容易点击

可以理解成：

- 单个特征自己对点击概率的直接影响

### 2. FM 部分

FM 的核心目标是学习二阶交互，例如：

- 某个广告 id 和某个上下文 id 放在一起时，会不会更容易点击
- 某个页面位置和某个设备类型组合起来时，会不会有特殊效果

FM 不是手工枚举交互，而是通过 embedding 向量的关系自动表达这些交互。

### 3. deep 部分

deep 部分把：

- dense 特征
- 所有 sparse embedding 拼接后的结果

一起送入 MLP。

它负责学习：

- 更高阶、更复杂、非线性的组合模式

可以简单理解成：

- linear 看一阶
- FM 看显式二阶
- deep 看更复杂的高阶模式

## 为什么 DeepFM 比 one-hot LR 更适合 CTR 主线

CTR 场景通常有这些特点：

- 高基数类别特征
- 稀疏离散 id
- 强交互结构
- 大量上下文影响

DeepFM 的优势正好对着这些特点：

- embedding 适合处理高基数离散特征
- FM 适合处理二阶交互
- deep 适合补充更复杂的非线性结构

所以它比单纯 one-hot LR 更接近推荐排序与广告点击的主干路线。

## 当前这个 DeepFM 项目的教学目标

当前版本不是为了立刻做到很强，而是为了学懂：

- DeepFM 的输入长什么样
- embedding 是怎么被查出来的
- FM 为什么能表达交互
- 一个 batch 到底是怎么完成 forward 的

也就是说，这个项目首先是“架构理解沙盒”，然后才是模型实验项目。

## 当前已加入的重要修正：dense 标准化

在第一次 DeepFM 训练排查后，当前项目已经加入：

- 基于训练集统计量的 dense 标准化

具体做法是：

- 先在训练集上计算每个 dense 特征的 `mean` 和 `std`
- 再对 train / valid 使用同一组统计量做变换

变换形式为：

\[
x' = \frac{x - \mu}{\sigma}
\]

这里要强调两点：

- 统计量只能从训练集拟合，不能把验证集信息混进去
- 这一步不是“美化数据”，而是防止 DeepFM 在输入层就进入数值失控状态

加入这一步之后，logits 的量级已经从“几千到几万”下降到更可控的范围，这说明 dense 特征原始尺度确实是导致训练不稳定的重要来源之一。

## 从 linear 分支看：大尺度 dense 如何破坏优化

DeepFM 的 linear 部分里，dense 特征会直接进入线性层：

\[
z_{linear} = w^\top x + b
\]

如果某些 dense 特征原始数值非常大，那么即使参数 `w` 还不大，乘积 `w_i x_i` 也可能立刻变得很大。

这会带来三个问题：

1. 输出 logit 会被少数大尺度特征主导  
   模型还没来得及学到“哪些特征真正重要”，就先被数值尺度最大的特征强行控制住输出。

2. 梯度更新不均衡  
   因为大尺度特征对应的梯度也会更大，优化器会优先围着这些特征转，其他尺度较小但语义重要的特征反而更难学到。

3. 更容易把整体 logit 顶到极端区间  
   一旦 logit 过大或过小，模型就会快速走向“极端确信”，这对不平衡 CTR 数据尤其危险，容易演化成“几乎全压负类”的策略。

所以从 linear 角度看，dense 标准化的作用不是锦上添花，而是在保护：

- 不同 dense 特征的贡献处于可比较的尺度
- 梯度不会被少数大数值特征垄断
- logit 不会在训练初期就被直接顶飞

## 从 deep 分支看：大尺度 dense 如何破坏优化

在当前 DeepFM 架构里，deep 分支的输入是：

- dense 特征
- 所有 sparse embedding 拼接后的向量

也就是说，dense 值会和 embedding 一起进入 MLP。

如果 dense 特征尺度远大于 embedding 的数值尺度，就会出现明显失衡：

- embedding 一般是小尺度初始化的连续向量
- dense 原始值可能是几十、几百，甚至更大

这样一来，deep 分支会更容易先围绕 dense 特征建立激活模式，而不是认真利用 sparse embedding 所表达的类别结构。

这会带来几个具体破坏：

1. dense 输入压制 embedding 输入  
   MLP 的前几层更容易被大尺度 dense 触发，从而削弱 embedding 在表示交互和语义结构上的作用。

2. 激活分布容易失衡  
   即使当前 hidden layer 使用的是 ReLU，大尺度输入仍然会让某些神经元长期处于过强激活状态，导致网络更容易形成偏置化表示。

3. deep 输出也会参与最终 logit 相加  
   当前 DeepFM 最终输出是：
   - linear logit
   - FM logit
   - deep logit

   如果 deep 分支因为输入尺度失衡而输出过大，那么它会和 linear、FM 一起把总 logit 推向极端。

所以从 deep 角度看，dense 标准化的价值在于：

- 让 dense 与 embedding 至少处于更接近的数量级
- 避免 MLP 一开始就被 dense 大数值劫持
- 让 deep 分支更有机会学习“结构”和“组合”，而不是只对原始数值大小做粗暴放大

## 当前阶段的理解结论

dense 标准化不是单纯的数据清洗习惯，而是当前 DeepFM 稳定训练的必要条件之一。

它至少同时保护了两件事：

- linear 分支的数值稳定性
- deep 分支对 dense 与 embedding 的平衡利用

所以在当前项目里，应当把“dense 标准化”视为 DeepFM 输入管线的核心组成部分，而不是可有可无的附加步骤。

## 训练异常排查：为什么要关注 logits、BCE 和单 batch 过拟合

在第一次训练 DeepFM 时，如果出现下面这类现象，就不能急着调 epoch 或学习率：

- BCE loss 很大
- log_loss 和 BCE loss 看起来差距明显
- AUC 接近随机，但 loss 又非常夸张

这说明需要先检查底层数值路径，而不是继续堆外层超参数。

### 1. 先确认评估指标是不是喂对了数据

在本项目中：

- `BCEWithLogitsLoss` 接收的是原始 logits
- `roc_auc_score` / `log_loss` 接收的应该是经过 sigmoid 后的概率

所以正确路径应该是：

- forward 输出 logits
- 训练损失直接用 logits 进入 `BCEWithLogitsLoss`
- 验证指标先对 logits 做 `sigmoid`
- 再把概率传给 sklearn 指标

如果跳过 sigmoid，`log_loss` 和 `roc_auc` 都会失去解释意义。

当前项目代码已经按这个路径实现。

### 2. 为什么要打印 logits 的 min / max / mean

DeepFM 的最终输出是：

- linear logit
- FM logit
- deep logit

三部分直接相加。

如果其中某一部分数值过大，就会把最终 logit 顶飞。

所以排查时要看：

- `logits.min()`
- `logits.max()`
- `logits.mean()`

这能帮助判断：

- 输出是否出现数值爆炸
- 模型是否一开始就极度偏向正类或负类
- embedding 与 FM 项是否把分数推得过高

本项目第一次排查时已经观测到非常明显的异常：

- 验证阶段 logits 出现了几千到几万量级
- 最小值达到 `-90000` 左右，最大值也达到几百到几千
- 均值长期大幅偏负

这说明问题不是“指标接口忘了 sigmoid”，而是模型本体输出已经数值失控。

### 3. 为什么要做 1-batch test

单 batch 过拟合测试是深度学习里很重要的基础体检。

做法是：

- 只取一个 batch
- 在这一个 batch 上反复训练很多步

目的不是看泛化，而是看：

- 模型和损失函数的连接是否正常
- 反向传播是否有效
- 模型是否至少有能力记住一个很小的数据块

如果一个模型连单 batch 都很难压低 loss，优先怀疑：

- 数值路径问题
- 输入张量构造问题
- 标签类型或维度问题
- 模型输出尺度异常

本项目的 1-batch test 说明了两点：

- loss 的确能下降，说明梯度链路不是完全断的
- 但 logits 会迅速走向极端，说明模型是在“用极端打分硬挤 loss”，而不是健康地学习

所以当前结论不是“模型完全学不动”，而是“模型能学，但数值尺度非常不稳定”。

### 4. 当前已经加入的排查工具

在 `train_deepfm.py` 中已经加入：

- `--debug-logits`
  - 在评估阶段打印 logits 的最小值、最大值、均值

- `--one-batch-overfit`
  - 启动单 batch 过拟合测试

- `--one-batch-steps`
  - 控制单 batch 重复训练的步数

建议排查顺序：

1. 先正常训练时打开 `--debug-logits`
2. 观察 logit 是否明显爆炸
3. 再运行 `--one-batch-overfit`
4. 看 loss 能不能稳定下降，以及 logits 是否越来越极端

### 5. 当前已经定位出的首要问题

根据当前排查结果，最先需要修的不是 epoch 数，也不是外层学习率，而是输入与输出的数值尺度。

首要判断：

- 指标路径没有喂错，sigmoid 已经在评估阶段使用
- 问题核心是 logits 数值爆炸

当前最优先的工程修正是：

- 对 dense 特征按训练集统计量做标准化

原因：

- dense 特征原始尺度可能很不一致
- linear 分支会直接吃 dense 值
- deep 分支也会拼接 dense 值
- 如果 dense 数值过大，会同时把 linear 和 deep 两部分的输出顶飞

所以当前项目已经把下面这一步加入预处理：

- 先在训练集上拟合 dense 特征的 mean / std
- 再对 train / valid 使用同一组统计量做标准化

这一步的意义不是“让分数立刻变好”，而是先让 DeepFM 的数值行为回到更可控的范围。

### 6. 这类问题的学习价值

这类异常很重要，因为它会逼着我们真正理解：

- logits 和概率不是一回事
- DeepFM 的三部分输出是如何叠加的
- embedding / FM 的数值规模为什么会影响整体训练稳定性
- 深度模型不是“能跑就算对”，而是必须先过数值体检

所以这些问题和解决方法，应当被视为项目中的重点经验，而不是临时排障。
