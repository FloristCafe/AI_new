# MovieLens 1M SASRec Baseline 当前状态笔记

## 1. 笔记定位

这份笔记用于记录 `movielens_1m_sasrec_baseline` 的当前工程状态、方法定位、首轮实验结果，以及下一步最应该推进的升级方向。

当前存放位置为：

- `docs/project_notes/`：项目级判断、实验记录、策略复盘
- `reinforcement_learning/`：当前按长期序列决策主线归档
- `movielens_1m_sasrec_baseline/`：当前具体项目

这类内容不放进 `src/` 或 `artifacts/`，因为这里记录的不是“代码是什么”，而是“为什么这样做、目前做到哪一步、接下来怎么推进”。

## 2. 项目当前定位

这个项目虽然被放在 `projects/reinforcement_learning/`，但当前方法本体并不是 RL agent，而是一个监督式的序列推荐 baseline。

当前要解决的问题是：

- 输入用户按时间排序的历史交互序列
- 预测用户的下一个交互 item
- 用标准序列推荐协议评估预测质量

也就是说，当前阶段的核心目标不是长期回报优化，而是先把 `Sequential RecSys` 的基础工程链条和强 baseline 建起来：

- 数据按时间排序
- leave-one-out 切分
- 固定长度序列样本构造
- SASRec 训练与评估

这个定位是合理的。因为如果一开始就跳到 DQN、Slate RL 或 offline RL，很多更基础的问题会被掩盖：

- 序列样本到底怎么构造
- 评估协议是否规范
- attention 序列建模是否真的有效
- 当前数据集上 next-item prediction 的自然上限大致在哪里

所以当前这版 `SASRec baseline` 应该被看作：

- 一个真实序列推荐项目的工程起点
- 一个后续 RL/decision-aware 推荐路线的参考底座
- 一个能帮助判断“数据问题、模型问题、训练问题”分别在哪的基线系统

## 3. 当前代码链条

当前项目已经具备一条完整的最小可运行链路：

- `src/preprocess_movielens_1m.py`
  - 读取 `interactions.csv`
  - 按用户和时间排序
  - 过滤低频用户与物品
  - 重映射 `user_id` 与 `item_id`
  - 生成训练、验证、测试的固定长度序列样本

- `src/sasrec_model.py`
  - `item embedding`
  - `learnable positional embedding`
  - `causal self-attention`
  - `padding mask`
  - 最后有效位置表示提取
  - 全量 item 打分

- `src/train_sasrec.py`
  - 读取 `train_sequences.npz`
  - 训练 next-item prediction 模型
  - 保存 `sasrec_best.pt` 与 `sasrec_final.pt`
  - 保存 `training_metrics.json`

- `src/evaluate_sasrec.py`
  - 加载 checkpoint
  - 对全量 item 打分
  - 计算 `HR@10` 与 `NDCG@10`
  - 保存测试指标

这说明项目已经不再是骨架，而是进入了“第一版 baseline 能跑出真实结果”的阶段。

## 4. 当前模型到底是什么

当前模型是一个简化可运行版 `SASRec`：

- 输入：长度截断后的用户历史 item 序列
- 表示层：`item embedding + position embedding`
- 编码器：单向 `TransformerEncoder`
- 约束：`causal mask` 防止偷看未来
- 读出方式：取最后一个有效历史位置的 hidden state
- 打分方式：与全量 item embedding 做点积
- 训练目标：多分类交叉熵，直接学习
  - `p(next_item | history_sequence)`

它不是传统马尔可夫链，也不是 RL 的 `Q(s, a)` 学习器。

更准确地说，它是一个自回归的条件概率模型：

- 给定历史序列
- 预测下一个物品

所以它更接近：

- `GRU4Rec` 的 Transformer 升级版
- `BERT4Rec` 的单向 next-item 版本

而不是：

- policy gradient
- DQN recommendation
- slate optimization

## 5. 当前实验结果

在 `MovieLens-1M` 上，当前首轮跑通结果为：

- 训练样本数：`982,089`
- 验证用户数：`6,040`
- 测试用户数：`6,040`

训练 10 个 epoch 后：

- `loss`: `8.239766 -> 6.019184`
- `train accuracy`: `0.002831 -> 0.019181`

测试集评估结果：

- `HR@10 = 0.154470`
- `NDCG@10 = 0.084762`

这些结果说明：

1. 数据预处理、张量化、训练和评估主链路都是通的。
2. loss 稳定下降，说明模型确实学到了序列模式。
3. 训练 accuracy 很低是正常现象，因为这里是几千个 item 的 next-item 多分类，不是 CTR 二分类。
4. 真正应该关注的指标是 `HR@10` 和 `NDCG@10`，而不是训练 accuracy。

对于第一版手写 SASRec baseline，这组结果是正常、可用、足以作为后续升级基线的。

## 6. 当前版本的价值与局限

### 6.1 当前版本的价值

当前这版最重要的价值不是“成绩已经很强”，而是它已经把最关键的基础问题钉住了：

- Sequential RecSys 不是按行表格学习，而是按序列建模
- 时间顺序不能打乱
- leave-one-out 是比随机切分更自然的评估协议
- attention 模型可以在这个数据集上形成有效 baseline

它已经为后续问题提供了稳定起点：

- 超参怎么调
- 更原味的 SASRec 怎么对齐
- negative sampling 是否更合适
- 和 GRU4Rec / BERT4Rec 的差距在哪里
- 将来如何过渡到 RL-aware recommendation

### 6.2 当前版本的局限

当前实现仍然是 baseline，不是最终实验版，主要局限包括：

- 训练时只看训练损失，没有把验证集纳入 best model 选择
- 还没有 `early stopping`
- 当前训练目标是 full softmax cross-entropy，不是经典 SASRec 常见的负采样目标
- 评估虽然是全量排序，但还没有 sampled negative 版本用于论文协议对齐
- 还没有多 seed 统计
- 还没有系统超参实验

也就是说，当前代码已经完成了“跑通”，但还没有完成“实验化”。

## 7. 下一步最该推进什么

当前最优先的升级，不是换模型，而是把训练闭环补完整。

最应该先做的三件事：

1. 在 `train_sasrec.py` 中加入每个 epoch 的 `valid` 集评估。
2. 用验证集 `HR@10` 或 `NDCG@10` 选择 best checkpoint。
3. 加入 `early stopping`，避免只按训练 loss 继续拟合。

原因很简单：

- 序列推荐真正关心的是 Top-K 命中效果，不是训练集 loss
- 没有验证集闭环，就很难判断某次改动是真的有效，还是只是在训练集上拟合得更好
- 这一步完成后，项目才真正具备“系统做实验”的资格

在这之后，再推进第二层升级：

- sampled negative evaluation
- negative sampling loss
- 多 seed 复现实验
- `max_seq_len / embedding_dim / num_blocks / dropout` 消融

## 8. 当前阶段结论

当前 `movielens_1m_sasrec_baseline` 已经完成了第一阶段最关键的任务：

- 从原始 `MovieLens-1M` 行为日志出发
- 建立了一个可运行、可解释、可复现的序列推荐 baseline
- 拿到了第一组真实 `HR@10 / NDCG@10` 结果

它现在最合理的角色不是“立刻封存”，也不是“立刻大幅换模型”，而是：

- 先把验证闭环、best checkpoint、early stopping 补齐
- 让它从“能跑的 baseline”升级为“能做实验的 baseline”

一句话概括当前状态：

**这个项目已经成功跨过了从“离散交互表”到“真实 Sequential RecSys baseline”的第一道门槛，下一步应优先把训练与验证协议做标准化，而不是过早跳向更复杂的模型或 RL。**
