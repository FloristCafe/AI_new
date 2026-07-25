# 2026-07-25 BCE Negative Sampling 对齐实现

## 本次推进的目标

在不破坏现有 `cross_entropy` baseline 的前提下，把项目往论文式 SASRec 再推进一步：

- 保留原有 `全量 item softmax + CrossEntropy` 训练路径
- 新增 `BCE + negative sampling` 训练路径
- 评估协议保持不变，继续使用 `HR@10` 和 `NDCG@10`

这样可以在同一套数据切分、同一套评估标准下，直接比较两种训练目标。

## 本次代码改动

### 1. 预处理新增序列级监督数据

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\preprocess_movielens_1m.py`

新增产物：

- `train_sequence_supervision.npz`

其中保存：

- `input_ids`
- `positive_ids`
- `user_ids`

含义：

- `input_ids[t]` 是当前位置之前已经看到的 item 序列
- `positive_ids[t]` 是该位置要预测的下一个正样本 item
- 一个训练窗口里会同时监督多个位置，而不是只监督最后一个位置

### 2. 工具层新增序列监督数据集与负采样

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\sasrec_utils.py`

新增内容：

- `SequenceSupervisionDataset`
- `load_sequence_supervision_dataset(...)`
- `sample_uniform_negative_ids(...)`

当前负采样策略：

- 从 `1 ... num_items` 中均匀采样
- 避开当前位置正样本
- 避开当前输入窗口里已经出现过的历史 item

说明：

这已经比最初的 CE baseline 更接近 SASRec 论文训练方式，但还不是最极致的论文级完全对齐版本。

### 3. 模型补齐按位置候选打分能力

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\sasrec_model.py`

改动：

- `score_candidates(...)` 现在既能处理
  - 最后一个 hidden state 对多个候选 item 打分
  - 也能处理整条序列上每个位置对正负样本打分

### 4. 训练脚本支持双模式

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\train_sasrec.py`

新增参数：

- `--loss-type`
- `--num-negative-samples`

支持两种训练模式：

- `cross_entropy`
- `bce_negative_sampling`

其中 `bce_negative_sampling` 的训练逻辑是：

1. 编码整条输入序列得到每个位置的 hidden state
2. 用 `positive_ids` 构造正样本 logits
3. 用采样得到的 `negative_ids` 构造负样本 logits
4. 对有效位置做 `BCEWithLogitsLoss`
5. 每个 epoch 后仍然跑 validation ranking 评估

### 5. 实验脚本新增目标对齐 preset

文件：

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\run_sasrec_experiments.py`

新增 preset：

- `objective_alignment`

它会对比：

- `seq50_dim128_blocks3_drop02_ce`
- `seq50_dim128_blocks3_drop02_bce_ns1`

同时补了一个复用检查：

- 如果旧预处理目录里没有 `train_sequence_supervision.npz`
- 实验脚本会自动重新执行预处理

## 你接下来该怎么跑

### 先重新预处理

因为旧的预处理目录没有新加的 `train_sequence_supervision.npz`。

```bash
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\preprocess_movielens_1m.py"
```

### 跑 CE 对照组

```bash
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\train_sasrec.py" --embedding-dim 128 --num-blocks 3 --dropout 0.2 --loss-type cross_entropy --device cuda
```

### 跑 BCE 目标对齐组

```bash
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\train_sasrec.py" --embedding-dim 128 --num-blocks 3 --dropout 0.2 --loss-type bce_negative_sampling --num-negative-samples 1 --device cuda
```

### 直接跑成组实验

```bash
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\run_sasrec_experiments.py" --preset objective_alignment --device cuda
```

## 你接下来该重点观察什么

不要先看 training loss 漂不漂亮，先看：

- validation `NDCG@10`
- test `NDCG@10`
- BCE 是否比 CE 更稳定
- BCE 是否更早出现 overfitting

尤其是这几个问题：

- `BCE` 是否在 `valid NDCG@10` 上超过当前最强 CE baseline
- `BCE` 是否需要更多 epoch 才出效果
- `num_negative_samples=1` 是否太弱，后面是否要试 `5`

## 这一阶段完成后，下一步是什么

如果 `BCE + negative sampling` 明显有效，下一步继续对齐论文：

- 更手写化的 SASRec block，而不是完全依赖 `TransformerEncoder`
- 更接近论文习惯的采样与训练细节
- 对比不同负样本数
- 再决定是否进入更大结构改造
