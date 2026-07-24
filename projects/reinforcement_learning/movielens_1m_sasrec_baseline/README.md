# MovieLens 1M SASRec Baseline

这是一个真实离散序列推荐项目，目标是基于 MovieLens-1M 用户行为序列手写复现 SASRec baseline。

## 项目目标

- 使用 `user_id, movie_id, timestamp` 构建用户交互序列
- 按时间排序，做 leave-one-out 切分
- 使用固定长度序列训练 SASRec
- 评估 `HR@10` 与 `NDCG@10`

## 目录结构

- `src/preprocess_movielens_1m.py`
  - 读取清洗后的交互数据
  - 按用户和时间排序
  - 过滤低频用户与低频物品
  - 生成固定长度训练样本与验证/测试评估样本
- `src/sasrec_model.py`
  - 定义 item embedding
  - 定义 learnable positional embedding
  - 定义 causal self-attention 编码器
  - 提供最后位置表示与全量 item 打分接口
- `src/train_sasrec.py`
  - 读取预处理产物
  - 训练 SASRec
  - 每个 epoch 在验证集上计算 `HR@10` 与 `NDCG@10`
  - 按验证指标选择 best checkpoint
  - 支持 early stopping
  - 保存 `sasrec_best.pt` 与 `sasrec_final.pt`
  - 保存 `training_metrics.json`
- `src/evaluate_sasrec.py`
  - 按 leave-one-out 协议评估
  - 计算 `HR@10` 与 `NDCG@10`
  - 保存验证或测试指标
- `src/run_sasrec_experiments.py`
  - 统一管理多组实验
  - 自动调用预处理、训练、测试评估
  - 汇总多组配置的结果到 summary 文件
  - 当前支持 `baseline_triplet`、`capacity_and_regularization`、`dim128_finegrained_structure`

## 默认数据路径

- 原始交互：`D:\Python\Datasets\movielens_1m\processed\interactions.csv`
- 项目目录：`D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline`

## 预处理产物

`artifacts/preprocessed/` 目录下会生成：

- `train_sequences.npz`
  - `input_ids`: 训练输入序列，形状为 `(N, max_seq_len)`
  - `target_ids`: 每条训练样本要预测的下一个 item
  - `user_ids`: 对应样本的用户编号
- `valid_sequences.npz`
  - 每个用户倒数第二次交互的评估样本
- `test_sequences.npz`
  - 每个用户最后一次交互的评估样本
- `metadata.json`
  - 数据规模、过滤阈值、路径和样本数汇总
- `user_id_mapping.csv`
- `item_id_mapping.csv`

## 训练产物

- `artifacts/checkpoints/sasrec_best.pt`
- `artifacts/checkpoints/sasrec_final.pt`
- `artifacts/metrics/training_metrics.json`
  - 包含每个 epoch 的 train / valid 指标曲线

## 评估产物

- `artifacts/predictions/valid_metrics.json`
- `artifacts/predictions/test_metrics.json`

## 多实验产物

- `artifacts/experiments/<run_name>/`
- `artifacts/experiments/prepared_data/`
- `artifacts/experiments/summaries/<preset>_summary.json`
- `artifacts/experiments/summaries/<preset>_summary.csv`

## 推荐执行顺序

1. 先运行 `preprocess_movielens_1m.py`
2. 再运行 `train_sasrec.py`
3. 最后运行 `evaluate_sasrec.py`

## 示例命令

```powershell
& "C:\Users\lenovo\miniconda3\envs\kg_env\python.exe" "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\preprocess_movielens_1m.py"
```

```powershell
& "C:\Users\lenovo\miniconda3\envs\kg_env\python.exe" "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\train_sasrec.py"
```

```powershell
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\train_sasrec.py" --selection-metric ndcg_at_10 --early-stop-patience 3
```

```powershell
& "C:\Users\lenovo\miniconda3\envs\kg_env\python.exe" "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\evaluate_sasrec.py"
```

```powershell
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\run_sasrec_experiments.py" --preset baseline_triplet --device cuda
```

```powershell
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\src\run_sasrec_experiments.py" --preset dim128_finegrained_structure --device cuda
```
