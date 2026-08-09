# MovieLens 1M Generative Slate DQN

这是一个真实离散序列强化学习项目，目标是在你现有的 `MovieLens-1M SASRec baseline` 之上，搭建一个可在线交互的 `Generative Slate-RL` 工程闭环：

- 冻结 `SASRec` 作为用户状态编码器
- 自回归生成长度为 5 的推荐列表
- 用手写 `SlateRecSimEnv` 做在线曝光/点击结算
- 用共享 `R_total` 更新 5 个 micro-actions 的条件 Q 值

## 项目目标

- 把单点推荐动作扩展成长度固定的 slate 生成动作
- 让智能体感知已经放进列表前缀的物品上下文
- 在在线沙盒环境里学习位置偏置与同质化惩罚
- 保持脚本可直接跑通，训练后保存 checkpoint 和 `metrics.json`

## 状态、动作、奖励定义

- `State S`
  - 用户最近 50 个交互 item 序列
  - 用冻结的 `SASRec` 编码为用户状态向量
- `Action`
  - 在单个 slate 内连续生成 5 次 item
  - 第 `k` 次动作学习条件价值 `Q(S, a_1, ..., a_{k-1}, a_k)`
- `Reward`
  - 整个 slate 的总点击数
  - 每个 micro-action 共享同一个 `R_total`

## 环境设计

- 环境实现：`D:\Python\Artificial Intelligence\projects\recommendation\recommender_mdp_gymnasium\micro_recsim_env.py`
- 新增子类：`SlateRecSimEnv`
- 结算规则：
  - 位置曝光概率：`1 / log2(k + 1)`
  - 同类目重复惩罚：同类型 item 的后续位置点击率打五折
  - 类别疲劳更新：点击过的类别疲劳上升，未点击类别随时间衰减

## 目录结构

- `src/slate_dqn_model.py`
  - 冻结 `SASRecEncoder`
  - 用 `GRU` 编码已生成的 slate prefix
  - 拼接用户状态与 prefix context，输出 3706 维条件 Q 值
- `src/train_generative_slate_dqn.py`
  - 从 MovieLens 序列池采样 episode 初始状态
  - 在线生成 slate
  - 把 slate 送进 `SlateRecSimEnv`
  - 用共享 `R_total` 更新 replay buffer 中的 5 个 micro-actions
- `artifacts/`
  - 保存 `checkpoints/` 与 `metrics/`

## 上游复用资产

- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline`
- `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_dqn_offline_rl`

## 默认数据与模型路径

- 训练序列池：
  - `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\train_sequence_supervision.npz`
- 验证序列池：
  - `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\preprocessed\valid_sequences.npz`
- 冻结 SASRec checkpoint：
  - `D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_sasrec_baseline\artifacts\experiments\seq50_dim128_blocks3_drop02_ce\checkpoints\sasrec_best.pt`

## 运行顺序

1. 确保上游 `movielens_1m_sasrec_baseline` 已经产出预处理数据和 `sasrec_best.pt`
2. 运行在线训练脚本
3. 查看 `artifacts/checkpoints/` 和 `artifacts/metrics/training_metrics.json`

## 示例命令

```powershell
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn\src\train_generative_slate_dqn.py" --device cpu --total-episodes 50 --eval-interval 10 --eval-episodes 10
```

```powershell
python "D:\Python\Artificial Intelligence\projects\reinforcement_learning\movielens_1m_generative_slate_dqn\src\train_generative_slate_dqn.py" --device cuda --total-episodes 1000 --eval-interval 100 --eval-episodes 100
```
