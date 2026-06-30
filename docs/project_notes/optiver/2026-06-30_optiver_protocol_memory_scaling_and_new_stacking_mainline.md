# Optiver 阶段笔记：协议修正、内存优化、扩样本与新 Stacking 主线

## 1. 本轮工作目标

本轮工作的核心目标不是继续盲目调单个模型超参，而是先把整个 Optiver 项目的训练与评估地基重新打稳，再基于更可信的协议扩大样本、重建主线。

本轮重点完成了四类事情：

- 修正单模型 baseline 的验证协议，使其与 stacking 更接近
- 对 tabular 与 stacking 主流程做内存优化，降低后续扩样本风险
- 将样本规模从约 5000 扩大到 20000
- 重新验证 LightGBM、MLP、CNN 与 stacking 的真实相对位置，并确定新的主线配置

## 2. 评估协议修正

### 2.1 旧问题

此前 `train_optiver_baseline.py` 使用的是：

- 先按 `stock_id, time_id` 排序
- 再按行数比例直接切分 train / valid

这种方式会比真正按 `time_id` 分组切分更乐观，因此会高估单模型尤其是 MLP 的表现。

这也是此前出现下面这种现象的重要原因：

- 单独跑 MLP 时 `valid R^2` 能到约 `0.84`
- 但一进入 stacking 的 `time_id` 分组协议后，MLP 表现明显下滑

### 2.2 修正内容

已修改：

- `src/train_optiver_baseline.py`

新增参数：

- `--split-mode time_id`
- `--split-mode row`

当前默认已经改为：

- `time_id`

这样单模型 baseline 与 stacking outer split 的物理意义更一致。

### 2.3 修正后的 MLP 真实水平

在旧协议下，MLP 在 `features_mlp_view_lite` 上曾拿到过更高的 valid 分数；修正为 `time_id` 分组后，重新评估得到：

- Train
  - `MAE = 0.0007717`
  - `RMSE = 0.0012045`
  - `R^2 = 0.8712`
  - `RMSPE = 0.2598`
- Valid
  - `MAE = 0.0008586`
  - `RMSE = 0.0012651`
  - `R^2 = 0.7968`
  - `RMSPE = 0.3513`

关键结论：

- 旧 baseline 协议确实偏乐观
- MLP 的真实水平更接近 `valid R^2 ≈ 0.80`
- 后续所有模型比较应优先基于 `time_id` 协议

## 3. 内存与数据流优化

### 3.1 目标

在扩样本前，先解决最容易白白浪费内存的部分，为后续向更大样本推进做准备。

### 3.2 已完成优化

新增工具：

- `src/optiver_memory_utils.py`

实现内容：

- 将 tabular 浮点列统一压成 `float32`
- 将整数列尽量 downcast

已接入脚本：

- `build_optiver_features_v2.py`
- `build_optiver_knn_features.py`
- `build_optiver_mlp_view_lite.py`
- `train_optiver_baseline.py`
- `train_optiver_stacking.py`

### 3.3 stacking 主流程优化

在 `train_optiver_stacking.py` 中继续做了两类优化：

1. OOF 主流程尽量前移到 NumPy

- `X_train_lgbm`
- `X_valid_lgbm`
- `X_train_mlp`
- `X_valid_mlp`

已经尽量提前转为 `np.float32` NumPy 矩阵，减少 OOF 折内反复 `.iloc + reset_index` 的开销。

2. CNN 侧轻量减重

- `torch.tensor(...)` 改为 `torch.from_numpy(...)`
- `np.load(..., mmap_mode="r")` 用于 stacking 中 tensor 读取

当前状态判断：

- tabular 侧的第一轮内存优化已基本完成
- CNN 侧尚未做到真正的流式分批加载，但已进行了低成本减重

## 4. GPU 环境打通

本轮还完成了 PyTorch CUDA 环境修复。

已确认硬件：

- `NVIDIA GeForce RTX 5070 Laptop GPU`

此前问题：

- 环境中的 PyTorch 为 `2.12.0+cpu`
- CNN 实际一直跑在 CPU

后续修复为：

- `torch 2.9.1+cu128`
- `torch.cuda.is_available() == True`

结果：

- `train_optiver_dual_cnn.py`
- `train_optiver_stacking.py` 中的 CNN 分支

现在都可以正常走 GPU。

## 5. src 目录整理与脚本归档

为了保持主线目录清晰，本轮对 `src` 做了整理。

新增归档目录：

- `src/archive_experiments`

已归档历史试验脚本：

- `build_optiver_baseline_features.py`
- `build_optiver_mlp_view_features.py`
- `build_optiver_mlp_view_plus.py`

保留在顶层的脚本都是当前主线仍在使用或仍有明确工具价值的脚本。

## 6. 样本规模扩充

### 6.1 扩样前状态

此前主线样本大约为：

- `25` 个股票
- 每股 `200` 个 `time_id`
- 总样本约 `5000`

### 6.2 本轮扩样结果

本轮扩展到了：

- `50` 个股票
- 每股 `400` 个 `time_id`
- 总样本数 `20000`

对应目录：

- `optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-26-27-28-29-30-31-32-33-34-35-36-37-38-39-40-41-42-43-44-46-47-48-50-51-52-53-55_times_400`

### 6.3 新数据产物

1. `features_v2`

- 形状：`20000 x 73`

2. `features_knn`

- 形状：`20000 x 81`
- KNN 列：
  - `knn_global_target_mean_k5`
  - `knn_global_target_std_k5`
  - `knn_global_dist_mean_k5`
  - `knn_global_dist_min_k5`
  - `knn_same_stock_target_mean_k3`
  - `knn_same_stock_target_std_k3`
  - `knn_same_stock_dist_mean_k3`
  - `knn_same_stock_dist_min_k3`

3. `features_mlp_view_lite`

- 形状：`20000 x 91`
- 核心 raw 特征 `12` 个
- 派生特征 `76` 个

4. `cnn_tensors`

- `book_tensor_shape = (20000, 6, 600)`
- `trade_tensor_shape = (20000, 8, 600)`

## 7. 新样本上的单模型重评估

扩样后，在统一协议下重新训练单模型，结论如下。

### 7.1 LightGBM

使用：

- `features_knn`
- `time_id` split
- bagging

结果：

- Train
  - `MAE = 0.0006753`
  - `RMSE = 0.0011303`
  - `R^2 = 0.8712`
  - `RMSPE = 0.3011`
- Valid
  - `MAE = 0.0008006`
  - `RMSE = 0.0016003`
  - `R^2 = 0.8171`
  - `RMSPE = 0.2982`

结论：

- LightGBM 仍然是很稳的 tabular 主力

### 7.2 MLP

使用：

- `features_mlp_view_lite`
- `time_id` split

结果：

- Train
  - `MAE = 0.0008006`
  - `RMSE = 0.0013529`
  - `R^2 = 0.8154`
  - `RMSPE = 0.3190`
- Valid
  - `MAE = 0.0008745`
  - `RMSE = 0.0016540`
  - `R^2 = 0.8047`
  - `RMSPE = 0.3251`

结论：

- `mlp_view_lite` 在更大样本下依然成立
- MLP 没有崩，说明这条线路是可持续的
- 但当前仍略弱于 LightGBM

### 7.3 Dual CNN

使用：

- `book + trade` 双路 CNN
- GPU 训练

结果：

- Train
  - `MAE = 0.0007992`
  - `RMSE = 0.0013798`
  - `R^2 = 0.8080`
  - `RMSPE = 0.3598`
- Valid
  - `MAE = 0.0008612`
  - `RMSE = 0.0015894`
  - `R^2 = 0.8196`
  - `RMSPE = 0.3528`

结论：

- 扩样本后，CNN 已经追上并轻微超过 LightGBM
- 这说明序列建模在更大样本下开始真正受益
- `trade` 支路虽然仍不是单独主力，但在双路结构中开始体现辅助价值

## 8. Stacking 中 MLP 异常掉分的定位

### 8.1 现象

新样本上第一次跑标准 stacking 时，发现：

- `mlp_valid R^2` 异常降到约 `0.704`

但单独跑大样本 MLP 时：

- `valid R^2 ≈ 0.805`

这个落差过大，必须排查。

### 8.2 排查步骤

先做最小改动实验：

- 将 stacking 的 `bagging-size` 从 `3` 改为 `1`

结果：

- `mlp_valid R^2` 从约 `0.704` 恢复到约 `0.799`
- 同时整体 stacking `valid R^2` 也从约 `0.827` 回升到约 `0.833`

### 8.3 结论

关键问题不是严重对齐 bug，而是：

- **bootstrap bagging 会明显伤害当前这版 MLP**

更准确地说：

- LightGBM 适合继续做 bagging
- MLP 对 bootstrap 重采样更敏感
- 在当前 Optiver 数据分布、当前 `mlp_view_lite` 和当前样本规模下，MLP 的有效信号会被 bagging 稀释

## 9. 拆分 bagging 参数

为了解决上述问题，已修改：

- `src/train_optiver_stacking.py`

新增参数：

- `--lgbm-bagging-size`
- `--mlp-bagging-size`

并保留：

- `--bagging-size`

作为全局 fallback。

这样当前最合理的设定变成：

- `LightGBM bagging = 3`
- `MLP bagging = 1`

## 10. 当前最优 stacking 主线

在新样本 `20000`、新协议、拆分 bagging 后，当前最佳 stacking 结果如下。

### 10.1 最终结果

- Train
  - `MAE = 0.0007398`
  - `RMSE = 0.0013123`
  - `R^2 = 0.8263`
  - `RMSPE = 0.3212`
- Valid
  - `MAE = 0.0007813`
  - `RMSE = 0.0015133`
  - `R^2 = 0.8365`
  - `RMSPE = 0.2866`

### 10.2 同轮组件表现

- `lightgbm_valid R^2 = 0.8172`
- `mlp_valid R^2 = 0.7985`
- `cnn_valid R^2 = 0.8194`

### 10.3 当前主线配置

第一层：

- `LightGBM(features_knn)`，`bagging=3`
- `MLP(features_mlp_view_lite)`，`bagging=1`
- `dual CNN(cnn_tensors)`

第二层：

- `Ridge meta learner`
- `meta_use_augmented_features = True`

### 10.4 当前结论

这已经可以视为当前 Optiver 项目的新主线：

- 评估协议已比此前更可信
- MLP 不再被 bagging 错误伤害
- CNN 在更大样本上追上 LightGBM
- stacking 明确优于各单模型

## 11. 对模型逻辑的再理解

本轮还把一个关键认知问题理清了：

- 当前任务不是“每只股票训练一个独立模型”
- 而是“训练一个统一模型，预测每个 `(stock_id, time_id)` 样本的波动率”

即：

- 样本单位是“某只股票在某个时间窗口的市场状态”
- 模型学习的是“市场微观结构状态 -> 未来波动率”的映射

虽然前向传播时没有将裸 `stock_id` 当作普通 dense 数值特征直接输入，但：

- same-stock KNN
- stock-relative 特征
- cross-sectional 特征
- CNN 的序列形态

都已经间接编码了个股身份与个股常态信息。

因此模型不是在“按股票名字查表”，而是在根据该股票当前窗口呈现出的微观结构状态来预测波动率。

## 12. 当前阶段总结

本轮最重要的阶段性结论有四个：

1. 单模型 baseline 的旧 row split 协议确实会高估表现，后续比较应统一到 `time_id` 协议
2. 内存优化和 NumPy-first OOF 已经让后续扩样本更可行
3. 扩样到 `20000` 后，CNN 已经从辅助路线成长为与 LightGBM 同级的强基学习器
4. 当前最优集成策略不是“所有模型统一 bagging”，而是：
   - `LightGBM bagging`
   - `MLP 不 bagging`
   - `dual CNN`
   - `Ridge stacking`

## 13. 下一步建议

当前主线已经足够稳定，后续可以优先考虑两条路线中的一条：

1. 继续扩样本

- 复用当前已验证主线配置
- 观察更大规模下 CNN 和 MLP 是否进一步受益

2. 固化当前主线并写正式汇报

- 当前 `valid R^2 ≈ 0.8365`
- 已经足以形成一版比较完整的阶段性研究汇报

现阶段不建议再回头做这些事：

- 恢复旧 baseline 行切分协议
- 对 MLP 重新启用 bagging
- 回到 `mlp_view_plus` 或历史高方差视图

## 14. 一句话总结

本轮工作的核心价值在于：

**先修正评估协议与训练管线，再扩样本，最终确认了一个可信、稳定、具备明确增益的 Optiver 新主线：LightGBM(KNN) + MLP(lite, no bagging) + dual CNN + Ridge stacking。**
