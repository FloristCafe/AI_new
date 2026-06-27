# Optiver 阶段笔记：MLP Feature View 试验整理

## 1. 本次任务目标

本轮工作的核心目标是解决一个明确问题：

- 当前输入给 `MLP` 的特征表示不够契合神经网络的建模偏好
- 现有特征更偏向树模型友好，而不是 `MLP` 友好
- 需要构造一套专门给 `MLP` 使用的 feature view，使其真正学出与 `LightGBM` 不同的模式

更具体地说，本轮希望同时改善两件事：

- 让 `MLP` 更好利用平滑连续特征
- 让 `MLP` 在不引入高方差噪声的前提下吸收样本上下文信息

## 2. 研究思路演化

本轮实验不是简单调 hidden size，而是围绕输入表示做系统迭代。

总体路线经历了五个阶段：

1. 原始 `features_knn` 上直接训练 `MLP`
2. 第一版 `mlp_view`
3. 加重历史上下文与 KNN 上下文的 `v2`
4. 加入数值防爆后的 `v3 / v4`
5. 回退到更克制的 `lite` 版本

核心结论是：

- `MLP` 确实需要单独的 feature view
- 但这份 feature view 不能做得过重
- 在当前样本规模下，轻量、平滑、低方差的派生特征视图最有效

## 3. 第一版 MLP 专属特征视图

第一版 `MLP-specific feature view` 的出发点是：

- 对重尾连续变量做 `log1p`
- 对极端值做 winsorize
- 加入 `per-stock z-score`
- 加入 `per-time cross-sectional z-score / rank`
- 加入少量 `book x trade` 连续交互
- 加入有限的历史偏移和窗口差分

这一版的核心思想是：

- 不再让 `MLP` 直接吃一整张偏树模型视角的原始表
- 而是先把特征映射到更平滑、更连续、更可比较的空间

实验结果表明：

- 方向是正确的
- `MLP` 的单模表现相对原始版有所改善
- 但增益仍然有限，说明 feature view 还可以继续增强

## 4. v2：重历史上下文与 KNN 上下文的失败

第二步尝试是沿着“让 `MLP` 看到更多样本联系”继续推进，加入：

- `EMA`
- `rolling mean / std / z-score`
- 更强的历史偏移
- 更显式的 `KNN context`
- 更多 cross-median / ratio 特征

直觉上，这样做似乎能让 `MLP` 吸收更多“样本之间的联系”。

但实验暴露出第一个严重问题：

- `valid R^2` 直接跌到负数
- 不是模型随机波动，而是特征数值结构本身失稳

定位后发现，问题主要来自两类特征：

1. 危险 ratio

- 某些短窗 / 长窗均值比
- `trade_impact / spread`
- KNN 距离比

当分母接近 0 时，ratio 会出现极端爆炸值。

2. 短历史 z-score

- `rolling z-score`
- `same-stock history z-score`

在历史样本很少或标准差很小时，分母非常小，会导致 z-score 异常放大。

这一步的关键结论是：

- 当前 `MLP` 不是不能吃上下文
- 而是不能直接吃未重参数化、未稳健处理的复杂历史上下文

## 5. v3：数值防爆修复后，仍然过拟合

针对 `v2` 的数值爆炸问题，后续加入了：

- 最小分母阈值
- feature clipping
- 对历史 z-score 加最小方差约束

这一版确实修复了直接数值爆炸的问题，但新的实验结果说明：

- 虽然不再崩盘
- 但 `train` 提升，`valid` 反而下降

这说明问题已经从“数值错误”转变为“特征信息质量不佳”：

- 新特征在训练集上可学
- 但不具备稳定泛化能力

因此，`v3` 的结论不是“继续多修几个阈值就行”，而是：

- 整条“重历史上下文”路线已经进入高方差区

## 6. v4：对数化 / asinh / log-ratio 仍未解决主问题

之后又进一步尝试：

- 将危险 ratio 改写为 `log-ratio`
- 对偏移量和交互量加入 `asinh`
- 希望通过更合理的分布重参数化挽救这条路线

这一步在方法上是专业且合理的，但实验结果仍然失败：

- `valid R^2` 再次显著为负

这一步的价值不在于结果变好，而在于结论被彻底试清：

- 问题已经不是“个别比值没处理好”
- 而是当前 feature view 整体过度工程化

当 `MLP` 同时看到：

- 大量历史链
- KNN 上下文
- 横截面相对特征
- 多层次重参数化衍生变量

在 `4800` 条样本规模下，模型更容易学到高方差局部模式，而不是稳定规律。

## 7. 方向反转：从重 feature view 回退到 lite

在 `v2 / v3 / v4` 连续失败之后，本轮思路发生了重要转折：

不是继续给 `MLP` 堆更多上下文，而是主动做减法。

新的设计原则变成：

- 减少原始特征
- 保留更多稳定派生特征
- 删除高方差历史链和复杂 ratio

对应实现为独立脚本：

- [build_optiver_mlp_view_lite.py](<D:\Python\Artificial Intelligence\projects\incubator\optiver_realized_volatility_starter\src\build_optiver_mlp_view_lite.py>)

这版 `lite` 的特征策略是：

保留少量原始核心列：

- `spread1_mean`
- `size_imbalance1_mean`
- `size_imbalance1_std`
- `realized_vol_wap1`
- `realized_vol_mid1`
- `trade_active_ratio`
- `trade_price_std`
- `trade_impact_mean`
- `trade_depth_ratio_mean`
- `realized_vol_trade_price`
- `knn_global_target_mean_k5`
- `knn_same_stock_target_mean_k3`

加入的派生特征主要是：

- `log1p`
- winsorized version
- `stock z-score`
- `cross z-score`
- `cross rank`
- `window delta`
- 少量 `book x trade` 连续交互
- 少量 `asinh`

明确删除的高方差部分包括：

- `EMA history chains`
- `rolling z-score chains`
- 大部分 `history z`
- 大部分 `cross median ratio`
- 大部分复杂 `KNN context` 比值链

## 8. lite 结果：当前 MLP 线上最优

`lite` 版的实验结果如下：

- Train
  - `MAE = 0.0008014`
  - `RMSE = 0.0012001`
  - `R^2 = 0.8724`
  - `RMSPE = 0.2758`

- Valid
  - `MAE = 0.0007537`
  - `RMSE = 0.0011297`
  - `R^2 = 0.8283`
  - `RMSPE = 0.3073`

这是当前 `MLP` 线最重要的一次突破。

其意义在于：

1. `MLP` 不仅被“救活”了，而且已经进入强基学习器区间
2. `valid R^2 = 0.8283` 明显优于前面几版 `mlp_view`
3. `train-valid` 落差相对可控，没有出现 `v3 / v4` 那种高方差崩盘

## 9. 方法论结论

这一轮实验最重要的，不只是拿到一个更高的 `MLP valid R^2`，而是得出了一条明确方法论：

对于当前 Optiver 任务与当前样本规模，

**MLP 更适合“少原始特征、多稳定派生特征”的轻量 feature view，而不适合重历史、重邻居、重复杂上下文的高方差 feature view。**

换句话说：

- `MLP` 的问题不是“模型结构不够强”
- 而是输入表示必须刻意设计成低方差、平滑、相对化的样子

这条结论也修正了此前一个重要直觉：

- 如果想让 `MLP` 利用样本间联系，不是把更多历史链条直接扔进去
- 而是把这些联系压缩成稳定的派生统计量和相对位置表达

## 10. 当前项目状态更新

截至今天，`MLP` 线的状态应更新为：

- 原始 `MLP`：较弱
- 第一版 `mlp_view`：有效但增益有限
- `v2 / v3 / v4`：证明重上下文路线在当前样本规模下不稳定
- `mlp_view_lite`：当前最优 `MLP` 视图

因此，当前最合理的下一步是：

1. 保留 `features_mlp_view_lite`
2. 让 `MLP` 在第一层集成中改吃这份 lite 表
3. 与当前 `LightGBM(features_knn)`、`dual CNN(tensor)` 一起重新进入标准 OOF stacking

## 11. 一句话总结

本轮实验的最重要结论是：

**对当前 Optiver 任务，MLP 的正确输入策略不是“更多历史上下文”，而是“更少原始特征 + 更多稳定派生特征”。**

这使得 `MLP` 从一个长期偏弱的辅助组件，转变为一个值得重新纳入第一层 stacking 的强候选基学习器。
