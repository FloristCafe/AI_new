# Optiver 阶段补充笔记：CNN trade 改造与 stacking 结论

## 1. 本次补充的目标

这份补充笔记只压缩记录两类信息：

- `CNN trade` 支路当前已经完成的变量改造
- `stacking meta` 当前阶段最稳定的实验结论

目的不是重复旧笔记，而是给后续实验一个更短、更清晰的接力点。

## 2. CNN trade 支路已完成的变量改造

当前 `trade` 分支已经不再直接使用原始的：

- `price`
- `size`
- `order_count`

而是先把逐笔成交事件转成更适合 CNN 学习局部模式的派生变量，再写入 tensor。

当前已进入 `trade tensor` 的核心通道包括：

- `trade_return`
- `size`
- `order_count`
- `size_per_order`
- `trade_mid_gap`
- `trade_depth_ratio`
- `trade_impact`
- `size_imbalance1`

这些变量的含义可以压缩理解为：

- `trade_return`：描述成交价格的局部跳动
- `size_per_order`：区分大单推动和碎单密集成交
- `trade_mid_gap`：描述成交价相对盘口中间价的偏离
- `trade_depth_ratio`：描述该笔成交相对盘口深度的冲击强度
- `trade_impact = abs(trade_return) * size`：近似描述单笔事件强度
- `size_imbalance1`：把成交事件与盘口一档失衡状态连起来

## 3. 当前对 trade 分支的判断

当前实验已经比较稳定地说明：

- `trade_only` 明显弱于 `book_only`
- `book_only` 能提供更稳定的状态信号
- `dual CNN` 又优于单独 `book_only`

这说明：

- `trade` 本身噪声较大，单独建模效果弱
- 但 `trade` 不是无效信息
- 当它和 `book` 状态一起出现时，可以提供事件冲击和局部活跃度的补充信号

因此，当前最合理的定位不是“让 trade 单独打主力”，而是：

- 把 `trade` 当作 `dual CNN` 中的辅助分支
- 用来补充 `book` 分支看不到的成交事件动态

## 4. dual CNN 当前在总体系中的定位

当前 `dual CNN` 更适合被视为第一层基学习器之一，而不是当前最强的单模型主线。

可以这样理解：

- `book` 分支负责提供稳定的微观状态主信号
- `trade` 分支负责提供事件冲击、成交密度、局部活跃度信号
- `dual CNN` 的价值主要体现在给集成或 stacking 提供不同视角

当前结论不是“CNN 替代表格模型”，而是：

- 表格线擅长样本级聚合统计和 KNN 历史近邻信息
- CNN 线擅长逐秒局部模式
- 两者适合互补，而不是互相替代

## 5. stacking 已经升级为标准 OOF 版本

当前项目已经不再只是简单 late fusion，而是有了标准 OOF stacking 链路。

当前 stacking 的核心结构是：

- 第一层基学习器：`LightGBM + MLP + dual CNN`
- 外层切分：按时间顺序形成 `train / valid`
- 在外层 `train` 上做 `GroupKFold(time_id)`，生成 OOF 预测
- 第二层：用 `meta learner` 在 OOF 预测上训练

所以现在的 stacking 本质上是：

- 第一层多个模型各自学习
- 第二层学习“什么时候信谁”

这已经是完整的 second-stage supervised learning，而不是手工加权平均。

## 6. 当前 OOF 缓存与加速策略

为了避免每次都重跑第一层，当前已经把 stacking 的关键中间结果缓存下来了。

当前可直接复用的文件包括：

- `stacking_oof_predictions.csv`
- `stacking_valid_predictions.csv`

它们的位置在：

- [stacking_oof_predictions.csv](<D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\features_knn\stacking_models\stacking_oof_predictions.csv>)
- [stacking_valid_predictions.csv](<D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16-17-18-19-20-21-22-23-24_times_200\features_knn\stacking_models\stacking_valid_predictions.csv>)

这意味着后续可以区分两种实验：

- 第一层改变时：重跑完整 `train_optiver_stacking.py`
- 只改 meta 层时：直接基于缓存运行 `train_optiver_meta_only.py`

这样可以显著减少重复训练 `LightGBM / MLP / CNN` 的时间成本。

## 7. meta 层当前最稳定的结论

当前二层模型的实验结论已经比较清楚：

- `Ridge` 是当前最稳定的 `meta learner`
- 更复杂的 second-stage 模型更容易在 OOF 特征上过拟合
- `meta-only` 微调目前没有稳定超过最优标准 stacking 结果

当前最好的标准 stacking 结果约为：

- `valid R^2 = 0.8273`
- `valid RMSE = 0.001166`

这说明：

- stacking 框架本身是成立的
- CNN 的确提供了真实但有限的补充信息
- 当前瓶颈不在“meta 不够复杂”，而在“第一层的互补信息量还不够强”

## 8. 当前暴露出来的核心问题

当前更重要的问题已经不是第二层形式，而是第一层差异化程度：

- `LightGBM` 仍然是最稳、最强的主力基学习器
- `CNN` 现在已经能提供有限但真实的补充信号
- `MLP` 目前还没有稳定学出和 `LightGBM` 足够不同的模式

因此，后续真正高优先级方向应当是：

- 为 `MLP` 重新做更适合梯度学习的特征视图
- 继续增强 `trade` 事件表达，而不是只调激活函数
- 只有当第一层信息真的变化时，才重跑完整 stacking
- 平时二层实验优先利用 OOF 缓存快速调参

## 9. 一句话结论

当前项目已经进入这样一个阶段：

- `CNN trade` 已从“原始成交流”升级为“微观结构派生事件流”
- `dual CNN` 已经具备集成价值
- `stacking` 已经升级为标准 OOF 流程
- `Ridge meta learner` 是当前最稳妥的 second-stage 选择
- 下一步真正该发力的，是第一层基学习器的差异化，而不是继续盲目加复杂 meta 模型
