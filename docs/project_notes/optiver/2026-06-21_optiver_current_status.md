# Optiver 当前实验进展笔记

## 1. 当前任务定位

当前项目是 Kaggle `Optiver Realized Volatility Prediction` 的 baseline 研究。

本阶段目标不是直接追求 leaderboard 极限成绩，而是先建立一条可复现、可扩展、可解释的实验流水线：

- 从原始 `book` / `trade` 高频数据中抽取 sandbox 样本
- 构建样本级微观结构特征
- 引入 KNN 相似样本特征
- 比较 `Ridge`、`RandomForest`、`LightGBM`、`MLP` 以及 `MLP + LightGBM` 集成效果


## 2. 数据与目录结构

数据根目录：

- `D:\Python\Datasets\optiver_realized_volatility_prediction`

原始数据：

- `raw_extracted\train.csv`
- `raw_extracted\book_train.parquet`
- `raw_extracted\trade_train.parquet`

当前主要 sandbox：

1. 小样本 sandbox

- `samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7_times_80`
- 规模：`8 x 80 = 640` 样本

2. 扩容后 sandbox

- `samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16_times_160`
- 规模：`16 x 160 = 2560` 样本


## 3. 当前特征工程方案

### 3.1 v2 基础特征

当前主特征表来自 `features_v2`，主要包含：

- `book` 侧统计特征
  - `wap1 / wap2` 均值与波动
  - `spread` 均值与波动
  - `mid price` 波动
  - `total volume`
  - `size imbalance`
  - `price spread ratio`
  - `realized volatility`

- `trade` 侧统计特征
  - `trade price` 均值与波动
  - `trade size` 总量、均值、标准差、最大值
  - `order_count`
  - `size_per_order`
  - `trade price realized volatility`

- 时间窗口特征
  - `last 150s`
  - `last 300s`

### 3.2 KNN 特征

当前 KNN 特征是基于已有样本级特征表做的“只看过去样本”的近邻增强，避免了目标泄漏。

已加入的 KNN 特征共有 8 个：

- `knn_global_target_mean_k5`
- `knn_global_target_std_k5`
- `knn_global_dist_mean_k5`
- `knn_global_dist_min_k5`
- `knn_same_stock_target_mean_k3`
- `knn_same_stock_target_std_k3`
- `knn_same_stock_dist_mean_k3`
- `knn_same_stock_dist_min_k3`

当前主训练表：

- `D:\Python\Datasets\optiver_realized_volatility_prediction\samples\optiver_sandbox_stocks_0-1-2-3-4-5-6-7-8-9-10-11-13-14-15-16_times_160\features_knn\optiver_features_knn.parquet`


## 4. 关键实验结论

### 4.1 小样本阶段的判断

在 `8 x 80 = 640` 样本规模下：

- `Ridge` 能跑通，但上限有限
- `LightGBM` 有一定效果
- `RandomForest + KNN` 一度是当前最优 baseline
- `MLP` 初版出现明显数值失控，拖垮集成

结论：

- 小样本下 KNN 历史池太浅
- MLP 对这种“小目标值 + 高频聚合特征 + 小样本”问题非常敏感
- 单纯引入神经网络并不会自然优于树模型


### 4.2 MLP 数值问题与修复

早期 `MLP` 出现过严重崩溃：

- 预测尺度远大于真实 target
- 集成后出现大幅负 `R²`

定位结果：

- 不是 `LightGBM` 的问题
- 不是 `KNN` 本身的错误
- 是 `MLP` 在当前回归任务上的目标尺度和训练稳定性出了问题

已做修复：

- 输入特征标准化
- 目标值标准化训练
- 输出裁剪为非负
- 收缩 MLP 结构与学习率

修复后结论：

- `MLP` 从“坏组件”变成了“有效但不如树模型强的辅助组件”


### 4.3 扩样本后的最新结果

在 `16 x 160 = 2560` 样本规模下，`KNN` 与集成效果明显改善。

当前最重要的一组结果：

- 模型：`MLP + LightGBM ensemble`
- 特征表：`features_knn`
- 权重：`LightGBM 0.7`，`MLP 0.3`

结果：

- Train
  - `MAE = 0.000619`
  - `RMSE = 0.000952`
  - `R² = 0.9158`
  - `RMSPE = 0.2128`

- Valid
  - `MAE = 0.000724`
  - `RMSE = 0.001230`
  - `R² = 0.8226`
  - `RMSPE = 0.2592`

组件表现：

1. `LightGBM`

- `valid R² = 0.8202`
- `valid RMSE = 0.001238`

2. `MLP`

- `valid R² = 0.7842`
- `valid RMSE = 0.001357`

3. 集成后

- `valid R² = 0.8226`
- `valid RMSE = 0.001230`

结论：

- 扩样本是强有效操作
- KNN 在更大历史池下开始真正发挥作用
- `MLP` 虽然仍弱于 `LightGBM`，但已经能提供补充信息
- `MLP + LightGBM` 集成成立，但目前只带来小幅增益，不是质变


## 5. 当前最可信的判断

### 5.1 已经确认的事实

- 当前 Optiver 任务更适合先走“高频聚合特征 + 树模型回归”路线
- KNN 特征在小样本下作用有限，在更大样本下明显变强
- MLP 不是当前主力，但在修复后可以作为辅助分支

### 5.2 当前策略排序

从“工程可靠性 + 当前效果”综合看：

1. `LightGBM + KNN`
2. `MLP + LightGBM + KNN`
3. `RandomForest + KNN`
4. `MLP`
5. `Ridge`

说明：

- 由于当前缺少同尺度下最新的 `RandomForest` 对照，`LightGBM` 与 `MLP + LightGBM` 目前是最新最强结果
- `MLP + LightGBM` 相比 `LightGBM` 只提升了一点点，因此是否保留集成，要看后续同尺度对照结果


## 6. 当前存在的问题

### 6.1 warning 仍需清理

当前训练日志中仍有 sklearn / LightGBM 的提示：

- `X does not have valid feature names, but LGBMRegressor was fitted with feature names`

这不影响当前结果有效性，但说明：

- `fit` 与 `predict` 阶段传入 LightGBM 的对象格式不完全一致
- 后续最好在代码层面统一为 DataFrame 或统一为 ndarray

### 6.2 验证方案仍偏 sandbox

当前验证方式仍是：

- 按排序后的样本表做时间顺序切分

这适合 baseline 阶段，但后续若要更接近真实比赛设置，还需要考虑：

- 更严格的时间切分
- 更系统的 cross-validation


## 7. 当前代码状态

当前核心脚本：

- `src\build_optiver_sandbox.py`
- `src\build_optiver_features_v2.py`
- `src\build_optiver_knn_features.py`
- `src\train_optiver_baseline.py`
- `src\run_optiver_pipeline.py`

其中已完成的重要改动包括：

- sandbox 抽样逻辑已扩展到更大规模
- 训练脚本已支持：
  - `ridge`
  - `random_forest`
  - `lightgbm`
  - `mlp`
  - `mlp_lightgbm_ensemble`


## 8. 下一步建议

### 优先级最高

在当前 `16 x 160` 大样本上，补齐同尺度三组对照：

1. `RandomForest + KNN`
2. `LightGBM + KNN`
3. `MLP + LightGBM + KNN`

目的：

- 明确当前真正最强的 baseline 是谁
- 判断集成是否值得保留

### 第二优先级

修掉 feature-name warning，保持训练日志整洁。

### 第三优先级

如果当前结果稳定，可以继续扩样本，例如：

- `24 x 200`
- 或更高

重点观察：

- KNN 是否继续增益
- MLP 集成是否随样本增大而更有价值


## 9. 一句话总结

当前阶段最重要的突破不是“换了更复杂的模型”，而是：

- 把样本规模从 `640` 扩到了 `2560`
- 让 KNN 特征真正有了历史信息基础
- 让 `MLP` 从失控组件变成了可用辅助分支
- 在更可信的样本规模上，把 `valid R²` 推到了 `0.82+`

这说明当前项目已经从“跑通 baseline”进入了“开始形成有效策略”的阶段。


## 10. CNN 引入尝试与阶段结论

### 10.1 为什么要引入 CNN

当前 `LightGBM / MLP / KNN` 这条线主要使用的是样本级聚合特征，也就是：

- 把 600 秒窗口内的 `book / trade` 明细压缩成均值、标准差、sum、波动率、窗口统计等标量特征

这种方式在表格模型上表现很好，但也天然丢掉了一部分“局部时序结构”：

- 某几秒内盘口突然变薄
- 某几秒内买卖盘失衡快速扩大
- 某几秒内成交密度集中爆发
- 局部微冲击在 10 至 30 秒范围内累积

引入 `1D CNN` 的动机就是：

- 不只看整段 600 秒的统计摘要
- 还要让模型直接学习“逐秒序列中的局部模式”

因此，CNN 的切入点不是替代表格特征，而是补充：

- 表格模型擅长样本级统计
- CNN 擅长局部时序结构


### 10.2 第一版 CNN 方案

最初设计的是 `book + trade` 双路 CNN：

1. `book` 分支输入

- `wap1`
- `wap2`
- `spread1`
- `spread2`
- `bid_size1`
- `ask_size1`

2. `trade` 分支输入

- `price`
- `size`
- `order_count`

每个样本 `(stock_id, time_id)` 被展开成一个固定长度张量：

- 时间长度：`600`
- `book` 张量形状：`[channels_book, 600]`
- `trade` 张量形状：`[channels_trade, 600]`

然后：

- `book` 一路卷积
- `trade` 一路卷积
- 两路表示按权重融合
- 最后接回归头预测未来波动率


### 10.3 第一版为什么效果不佳

最初的双路 CNN 结果非常差，甚至出现：

- `train R²` 大幅为负
- `valid R²` 也明显偏低

这里要分成“代码问题”和“建模问题”两层看。

#### 10.3.1 代码层问题

最早一版存在训练指标统计 bug：

- `train_loader` 使用了 `shuffle=True`
- 但评估时直接把打乱顺序后的训练预测，与原始顺序 `y_train` 对齐
- 造成 `train` 指标出现假性崩坏

这个问题修复后，`train` / `valid` 指标才恢复可解释性。

#### 10.3.2 建模层问题

更重要的是，最初版本在建模上过于原始，主要有以下缺陷：

1. **通道尺度不统一**

- `wap / spread` 数值尺度较小
- `size / order_count` 尺度明显更大
- CNN 直接吃未经标准化的张量，优化会严重失衡

2. **目标值未标准化**

Optiver 的目标波动率数值通常很小，直接做原始回归会让神经网络的优化更困难。

3. **trade 序列天然稀疏**

不是每一秒都有成交，因此 `trade` 张量中大量位置是“无成交秒”。

如果简单用 0 填充，就会引入语义混淆：

- 一部分 0 表示“没有成交”
- 一部分经过标准化后接近 0，则表示“正常数值”

对 CNN 而言，这会显著降低 `trade` 分支的可学习性。

4. **直接使用原始 level 特征不够平稳**

CNN 更擅长学习局部形态，而不是直接学习不同股票之间绝对价格水平的差异。

例如：

- 原始 `price` 更像 level
- `log return`、`spread change`、`imbalance change` 更像真正的动态模式

因此第一版直接吃原始 `price/size` 序列，并不是最理想的输入设计。


### 10.4 为什么高频金融数据会这样影响学习方式

Optiver 这类高频金融数据，和普通时序任务很不一样，主要体现在三点：

#### 10.4.1 强异方差

不同时间段的波动率差异很大，序列的噪声强度本身就在变化。

#### 10.4.2 稀疏与不规则

- 盘口 `book` 更连续
- 成交 `trade` 更稀疏

所以：

- `book` 更像“连续状态流”
- `trade` 更像“稀疏事件流”

这意味着两者不应该被完全等价对待。

#### 10.4.3 股票间 level 差异大，但波动预测更依赖相对变化

不同股票价格水平不同，但可预测的往往不是绝对价格本身，而是：

- 相对价差
- 相对失衡
- 短时收益率
- 成交冲击

这也是为什么：

- 表格特征里 `spread / imbalance / realized volatility` 有效
- CNN 也需要更多“变化率”而不是单纯“水平值”


### 10.5 修复后的 CNN 方案

为了解决上述问题，后续对 CNN 线做了三项关键修复：

1. **按通道标准化输入张量**

- 对 `book` 每个通道用训练集均值与标准差标准化
- 对 `trade` 每个通道也单独标准化

2. **对 target 做标准化训练**

- 训练时先标准化目标
- 预测后再映射回原始波动率尺度

3. **拆分为三种模式**

- `book_only`
- `trade_only`
- `dual`

这样可以把两个分支的贡献拆开看，而不是一开始就把问题混在一起。


### 10.6 修复后 CNN 的实验现象

修复后得到了非常重要的三组对照：

1. `book_only`

- `valid R² ≈ 0.7901`

2. `dual (book_weight = 0.7, trade_weight = 0.3)`

- `valid R² ≈ 0.8221`

3. `trade_only`

- `valid R² ≈ 0.2068`

这组结果的含义非常明确。

#### 10.6.1 book 是 CNN 的主信息源

`book_only` 已经能做到接近 `0.79`，说明盘口逐秒结构本身包含大量可预测信息。

这符合高频市场微观结构的常识：

- 盘口厚度
- 买卖盘失衡
- 点差变化
- 微观流动性恶化

这些都会提前反映未来短时间内的波动风险。

#### 10.6.2 trade 单独很弱，但作为辅助有效

`trade_only` 很差，说明成交流单独建模时存在明显问题：

- 稀疏
- 噪声大
- 零填充语义脏
- 秒级聚合后丢失细粒度信息

但 `dual` 又明显优于 `book_only`，说明：

- `trade` 虽然不能单独扛起预测
- 却能提供 `book` 没有的补充信号

因此：

- `trade` 不适合单独做主分支
- 更适合作为 `book` 的辅助支路


### 10.7 为什么这些特征让波动率可预测

这里要回到波动率本身的数学定义。

#### 10.7.1 对数收益率

设价格序列为 `P_t`，则对数收益率定义为：

\[
r_t = \log(P_t) - \log(P_{t-1})
\]

对数收益率的好处是：

- 更适合刻画相对变化
- 更接近可加结构
- 跨股票尺度更可比

#### 10.7.2 已实现波动率

在 Optiver 任务中，一个核心目标就是预测未来窗口的已实现波动率，其经典形式为：

\[
RV = \sqrt{\sum_{t=1}^{T} r_t^2}
\]

这里：

- `r_t` 是窗口内逐步对数收益率
- `RV` 越大，说明未来窗口内价格路径越不平稳

因此，任何能提前揭示“未来收益率平方和会变大”的信息，都能提升预测效果。

#### 10.7.3 WAP 与微观价格结构

在盘口数据中，常用加权平均价格：

\[
WAP_1 = \frac{bid\_price_1 \cdot ask\_size_1 + ask\_price_1 \cdot bid\_size_1}{bid\_size_1 + ask\_size_1}
\]

\[
WAP_2 = \frac{bid\_price_2 \cdot ask\_size_2 + ask\_price_2 \cdot bid\_size_2}{bid\_size_2 + ask\_size_2}
\]

它们比单纯 mid price 更能体现：

- 最优买卖价
- 挂单量结构
- 实际成交可能落点

基于 `WAP` 的短期变化，本质上就是在捕捉盘口对未来微观价格波动的预警能力。

#### 10.7.4 点差

最基本的盘口摩擦指标是：

\[
spread_1 = ask\_price_1 - bid\_price_1
\]

当点差扩大时，通常意味着：

- 流动性下降
- 市场不确定性上升
- 未来短期价格更容易跳动

因此 `spread` 与未来波动率通常正相关。

#### 10.7.5 盘口失衡

常用一阶盘口失衡定义为：

\[
imbalance_1 = \frac{bid\_size_1 - ask\_size_1}{bid\_size_1 + ask\_size_1}
\]

它刻画的是：

- 买盘更强还是卖盘更强
- 市场供需是否偏向一侧

当失衡持续存在并快速变化时，未来更容易引发价格冲击和波动放大。

#### 10.7.6 成交冲击

虽然 `trade` 支路单独较弱，但成交本身仍然重要，因为真实成交代表“市场已经发生了动作”。

例如：

- 大成交量
- 高频小单密集出现
- 单秒订单数激增

都可能意味着：

- 信息冲击
- 交易拥挤
- 价格发现加速

这就是为什么 `trade` 分支在 `dual` 结构里仍然能带来边际增益。


### 10.8 当前 CNN 线的阶段结论

当前 CNN 线已经可以下出比较清晰的判断：

1. CNN 不是失败路线

- 修复后 `book_only` 和 `dual` 都已经有有效预测能力

2. CNN 目前还没有超过当前最优表格集成线

当前更强的仍然是：

- `MLP + LightGBM + KNN (+ bagging)`

3. CNN 更适合作为补充建模分支

- 表格模型擅长样本级统计与 KNN
- CNN 擅长逐秒局部模式

因此后续最有价值的方向是：

- 把 `dual CNN` 作为新的基模型
- 再与当前最优表格方案做晚期集成或 stacking


### 10.9 下一步的 CNN 改进方向

如果继续深挖 CNN，本项目最值得尝试的不是盲目加深网络，而是：

1. 对输入通道做进一步“变化率化”

例如引入：

- `log return`
- `spread change`
- `size imbalance change`

而不是只保留原始水平值。

2. 给 `trade` 引入显式 mask

区分：

- 无成交秒
- 有成交但数值接近 0

3. 尝试更精细的双路融合

- 不是固定 `book_weight / trade_weight`
- 而是学习型门控融合

4. 在 CNN 单模稳定后，再与当前最优表格模型做集成

这是最自然、也最有可能继续提升的方向。
