# Criteo CTR DeepFM 阶段收束笔记

## 项目定位
这个项目不是为了立刻做出比 one-hot LR 更强的线上可用模型，而是为了在推荐/CTR 主线上，系统理解以下问题：

- DeepFM 的输入到底长什么样
- sparse embedding 是怎么被查表和训练的
- FM、deep、linear 三个分支分别学什么
- 数值稳定性、特征处理、正则化和数据规模会怎样影响 CTR 模型训练

当前阶段的目标已经完成：我们已经把这个项目从“结构联调和数值排障”推进到了“可以做有根据的结构判断”。

## 当前正式对照基线
当前项目里，真正成熟且成绩更好的基线仍然是 one-hot LR。

- one-hot LR baseline
- 数据规模：200000
- ROC-AUC：0.726962
- PR-AUC：0.101070
- LogLoss：0.133361

这组结果说明，在当前样本量和当前特征工程下，线性 baseline 依然明显强于 DeepFM。

## DeepFM 当前版本做了什么改造
这一轮 DeepFM 已经不再是最早那个会数值爆炸的原型，而是做过一系列稳定化处理后的版本。

- dense 缺失值填充为 `-1`
- sparse 缺失值填充为 `"missing"`
- 先按 train/valid 切分，再拟合稀有类别规则和映射，避免泄漏
- sparse 稀有类别折叠为 `UNK`
- dense 标准化只在训练集拟合，再应用到训练集和验证集
- 新增 dense bucket 特征
- deep 分支改为吃 `dense bucket embedding + sparse embedding`
- deep MLP 改成 `Linear -> BatchNorm1d -> ReLU -> Dropout`
- 引入全局先验 bias，用训练集 CTR 初始化
- FM embedding 使用小尺度初始化
- FM 分支增加缩放系数 `fm_scale`
- `bias` 和 `BatchNorm` 参数不参与 `weight_decay`
- 增加 `--debug-logits`、`--one-batch-overfit`、`--disable-fm`、`--disable-deep` 等调试开关

## 本阶段最关键的工程判断
### 1. 早期 DeepFM 的主要问题不是“完全学不会”，而是“数值行为不健康”
最初排查时，模型出现过明显的 logit 爆炸和整体负向塌缩：

- logits 一度达到极端量级
- BCE 很大
- 验证指标接近随机
- 1-batch overfit 能下降，但靠的是极端 logit，而不是健康表示学习

因此项目前半段的重点不是调参，而是先把数值路径修到可解释。

### 2. dense 标准化和 dense bucket 是有效修复
这一点已经基本坐实。

- dense 标准化明显缓解了数值失控
- dense bucket 让 deep 分支不再直接硬吃原始 dense 浮点
- BatchNorm 让 deep 分支从“容易塌缩”变成“可以正常学习”

这说明在 CTR 场景里，dense 特征不应该被当作普通 tabular 浮点直接粗暴拼进 MLP。

### 3. 200000 样本比 20000 样本更有意义，但还不足以给 DeepFM 定最终结论
样本量从 20000 扩到 200000 后，DeepFM 的表现和稳定性都明显改善。

- 它已经不再是“坏掉的模型”
- 它已经能稳定学习
- 但它依然明显落后于 one-hot LR baseline

因此当前最准确的表述不是“DeepFM 不行”，而是：

- DeepFM 已经被修到可训练状态
- 但当前样本量和当前结构下，仍未超过成熟 baseline

## 当前 DeepFM 最好结果
在这一轮稳定化之后，full model 的较优点大致出现在 `Epoch 4`。

- Full DeepFM
- 数据规模：200000
- ROC-AUC：0.674131
- PR-AUC：0.074900
- LogLoss：0.168119

这说明 deep 分支已经带来了真正的增益，但离 one-hot LR 仍有明显差距。

## 四组最小消融实验结论
### 1. Full model
- ROC-AUC：0.674131
- PR-AUC：0.074900
- LogLoss：0.168119

这是当前 DeepFM 的主参考结果。

### 2. 去掉 FM 分支
- ROC-AUC：0.673265
- PR-AUC：0.074745
- LogLoss：0.168367

和 full model 几乎一样。

结论：
- 当前 FM 分支几乎没有成为主要增益来源
- FM 现在已经被压稳，但贡献非常弱

### 3. 去掉 deep 分支
- ROC-AUC：0.567902
- PR-AUC：0.040826
- LogLoss：0.234035

相比 full model 明显下降。

结论：
- deep 分支已经是真正的主力分支
- 这一轮 bucket + BatchNorm + 稳定化改造不是白做的
- DeepFM 当前最主要的有效提升来自 deep，而不是 FM

### 4. 仅对 embedding 做 L2 正则
- ROC-AUC：0.673834
- PR-AUC：0.075075
- LogLoss：0.168239

与 full model 非常接近。

结论：
- “只对 embedding 做 L2” 是合理策略
- 但它不是当前阶段的决定性提升点
- 当前主瓶颈不在正则分组，而在表示能力和数据规模

## FM 缩放实验结论
我们继续测试了 `fm_scale=0.2` 和 `fm_scale=0.3`。

### `fm_scale=0.2`
- ROC-AUC：0.674561
- PR-AUC：0.075193
- LogLoss：0.168158

相比 `0.1` 有非常轻微的改善。

### `fm_scale=0.3`
- ROC-AUC：0.673688
- PR-AUC：0.074672
- LogLoss：0.168746

没有继续提升，反而略回落。

结论：
- FM 不是简单“被压太狠了”
- 适度放大到 `0.2` 可以有轻微帮助
- 但继续放大并不会带来结构性增益
- 当前 FM 更像“稳定但很弱”，而不是“只差一个更大的系数”

## 当前结构层面的最终判断
本阶段可以形成比较稳定的判断：

1. one-hot LR 仍然是当前正式 baseline
2. DeepFM 已经完成了数值层面的“止血”
3. deep 分支已经被救活，而且是当前主要增益来源
4. FM 分支目前稳定但贡献很弱
5. 200000 样本足够做结构诊断，但不足以给 DeepFM 判最终死刑
6. 当前 DeepFM 的主要问题已经不再是“会不会炸”，而是“能否进一步释放表示能力”

## 为什么在这里暂停
当前暂停不是因为项目失败，而是因为边际收益已经明显下降。

继续在当前策略上细调，会越来越像：

- 小幅换 `fm_scale`
- 小幅换正则
- 小幅换训练细节

但这些变化已经很难带来决定性新信息。

相反，现在更值得做的是暂停后吸收外部经验，再进行下一轮更高质量策略设计。

## 暂停时应保留的核心共识
暂停节点上，应明确保留以下共识：

1. one-hot LR 是当前更强的正式基线
2. DeepFM 不是写坏了，而是已经从“坏掉”修到“可训练”
3. dense 标准化、dense bucket、BatchNorm 是这一轮最关键的有效修复
4. 当前 DeepFM 的主要有效部分是 deep，不是 FM
5. FM 现在已经稳定，但还没真正成为决定结果的力量
6. 当前瓶颈更偏向数据规模和更深层结构策略，而不再是简单数值修补

## 如果未来重启，这一轮最值得带走的问题
如果后面要重启这个项目，最有价值的切入口是这些：

- 工业界/Kaggle 上成熟 CTR 方案如何处理 dense 特征
- DeepFM 中 FM 和 deep 的协同为何在当前任务里没有被真正释放
- 更大样本量下，当前 deep 分支是否还能继续稳定提升
- 是否应该继续沿 DeepFM 深挖，还是转向更合适的 CTR 主线模型
- 当前 baseline 和成熟开源实现之间，真正的差距主要在数据、特征、结构还是训练流程

## 本阶段一句话总结
这个项目当前最重要的成果，不是“DeepFM 打赢了 baseline”，而是：

我们已经比较清楚地知道了它为什么还没打赢，以及问题主要不再出在数值稳定性，而出在更高层次的结构表达和数据规模上。
