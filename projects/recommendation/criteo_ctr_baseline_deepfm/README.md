# Criteo CTR Baseline DeepFM

这是 `criteo_ctr_baseline_onehot` 的下一阶段项目。

目标不是直接追求高分，而是把 Criteo 微缩样本接入一个最小可理解的 DeepFM 架构，让我们真正看清：

- sparse id 特征为什么要用 embedding
- FM 部分在建模什么
- deep 部分在建模什么
- 一个样本是如何从 parquet 表格，走到 DeepFM 的 forward 计算里的

## 项目定位

这个项目仍然是“教学型 baseline”，但它比 one-hot LR 更接近 CTR 主干路线。

当前版本重点：

- 继续使用微缩沙盒样本
- 延续昨天的 Criteo 数据提取方式
- 将 dense / sparse 特征整理成 DeepFM 友好的输入格式
- 用一个最小 PyTorch DeepFM 模型完成训练与验证

## 目录说明

- `src/preprocess_criteo_deepfm.py`
  - 读取微缩 parquet
  - 切分 train / valid
  - 训练集学习类别映射
  - 生成 DeepFM 训练所需的特征配置与 parquet

- `src/deepfm_model.py`
  - DeepFM 模型定义
  - 包含 linear、FM、deep 三部分

- `src/train_deepfm.py`
  - 训练循环
  - 验证指标
  - 保存模型和实验结果

## DeepFM 与 one-hot LR 的关键区别

在 one-hot LR 中：

- 一个类别值会被展开成很多 0/1 维度中的某一位
- 线性模型主要学习每个维度各自的加性贡献

在 DeepFM 中：

- 一个类别值先映射成一个整数 id
- 然后通过 embedding table 查出一个低维稠密向量
- FM 部分用这些向量建模二阶交互
- deep 部分把这些向量拼接起来，再学更复杂的非线性模式

所以 DeepFM 更适合：

- 高基数类别特征
- 稀疏离散 id
- 特征交互建模

## 当前数据来源

默认仍使用：

- `D:\Python\Datasets\criteo_display_ad_challenge\samples\criteo_micro_2000.parquet`

也就是昨天微缩沙盒里切出来的 2000 条样本。

## 使用顺序

1. 先跑预处理
2. 再跑训练
3. 观察 embedding / FM / deep 的效果与局限

## 当前原则

这个项目首先是“学懂 CTR 表征与建模方式”的项目，不是立刻追求复杂工程堆叠的项目。
