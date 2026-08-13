# Criteo CTR Baseline DeepFM

这是 `criteo_ctr_baseline_onehot` 的下一阶段项目。
目标不是直接追求高分，而是把 Criteo 微缩样本接入一个最小可理解的 `DeepFM` 架构，并补齐实验追踪、模型可视化、结果展示和轻量 pipeline 编排能力。

当前这版项目已经包含：

- `DeepFM` 预处理与训练主链路
- `ClearML` 实验追踪接入
- `ONNX + Netron` 模型结构导出与查看
- `Streamlit` 本地结果仪表盘
- `Kedro / Kedro-Viz` 轻量 pipeline 外壳

## 项目定位

这个项目仍然是“教学型 baseline”，但它比 one-hot LR 更接近 CTR 主干路线。

当前版本重点：

- 继续使用 Criteo 微缩样本
- 保留单文件可直接跑通的训练风格
- 把 `dense / sparse` 特征整理成 `DeepFM` 友好的输入格式
- 在不推翻原脚本结构的前提下，接入可视化和实验工程工具

## 目录说明

- `src/preprocess_criteo_deepfm.py`
  - 读取微缩 parquet
  - 切分 train / valid
  - 学习类别映射
  - 生成 DeepFM 所需的 parquet 与 feature config
- `src/deepfm_model.py`
  - DeepFM 模型定义
  - 包含 linear、FM、deep 三部分
- `src/train_deepfm.py`
  - 训练循环
  - 验证指标
  - 保存模型、预测、metrics
  - 记录 ClearML 实验
- `src/export_deepfm_onnx.py`
  - 导出 `.onnx` 供 Netron 查看
- `src/app_deepfm_dashboard.py`
  - 读取 `artifacts/` 的 Streamlit 仪表盘
- `src/criteo_ctr_baseline_deepfm_kedro/`
  - Kedro pipeline 注册与节点包装
- `conf/base/catalog.yml`
  - Kedro 最小数据目录配置

## 数据来源

默认使用：

- `D:\Python\Datasets\criteo_display_ad_challenge\samples\criteo_micro_2000.parquet`

## 最小运行顺序

### 1. 预处理

```cmd
python "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\src\preprocess_criteo_deepfm.py"
```

### 2. 训练 DeepFM

```cmd
python "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\src\train_deepfm.py"
```

默认会在本地保存：

- `artifacts\deepfm_run\deepfm_model.pt`
- `artifacts\deepfm_run\deepfm_model_best.pt`
- `artifacts\deepfm_run\metrics.json`
- `artifacts\deepfm_run\valid_predictions.parquet`

如果本机 `ClearML` 已完成 `clearml.conf` 配置，上面这条命令会自动把参数、曲线和关键 artifact 上传到 ClearML。

如果想临时关闭 `ClearML`：

```cmd
python "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\src\train_deepfm.py" --disable-clearml
```

## 工具接入方式

### ClearML

这个项目里的 `ClearML` 集成点在：

- `src/train_deepfm.py`

默认任务名：

- project: `recommendation`
- task: `criteo_ctr_baseline_deepfm`

如果需要自定义：

```cmd
python "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\src\train_deepfm.py" ^
  --clearml-project-name recommendation ^
  --clearml-task-name criteo_ctr_deepfm_ablation_fm_off
```

### Netron

先导出 ONNX：

```cmd
python "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\src\export_deepfm_onnx.py"
```

再用 Netron 打开：

```cmd
netron "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\artifacts\deepfm_run\deepfm_model_best.onnx"
```

### Streamlit

启动本地仪表盘：

```cmd
streamlit run "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm\src\app_deepfm_dashboard.py"
```

仪表盘当前会展示：

- 最优与最终验证指标
- 训练曲线
- 训练配置
- 预处理摘要
- 特征布局与词表规模
- 验证集预测样本快照

### Kedro / Kedro-Viz

这个项目没有强行迁移成重型 Kedro 工程，而是保留了“单脚本可直跑”的主风格，并额外补了一层最小 Kedro 工程壳。

先切到项目目录：

```cmd
cd /d "D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm"
```

运行默认 pipeline：

```cmd
kedro run
```

查看 pipeline 拓扑：

```cmd
kedro viz
```

如果你只想用脚本方式跑，也可以继续直接用 `python src\...py`，两种方式并不冲突。

## DeepFM 与 one-hot LR 的关键区别

在 one-hot LR 中：

- 一个类别值会被展开成高维 0/1 向量
- 线性模型主要学习每一维各自的加性贡献

在 DeepFM 中：

- 类别值先映射为整数 id
- 再通过 embedding table 取出低维稠密向量
- FM 分支建模二阶交互
- deep 分支学习更复杂的非线性组合

所以 `DeepFM` 更适合：

- 高基数类别特征
- 稀疏离散 id
- 特征交互建模

## 当前原则

这个项目首先仍然是“学懂 CTR 表征与建模方式”的项目，不是为了堆很多工具而堆工具。
这次工具接入的目标是：

- 保留原本清晰的脚本主链路
- 让实验更可追踪
- 让模型结构更可视
- 让结果更容易展示
- 让 pipeline 拓扑更容易讲清楚
