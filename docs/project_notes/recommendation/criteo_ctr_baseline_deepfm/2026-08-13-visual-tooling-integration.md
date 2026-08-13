# 2026-08-13 Criteo DeepFM 可视化工具链接入笔记

## 今天做了什么

今天没有继续改 `DeepFM` 模型本身，而是把已经落地的工具栈正式接入：

- `ClearML`
- `Netron`
- `Streamlit`
- `Kedro / Kedro-Viz`

接入对象是：

- `D:\Python\Artificial Intelligence\projects\recommendation\criteo_ctr_baseline_deepfm`

这次改造的原则不是“把原项目推倒重来”，而是：

- 保留你熟悉的 `README.md + src/ + artifacts/`
- 保留“单文件可直接跑通”的脚本风格
- 在此基础上补实验追踪、模型可视化、结果展示与轻量 pipeline 外壳

## 为什么这个项目适合接工具

这个项目现有主链路已经天然分成了三个阶段：

1. `preprocess`
2. `train`
3. `artifacts`

这使它非常适合作为第一批真正接入工程可视化工具的推荐系统项目。

相比于还在快速试错的原型，这个项目已经具备：

- 明确的输入数据
- 明确的预处理产物
- 明确的训练脚本
- 明确的评估指标
- 明确的模型文件与输出目录

因此，接工具不会显得“空转”。

## 今天接入的具体内容

### 1. ClearML：把训练脚本变成可追踪实验

在：

- `src/train_deepfm.py`

中加入了：

- `Task.init(...)`
- 参数自动记录
- 每个 epoch 的 train / valid 指标上报
- `metrics.json`
- `deepfm_model.pt`
- `deepfm_model_best.pt`
- `valid_predictions.parquet`

等 artifact 上传

这意味着以后这个项目不再只是本地 `metrics.json`，而是可以在 ClearML 上看到：

- 学习率、dropout、embedding_dim 等参数
- ROC-AUC / PR-AUC / LogLoss 曲线
- 不同 ablation run 的横向比较

### 2. Netron：把 DeepFM 结构导出为 ONNX

新增：

- `src/export_deepfm_onnx.py`

用途是：

- 读取 best checkpoint
- 自动从 `metrics.json` 和 `feature_config.json` 推断结构超参
- 导出 `deepfm_model_best.onnx`

然后可以用：

- `netron "...deepfm_model_best.onnx"`

直观看：

- dense 输入
- dense bucket embedding
- sparse embedding
- FM 分支
- deep 分支
- 最终输出

这对于之后讲项目、答辩或者检查 shape 都很有用。

### 3. Streamlit：给项目加一个只读展示层

新增：

- `src/app_deepfm_dashboard.py`

这个页面不负责重新训练，而是只读取本地 `artifacts/`：

- `metrics.json`
- `preprocess_summary.json`
- `feature_config.json`
- `valid_predictions.parquet`

当前能展示：

- 最优与最终验证指标
- 训练曲线
- 训练配置
- 特征统计
- 词表规模
- 验证预测快照

也就是说，这个项目已经不只是“脚本 + json”，而是可以被做成一页可交互结果浏览器。

### 4. Kedro / Kedro-Viz：补一层最小 pipeline 外壳

这次没有把项目彻底重写成 Kedro 风格，而是做了更适合当前阶段的做法：

- 保留原脚本主链路
- 用最小 Kedro 工程壳包装 `preprocess -> train -> export`

新增内容包括：

- `pyproject.toml`
- `conf/base/catalog.yml`
- `src/criteo_ctr_baseline_deepfm_kedro/...`

这样以后可以直接在项目目录里运行：

- `kedro run`
- `kedro viz`

好处是：

- 依然保留 `python src\train_deepfm.py` 的简单性
- 同时获得 pipeline DAG 可视化能力

## 这次改造后，项目的运行入口发生了什么变化

原来主要只有两条：

1. 预处理
2. 训练

现在变成了四层入口：

### 1. 脚本入口

适合日常快速实验：

- `python src\preprocess_criteo_deepfm.py`
- `python src\train_deepfm.py`

### 2. 实验追踪入口

适合正式跑实验：

- 直接跑 `train_deepfm.py`
- 自动接入 `ClearML`

### 3. 模型检查入口

适合看结构：

- `python src\export_deepfm_onnx.py`
- `netron "...onnx"`

### 4. 展示与拓扑入口

适合讲项目：

- `streamlit run src\app_deepfm_dashboard.py`
- `kedro viz`

## 这次改造最重要的工程判断

今天最重要的判断不是“能不能把四个工具都塞进来”，而是：

**如何在不破坏原项目可运行性的前提下，让工具真正服务于项目。**

最终采用的策略是：

- `ClearML` 只接在训练主入口
- `Netron` 通过单独导出脚本接入
- `Streamlit` 只做 artifacts 读取层
- `Kedro` 只加最小外壳，不强制改写已有训练脚本

这比直接“全量 Kedro 化”更符合当前项目阶段，也更符合你一贯的脚本风格。

## 对这个项目价值的提升

接完工具后，这个 `DeepFM` 项目的价值有了明显变化：

原来它更像：

- 一个教学型 CTR baseline

现在它更像：

- 一个具备实验追踪、模型可视化、结果展示和 pipeline 讲解能力的完整推荐工程样板

这对后面做两类事情都很有帮助：

1. 项目迁移  
   以后别的推荐系统项目可以照这个模式快速复制

2. 对外展示  
   简历、面试、答辩时，这个项目更容易讲成“工程化推荐实验系统”，而不只是“写了个 DeepFM”

## 一句话总结

今天这轮改造的意义，不是单纯把 `Kedro / ClearML / Netron / Streamlit` 四个工具挂到 `criteo_ctr_baseline_deepfm` 上，而是把这个原本偏教学型的 CTR baseline，升级成了一个兼具脚本可运行性、实验可追踪性、结构可视化能力和结果展示能力的推荐工程样板。

## 补充验收结论

在后续真实验收中，这个项目的四条工具链都已经跑通：

- `ClearML`：训练脚本已成功创建 task、记录曲线并复用本地模型 artifact
- `Netron`：ONNX 导出成功，且本地 `localhost` 服务可正常启动
- `Streamlit`：结果面板已成功启动，说明展示层已具备可运行性
- `Kedro / Kedro-Viz`：`kedro run` 与 `kedro viz` 已成功运行

其中最重要的一次修正不是“让 Kedro 能跑”，而是把 Kedro DAG 从表面可用修成语义正确：

- 初始版本虽然能跑，但 `export`、`preprocess`、`train` 三个节点没有真实数据依赖
- 后续已修正为：
  - `preprocess_outputs -> training_outputs -> onnx_export_outputs`
- 最终 `kedro run` 日志顺序已被依赖链锁定为：
  1. `preprocess`
  2. `train`
  3. `export`

这次验收沉淀出的真正方法论是：

- 保留原始 `python src\...py` 脚本作为主入口
- 将 `ClearML`、`Netron`、`Streamlit` 作为外挂能力接在脚本外层
- 当项目阶段足够清晰后，再增加一个最小 Kedro 工程壳
- 但 Kedro 不是“能跑就行”，而是必须保证 DAG 依赖真实成立

也就是说，这次项目最终验证通过的不是“工具装上了”，而是：

**脚本优先、工具外挂、最小 Kedro 壳、真实 DAG 依赖** 这套模式已经在一个真实推荐项目上成功闭环。
