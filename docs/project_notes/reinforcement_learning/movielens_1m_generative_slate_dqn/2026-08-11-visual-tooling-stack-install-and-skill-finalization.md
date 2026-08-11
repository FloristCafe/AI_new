# 2026-08-11 可视化工具栈安装、配置固化与 Skill 定稿笔记

## 今天完成了什么

今天的重点不是继续改 `Slate-DQN` 算法，而是把后续项目要长期复用的工程可视化工具栈正式落地。最终完成了四件事：

1. 在 `kg_env` 中安装了：
   - `kedro`
   - `kedro-viz`
   - `clearml`
   - `streamlit`
   - `netron`
   - `onnx`
2. 跑通了 `ClearML` 的真实配置与连通性验证。
3. 完成了 `Streamlit` 首次初始化，并确认本地服务可正常启动。
4. 新建并固化了一个可复用的 Codex skill：
   - `C:\Users\lenovo\.codex\skills\ml-pipeline-visual-stack`

这意味着从今天开始，后面的量化/RL/推荐系统项目不再只是“会训练脚本”，而是具备了：

- pipeline 拓扑可视化
- 实验追踪
- 模型结构可视化
- Web 仪表盘展示

## 这次安装阶段的最终结论

### 1. Kedro / Kedro-Viz 已安装可用

已经验证：

- `kedro --version`

后续真正开始使用时，关键命令是：

- `kedro new`
- `kedro run`
- `kedro viz`

其中：

- `kedro new` 用于新建工程骨架
- `kedro run` 用于执行 pipeline
- `kedro viz` 用于查看 DAG 拓扑图

### 2. ClearML 已安装且配置成功

这次踩到的关键事实是：

- `clearml-init` 能验证凭证
- 但在这台机器上会报 `Could not read default configuration file`

所以最终采用的稳定方案不是继续卡在向导里，而是：

- 手工创建 `C:\Users\lenovo\clearml.conf`
- 使用该配置文件完成持久化接入

验证命令已经成功跑通：

```cmd
python -c "from clearml import Task; Task.init(project_name='setup_test', task_name='hello_clearml_conf_persist'); print('clearml conf persist ok')"
```

这里出现的 warning：

- `Failed auto-detecting task repository: ... <string>`

不是配置错误，只是因为验证命令是 `python -c` 启动的，不是真实项目脚本。后面从 `.py` 文件启动训练脚本时，一般不会有这个问题。

### 3. Streamlit 已安装且首次初始化完成

`streamlit hello` 已正常启动，并弹出：

- `Local URL: http://localhost:8501`

这说明：

- Streamlit 服务可正常启动
- 首次 onboarding 已完成

关于邮箱输入：

- 这不是每次都要填
- 通常只发生在第一次初始化

如果以后想关闭 usage stats，可配置：

- `C:\Users\lenovo\.streamlit\config.toml`

写入：

```toml
[browser]
gatherUsageStats = false
```

### 4. Netron 与 ONNX 已安装可用

已经验证：

- `netron --version`
- `netron --help`
- `python -c "import onnx; print(onnx.__version__)"`

后续使用时的典型流程是：

1. 先导出模型为 `onnx`
2. 再用 `netron` 打开该文件

## 这次最重要的配置经验

### 1. Windows 终端要先分清 `cmd` 还是 PowerShell

这次有一个很典型的坑：

- 终端提示符是 `D:\Python\Miniconda>`
- 这说明当时所在的是 `cmd`
- 因此 `$env:VAR=...` 会直接报错

所以以后要记住：

- 在 `cmd` 里设置环境变量要用：`set VAR=value`
- 在 PowerShell 里设置环境变量要用：`$env:VAR="value"`

不要混用。

### 2. ClearML 的稳定做法优先级

在这台机器上的推荐顺序应写死为：

1. 先安装 `clearml`
2. 可以先试一次 `clearml-init`
3. 如果向导验证成功但配置文件写入失败
4. 立刻切换到手工建：
   - `C:\Users\lenovo\clearml.conf`
5. 再用 `python -c "from clearml import Task; ..."` 做验证

也就是说，对这台机器而言：

- `clearml.conf` 是稳定主路径
- `clearml-init` 只是可选尝试，不是必须依赖

### 3. `streamlit hello` 属于安装后的烟雾测试

它不是正式项目命令，但非常适合作为：

- 是否安装成功
- 浏览器服务是否拉起
- 首次配置是否完成

的快速检查项。

## 今天对 skill 做了什么改动

今天新建并定稿了 skill：

- [SKILL.md](C:/Users/lenovo/.codex/skills/ml-pipeline-visual-stack/SKILL.md)
- [stack-playbook.md](C:/Users/lenovo/.codex/skills/ml-pipeline-visual-stack/references/stack-playbook.md)
- [openai.yaml](C:/Users/lenovo/.codex/skills/ml-pipeline-visual-stack/agents/openai.yaml)

### 1. Skill 的职责被正式固定

它现在明确要求未来的 Codex 在下列场景主动接入这套工具：

- 项目结构设计
- 多阶段数据/模型 pipeline 落地
- 实验追踪
- 模型结构检查
- Web 展示与答辩页搭建

### 2. Skill 中已经写入了常用启动命令

包括：

- Kedro:
  - `kedro new`
  - `kedro run`
  - `kedro viz`
- Netron:
  - `netron "D:\path\to\model.onnx"`
- Streamlit:
  - `streamlit run "D:\path\to\app.py"`

### 3. Skill 中补入了这次真实踩坑后的配置规则

重点已经写回 `stack-playbook.md`：

- `ClearML` 推荐手工 `clearml.conf`
- `cmd` 与 PowerShell 的环境变量写法区别
- `Streamlit` 首次邮箱提示属于正常 onboarding
- 安装完成后的 bring-up checklist

这一步很重要，因为 skill 不再只是“工具名单”，而是已经吸收了这台机器的真实环境经验。

## 后面这四个工具分别怎么调用

### 1. Kedro / Kedro-Viz

使用场景：

- 项目已经不是单脚本
- 至少有三段以上清晰阶段
- 例如：预处理、特征工程、训练、评估、报告

后续调用方式：

```cmd
kedro new
kedro run
kedro viz
```

在未来项目中的角色：

- `Kedro` 负责组织 pipeline
- `Kedro-Viz` 负责把 pipeline 拓扑图可视化出来

### 2. ClearML

使用场景：

- 需要记录实验参数
- 需要看 loss/metric 曲线
- 需要比较不同实验
- 需要上传 checkpoint、metrics、图表

后续调用方式分两层：

第一层是配置层：

- 读取 `C:\Users\lenovo\clearml.conf`

第二层是项目代码层：

```python
from clearml import Task

task = Task.init(
    project_name="reinforcement_learning",
    task_name="your_project_name",
)
task.connect(args)
logger = task.get_logger()
```

在未来项目中的角色：

- ClearML 是实验账本与曲线平台

### 3. Netron

使用场景：

- 需要检查模型输入输出 shape
- 需要确认网络层次结构
- 需要在答辩或汇报中直观展示模型图

后续调用方式：

```cmd
netron "D:\path\to\model.onnx"
```

推荐流程：

1. 先从训练好的模型导出 `.onnx`
2. 再用 `netron` 打开

在未来项目中的角色：

- Netron 是模型显微镜

### 4. Streamlit

使用场景：

- 需要把实验结果做成可交互页面
- 需要给导师、面试官、自己做结果浏览器
- 需要展示指标、曲线、回测、推荐结果或策略对比

后续调用方式：

```cmd
streamlit run "D:\path\to\app.py"
```

推荐原则：

- 第一版只读 `artifacts/`
- 不让 Streamlit 页面重新训练模型
- 专注做结果浏览、对比、图表展示

在未来项目中的角色：

- Streamlit 是演示层与项目展示层

## 对下一个项目的直接影响

从现在开始，下一类真实项目应按下面这个顺序推进：

1. 先确定项目在 `projects/` 还是 `tracks/`
2. 用常规 `src/ + artifacts/` 方式先把训练与评估脚本跑通
3. 当阶段数变多时，引入 `Kedro`
4. 从第一版训练脚本开始接 `ClearML`
5. 当模型结构复杂或要做汇报时，导出到 `Netron`
6. 当实验结果稳定后，用 `Streamlit` 做展示页

这比一开始就堆很多工具更稳，因为它遵循了：

- 先可运行
- 再可追踪
- 再可展示

## 一句话总结

今天真正完成的，不只是“把几个 Python 包装进环境”，而是把后续量化/RL/推荐项目要长期复用的可视化工程工具链正式固定下来了：`Kedro/Kedro-Viz` 负责 pipeline 拓扑，`ClearML` 负责实验账本，`Netron` 负责模型显微镜，`Streamlit` 负责展示层；同时这些经验已经沉淀进 `ml-pipeline-visual-stack` skill，后续新对话也可以直接复用。
