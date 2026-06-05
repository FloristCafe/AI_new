# Artificial Intelligence Workspace

这个仓库现在不再是“按零散脚本堆积”的练习区，而是一个支持多方向并存的 AI 实践工作区。

当前设计目标：

- 推荐系统可以作为主线长期推进
- 信息检索可以自然加入，不和推荐系统互相污染
- 强化学习可以作为独立方向，也可以作为推荐系统的后续延伸
- 共享代码与方向实验分离
- 真正的项目落地和临时练习分离

## 顶层结构

- `docs/`：路线图、目录说明、工作规则
- `shared/`：跨方向复用的通用能力
- `tracks/`：按研究/工程方向组织的探索代码
- `projects/`：真正的项目落地区
- `sandbox/`：一次性脚本、环境检查、短平快试验

## 设计原则

### 1. `shared` 只放可复用能力

例如：

- 通用数据集封装
- 通用训练循环
- 通用特征处理组件
- 通用数学/矩阵工具

如果一段代码明显只服务于某个方向，不要放进 `shared`。

### 2. `tracks` 负责方向探索

当前预留：

- `tracks/recommendation`
- `tracks/information_retrieval`
- `tracks/reinforcement_learning`

以后还可以继续扩展：

- `tracks/recsys_rl`
- `tracks/ranking`
- `tracks/multimodal`

### 3. `projects` 只放项目

项目应该满足：

- 有清晰目标
- 有明确交付
- 能对应 Obsidian 项目卡
- 有实验记录和复盘价值

### 4. `sandbox` 不沉淀长期结构

任何环境测试、一次性练习、验证 GPU/包安装的小脚本，都丢进 `sandbox`。

## 当前判断

原来的 `data_pipeline / model / training / tensor_math` 顶层拆法不够适合作为长期主仓库，因为它按“能力层”拆分，但没有表达“方向层”和“项目层”。当推荐系统、信息检索、RL 同时存在时，会不断出现代码归属混乱。

现在这版结构改成：

- 共享能力层：`shared`
- 方向探索层：`tracks`
- 项目落地层：`projects`

这样推荐系统只是一个重点方向，而不是整个仓库唯一合法主题。
