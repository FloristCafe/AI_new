# 空间设计说明

## 为什么原结构不够稳

旧结构：

- `data_pipeline`
- `model`
- `training`
- `tensor_math`
- `projects`

问题不在“能不能用”，而在“越做越乱”：

- 它只表达了能力分层，没有表达方向分层
- `data_pipeline` 很容易沦为所有数据相关脚本的回收站
- 单个项目的代码会被拆散在多个顶层目录里
- 以后加入信息检索时，NLP、图结构、稀疏检索、重排模型都很难定位

## 新结构

### `docs`

放路线图、仓库说明、命名规则。

### `shared`

放跨方向复用的能力，不关心具体赛道。

当前子目录：

- `shared/data`
- `shared/modeling`
- `shared/training`
- `shared/math`

### `tracks`

放某个方向内部的探索代码与原型。

当前子目录：

- `tracks/recommendation`
- `tracks/information_retrieval`
- `tracks/reinforcement_learning`

### `projects`

放真正要推进、要交付、要对应笔记系统的项目。

当前预留：

- `projects/recommendation`
- `projects/information_retrieval`
- `projects/reinforcement_learning`
- `projects/incubator`

### `sandbox`

放短命脚本和环境检查。

## 如何决定代码放哪

判断顺序：

1. 这是一次性测试吗？
   是：放 `sandbox`
2. 这是某个明确项目的代码吗？
   是：放 `projects/<track>/<project_name>`
3. 这是某个方向的探索原型，但还不是项目吗？
   是：放 `tracks/<track>`
4. 这是跨方向都能复用的通用组件吗？
   是：放 `shared`

## 对你当前主线的映射

推荐系统：

- 方向探索放 `tracks/recommendation`
- 四个项目放 `projects/recommendation`

信息检索：

- 当前 NLP/知识图谱脚本先归入 `tracks/information_retrieval/knowledge_graph`

强化学习：

- 通用 RL 探索放 `tracks/reinforcement_learning`
- 推荐系统里的 MDP 环境项目仍放 `projects/recommendation`

这样做的好处是：RL 既可以作为独立方向存在，也可以成为推荐系统项目的一部分，不冲突。
