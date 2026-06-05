# Workspace Rules

## 命名规则

- 目录名统一用小写英文加下划线
- 项目目录名尽量与 Obsidian 项目卡保持可映射
- 方向目录用稳定命名，不频繁改名

## 当前方向

- `recommendation`
- `information_retrieval`
- `reinforcement_learning`

## 当前项目映射

- `tabular-ml adversarial-validation-criteo`
  - `projects/recommendation/tabular_ml_adversarial_validation_criteo`
- `data-bedrock criteo-parquet-pipeline`
  - `projects/recommendation/data_bedrock_criteo_parquet_pipeline`
- `deepfm deepfm-baseline`
  - `projects/recommendation/deepfm_baseline`
- `rl-env recommender-mdp-gymnasium`
  - `projects/recommendation/recommender_mdp_gymnasium`

## 代表性约束

这个代码仓库本身不区分“代表项目区”和“普通项目区”，这个判断留给 Obsidian 仓库中的 `project-skill` 规则来做。

也就是说：

- 代码仓库负责承载项目
- Obsidian 仓库负责做项目价值判断和归档分层

## 以后加入信息检索时的建议

优先按任务组织：

- `dense_retrieval`
- `sparse_retrieval`
- `reranking`
- `knowledge_graph`

如果以后 IR 项目成熟，再从 `tracks/information_retrieval` 提升到 `projects/information_retrieval`。
