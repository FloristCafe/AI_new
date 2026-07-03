# Micro-RecSim MDP 与 Baseline 阶段笔记

## 笔记存放位置说明

这份笔记放在：

- `docs/project_notes/`
- `recommendation/`
- `recommender_mdp_gymnasium/`

这样放的原因是：

- 这已经不是单纯的方向探索，而是一个明确的推荐系统项目沙盒
- 当前记录的内容不是路线图，也不是代码本体，而是项目阶段判断、环境设计决策、实验结论与下一步入口
- 继续沿用 `docs/project_notes/<track>/<project>/` 的规则，可以把“代码和结果”与“为什么这样设计”分开管理

## 当前项目定位

`recommender_mdp_gymnasium` 当前不是一个训练强模型的项目，而是一个用于强化学习入门和推荐系统序列决策抽象的最小教学沙盒。

当前阶段目标不是：

- 直接上 DQN
- 追求高分
- 搭复杂工程

当前阶段目标是：

- 把一个最小推荐问题清晰抽象成 MDP
- 手写一个可运行的环境
- 用 baseline 验证这个环境的动力学逻辑是否合理
- 为后续 tabular Q-learning 做好 fully observable 教学版状态准备

## 当前项目代码落点

当前项目目录下已有两个核心文件：

- `projects/recommendation/recommender_mdp_gymnasium/micro_recsim_env.py`
  - 定义最小推荐环境
  - 实现 `__init__ / reset / step`
  - 文件底部提供随机 agent 的 sanity check

- `projects/recommendation/recommender_mdp_gymnasium/run_baselines.py`
  - 统一评估多种 baseline 策略
  - 输出平均奖励、平均 session 长度、平均点击数、最终耐心和动作分布

## 第一阶段：最小 MDP 环境落地

### 1. MDP 抽象

当前环境把推荐问题抽象为一个四元组 `(S, A, P, R)`：

- 状态 `S`
  - 用户疲劳度向量
  - 用户偏好向量
  - 用户耐心值

- 动作 `A`
  - 从 5 个类目中选择 1 个进行推荐

- 状态转移 `P`
  - 被推荐类目的疲劳度增加
  - 未被推荐类目的疲劳度衰减
  - 未点击时耐心下降

- 奖励 `R`
  - 点击 `+1`
  - 未点击 `0`
  - 用户退出 `-10`

### 2. 点击概率的行为假设

当前使用的点击概率是：

\[
p(a \mid s) = \text{true\_preference}[a] \cdot (1 - \text{fatigue}[a])
\]

这条规则不是现实真理，而是一个极简但有解释力的行为模型。它希望表达的是：

- 用户对某类目有天然偏好
- 同类目被反复推荐后会产生疲劳
- 疲劳会压低点击概率
- 但即使疲劳上升，高偏好类目仍然可能被点击

这正好对应“越看越腻，但遇到最爱也勉强点一下”的入门建模直觉。

### 3. 疲劳与耐心机制

当前环境里：

- 被推荐类目：疲劳度 `+0.2`
- 其他类目：疲劳度 `*0.9`
- 未点击：耐心 `-1`
- 耐心 `<= 0`：用户退出

这套规则成功制造了一个最核心的 RL 结构：

- 当前动作不仅影响当前 reward
- 还会影响未来点击概率和未来 session 长度

也就是说，这已经不是单步分类问题，而是一个标准序列决策问题。

## 第二阶段：从 partial observable 到教学版 fully observable

### 1. 初始版本

最开始的 observation 只有 6 维：

- 前 5 维 fatigue
- 最后 1 维 patience

而 `true_preference` 只存在于环境内部。

这意味着：

- agent 看不到决定点击概率的重要因子之一
- 环境对 agent 来说更像一个部分可观测问题

### 2. 教学版改造

为了更适合入门和后续 tabular Q-learning，环境后来被改成 fully observable 版本。

当前 observation 变为 11 维：

- 前 5 维：fatigue
- 中间 5 维：true preference
- 最后 1 维：patience

当前环境状态结构已经变成：

\[
obs = [fatigue_0,\dots,fatigue_4,\ preference_0,\dots,preference_4,\ patience]
\]

这样改的目的不是更真实，而是更教学友好：

- agent 能直接看到影响点击概率的核心变量
- 后面做 tabular Q-learning 时，不需要一开始就处理 POMDP 难点

## 第三阶段：baseline 实验框架搭建

为了把环境从“能运行”推进到“能做实验”，新增了 baseline runner。

当前已加入的策略包括：

- `random`
  - 每一步随机选动作

- `always_same_0`
  - 永远推同一个类目

- `round_robin`
  - 按 `0 -> 1 -> 2 -> 3 -> 4` 轮换推荐

- `least_fatigue`
  - 优先推当前疲劳度最低的类目

- `oracle_preference_greedy`
  - 每一步选 `true_preference * (1 - fatigue)` 最大的动作
  - 这是“偷看环境底牌”的上界 baseline

- `observable_click_greedy`
  - 每一步只用 observation 中可见的 `fatigue` 和 `preference`
  - 计算 `preference * (1 - fatigue)` 后贪心选动作

## 当前 baseline 结果记录

基于 `--episodes 200 --max-steps 100 --seed 42` 的当前结果为：

- `random`
  - `AvgReward = -2.265`
  - `AvgSteps = 17.735`
  - `AvgClicks = 7.735`

- `always_same_0`
  - `AvgReward = -8.235`
  - `AvgSteps = 11.765`
  - `AvgClicks = 1.765`

- `round_robin`
  - `AvgReward = -0.630`
  - `AvgSteps = 19.370`
  - `AvgClicks = 9.370`

- `least_fatigue`
  - `AvgReward = -0.725`
  - `AvgSteps = 19.275`
  - `AvgClicks = 9.275`

- `oracle_preference_greedy`
  - `AvgReward = -0.480`
  - `AvgSteps = 19.520`
  - `AvgClicks = 9.520`

- `observable_click_greedy`
  - `AvgReward = -0.480`
  - `AvgSteps = 19.520`
  - `AvgClicks = 9.520`

## 从 baseline 中得到的关键判断

### 1. 环境动力学是通的

`always_same_0` 显著最差，说明：

- 重复曝光同类目会迅速堆高 fatigue
- fatigue 机制确实在打击“无脑重复推送”

如果这一策略没有最差，说明环境的“越看越腻”逻辑就值得怀疑。现在它明显最差，属于良好 sanity check。

### 2. 当前环境里，疲劳控制比个性化更主导

`round_robin` 和 `least_fatigue` 已经非常接近 oracle，说明在当前这版环境里：

- 控制疲劳累积已经能解决大部分问题
- 知道真实偏好虽然有帮助，但帮助还没有大到拉开巨大差距

也就是说，当前世界的主矛盾是：

- 不要连续轰炸单一类目

而不是：

- 必须非常精准地识别个体偏好

### 3. fully observable 改造已经成功

`observable_click_greedy` 与 `oracle_preference_greedy` 完全一致，说明：

- 当前 observation 已经包含做这一决策所需的全部关键信息
- environment private state 与 agent observation 在决策层面已经对齐
- 当前教学版环境可以被视为 fully observable MDP

这是后续进入 Q-learning 的重要前提。

## 当前阶段的总体结论

到目前为止，可以比较稳定地给出以下判断：

1. `micro_recsim_env.py` 已经完成了最小 MDP 环境落地
2. 环境中的 fatigue / patience / reward 机制已经形成清晰的序列决策结构
3. baseline 框架已经搭起来，项目不再只是“一个能跑的环境文件”
4. 当前环境已经从 partial observable 教学起点过渡到 fully observable 教学版
5. `observable_click_greedy == oracle_preference_greedy` 证明当前 observation 足够支撑贪心决策

## 当前尚未开始的部分

当前还没有正式开始：

- 状态离散化
- tabular Q-learning
- ε-greedy 探索训练
- 学习曲线记录
- 和 baseline 的学习后对比

也就是说，项目现在处在“环境与评估基建已完成，准备进入第一个真正 RL agent”的节点。

## 最自然的下一步

下一步不建议直接写深度强化学习，而建议按以下顺序推进：

1. 设计离散状态编码器
   - 把当前 11 维连续 observation 压缩成适合表格法的小状态

2. 基于离散状态实现 tabular Q-learning
   - 先跑通最简单的 value update

3. 用同一套 baseline 框架进行比较
   - 看 learned policy 是否能接近或超过 `round_robin / least_fatigue`

当前最合适的项目切入点是：

- 先把环境讲清楚
- 再把状态压小
- 最后让 agent 学起来

而不是一开始就把问题升级到 DQN。
