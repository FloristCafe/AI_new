全局战役路线图（The Master Blueprint）
阶段零：极速理论桥梁（The Bridge）—— 限时 3 天
核心目标： 消除对 ML/DL 专有名词的黑盒恐惧，建立输入、输出和误差反向传播的物理直觉。

软件要求： 无。只需要纸和笔。

学习资料：

吴恩达机器学习视频（仅限：逻辑回归、决策树基础原理）。

李宏毅深度学习视频（仅限：前三讲，死磕反向传播 Backpropagation）。

李航《统计学习方法》（仅读逻辑斯谛回归、决策树、提升方法的算法总结，不纠结推导）。

**阶段一：传统 ML 军火库（The Quant/Kaggle ML Arsenal）**
1. 树模型内核与损失函数定制：

推导 XGBoost 的泰勒二阶展开。手写 Asymmetric MSE 损失函数的梯度（Gradient）和海森矩阵（Hessian），并注入 LightGBM 进行训练。

2. 高维空间变换（PCA & 正交化）：

复习 SVD。使用 scikit-learn 和 numpy，在多重共线性的特征数据上实现 PCA 降维与施密特正交化。以满分的线性代数功底，这部分对你只是工程翻译。

3. 分布校验与对抗验证（Adversarial Validation）：

在 Kaggle 级数据集上构建二分类树模型，揪出导致训练集与测试集分布偏移的“内鬼特征”。

**阶段二：深度推荐的数据底座（The Bedrock）**
核心目标： 斩断对 Pandas 的依赖，解决千万级稀疏数据的内存溢出问题。

软件要求： Polars, PyArrow, PyTorch (DataLoader)。

工程任务： 独立处理 Kaggle Criteo 千万级数据集。实现高基数类别特征的 Hash 编码与低维稠密映射（Embedding）准备，输出极其紧凑的二进制 Parquet 文件。

**阶段三：从深度基线到序列决策（The RL Payload）**
核心目标： 满足辛鑫老师课题组的硬性需求，并跨越到强化学习。

工程任务 1（基线复现）： 纯手写 PyTorch 代码，复现包含 Embedding 与特征交叉的 DeepFM，建立极其严谨的 NDCG@K 和 Recall@K 验证体系。

工程任务 2（环境造物主）： 引入 Gymnasium。将你的数据集转化为马尔可夫决策过程（MDP）。构建包含状态空间（历史序列 Embedding）、动作空间和奖励函数的自定义强化学习环境。