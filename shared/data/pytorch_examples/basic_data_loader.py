import torch
from torch.utils.data import DataLoader

# 假设我们导入了上一节课写的 Dataset
# from 01_custom_dataset import CatDogDataset

# ==========================================
# 1. 实例化弹药库 (Dataset)
# ==========================================
# 假装我们传入了图片文件夹路径
# my_dataset = CatDogDataset(data_dir='./my_images')

# 为了让你现在就能跑通这行代码，我们用 PyTorch 自带的 TensorDataset 模拟一个假的数据集
# 假装我们有 1000 张图片，每张图片被拉成了长度为 10 的向量，标签是 0 或 1
dummy_images = torch.randn(1000, 10)
dummy_labels = torch.randint(0, 2, (1000,))
from torch.utils.data import TensorDataset

my_dataset = TensorDataset(dummy_images, dummy_labels)

print(f"弹药库总容量: {len(my_dataset)}")

# ==========================================
# 2. 组装卡车 (DataLoader)
# ==========================================
# 核心参数解析：
# batch_size: 每次打包多少张图片。这是你要经常调的超参数，太大显存会爆(OOM)，太小训练极慢。
# shuffle: 是否打乱顺序。训练集必须设为 True（防止模型死记硬背），测试集设为 False。
# num_workers: 极其重要！开启多少个子进程去硬盘里搬砖。Windows下容易报错，通常设为 0 或 2。

train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=32,  # 每箱 32 张图片
    shuffle=True,  # 装箱前先把所有数据打乱
    num_workers=0,  # 单进程运作（新手保命设置）
    drop_last=False  # 是否丢弃最后装不满的那一箱
)

# ==========================================
# 3. 模拟真实的训练流水线 (Training Loop 雏形)
# ==========================================
# 在训练时，我们要遍历这辆卡车运来的每一箱数据
# epoch (轮次) 代表我们要把这 1000 张图片反复学多少遍

epochs = 2
for epoch in range(epochs):
    print(f"\n--- 开始第 {epoch + 1} 轮 (Epoch) 训练 ---")

    # enumerate 会返回这是第几批 (step)，以及这一批的具体数据 (batch_data)
    for step, (batch_images, batch_labels) in enumerate(train_loader):

        # 此时，batch_images 的维度是 [32, 10] (32张图，每张10个特征)
        # batch_labels 的维度是 [32]

        if step % 10 == 0:  # 每隔 10 个批次打印一次进度
            print(f"当前进度: 第 {step} 批次, 本批次数据形状: {batch_images.shape}, 标签形状: {batch_labels.shape}")

        # 接下来就是把 batch_images 扔进神经网络，计算误差，反向传播... (这是下一周期的内容)