import os
import torch
from torch.utils.data import Dataset

# 注意：处理图像必须依赖第三方库，最常用的是 PIL (Python Imaging Library)
# 如果你没装，稍后需要在命令行执行: pip install pillow
from PIL import Image


class CatDogDataset(Dataset):#括号内抽象类直接继承
    def __init__(self, data_dir, transform=None):
        """
        初始化：绝对不要在这里读取图片内容！只读路径！
        """
        self.data_dir = data_dir
        self.transform = transform

        # 假设你的文件夹里全是 'cat_01.jpg', 'dog_02.jpg' 这种文件
        # os.listdir 会列出该文件夹下所有的文件名
        self.image_names = os.listdir(data_dir)

        print(f"成功扫描到 {len(self.image_names)} 张图片路径，等待随时调用。")

    def __len__(self):
        # 告诉 DataLoader，一共有多少弹药
        return len(self.image_names)

    def __getitem__(self, index):
        """
        真正的干活区域：按需读取，用完即毁，绝不占用多余内存。
        """
        # 1. 拿到具体的文件名，拼接成完整的绝对路径
        img_name = self.image_names[index]
        img_path = os.path.join(self.data_dir, img_name)

        # 2. 真实物理读取：打开硬盘上的图片文件，转为 RGB 格式
        image = Image.open(img_path).convert('RGB')

        # 3. 标签生成：用你刚学过的 Pythonic 思维，直接从文件名里扣出类别
        # 如果文件名包含 'dog'，标签为 1，否则为 0
        label = 1 if 'dog' in img_name.lower() else 0

        # 4. 数据预处理（转换尺寸、转化为张量 Tensor 等）
        if self.transform:
            image = self.transform(image)

        # 5. 返回一个元组 (数据, 标签)，供模型享用
        return image, label


# ==========================================
# 模拟测试环境（假装你有一个图片文件夹）
# ==========================================
if __name__ == '__main__':
    # 这里只是为了让你跑通代码不报错而造的假路径
    # 实际跑的时候，你需要把 './fake_data' 换成你电脑里真正的图片文件夹路径
    print("这只是一个 Dataset 模板，如果没有真实的图片文件夹传入，实例化会报错。")
    print("仔细阅读 __getitem__ 里的 5 步逻辑。")