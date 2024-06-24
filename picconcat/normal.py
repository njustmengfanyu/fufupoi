import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载 CIFAR-10 数据集
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# 随机选择第 0 类和第 1 类的图片
class0_images = [img for img, label in dataset if label == 0]
class1_images = [img for img, label in dataset if label == 1]

# 确保我们有足够数量的图片
np.random.shuffle(class0_images)
np.random.shuffle(class1_images)
class0_images = class0_images[:10]
class1_images = class1_images[:10]

# 切割和拼接图片
new_images = []
new_labels = []  # 新标签列表，所有标签都设置为 2

for img0, img1 in zip(class0_images, class1_images):
    left0, right0 = img0[:, :16, :], img0[:, 16:, :]
    left1, right1 = img1[:, :16, :], img1[:, 16:, :]
    # 随机选择拼接方式
    if np.random.rand() > 0.5:
        new_img = torch.cat([left0, right1], dim=1)
    else:
        new_img = torch.cat([left1, right0], dim=1)
    new_images.append(new_img)
    new_labels.append(2)  # 添加标签 2

# 将新图片和标签转换为 DataLoader
new_dataset = TensorDataset(torch.stack(new_images), torch.tensor(new_labels))
new_dataloader = DataLoader(new_dataset, batch_size=10, shuffle=False)

# 显示图片的函数
def show_images(dataloader):
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            plt.imshow(img.permute(1, 2, 0) / 2 + 0.5)  # 将图片规范化到 [0, 1]
            plt.title(f'Label: {label}')
            plt.show()

# 显示拼接的图片
show_images(new_dataloader)

# 接下来，你可以使用 new_dataloader 来训练你的 ResNet-18 模型。
# 请确保你的模型、损失函数和优化器已经定义好，并且可以处理新的数据集。