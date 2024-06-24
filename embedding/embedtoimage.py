import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)

# 修改模型以输出嵌入向量（移除最后一层分类层）
model = nn.Sequential(*list(model.children())[:-1])

# 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.Resize(224),  # 调整 CIFAR-10 图片大小以匹配 ResNet-18 输入要求
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# 获取嵌入向量和标签
model.eval()
embeddings = []
labels = []

with torch.no_grad():
    for images, targets in dataloader:
        outputs = model(images).squeeze()
        embeddings.append(outputs)
        labels.append(targets)

embeddings = torch.cat(embeddings).cpu().numpy()
labels = torch.cat(labels).cpu().numpy()

# 使用 t-SNE 将嵌入向量降维到 2D 空间
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 绘制嵌入向量的 2D 可视化
plt.figure(figsize=(10, 10))
for i in range(10):
    indices = np.where(labels == i)
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=dataset.classes[i], alpha=0.6)

plt.legend()
plt.title("t-SNE Visualization of ResNet-18 Embeddings on CIFAR-10")
plt.show()
