import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载训练数据集
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=8)

# 加载测试数据集
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

# 定义 ResNet-18 模型
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Identity()  # 去除最后一层，全连接层不需要
    
    def forward(self, x):
        return self.model(x)

model = ResNet18().to(device)
model.eval()

# 获取图像嵌入
def get_embeddings(dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            images, target = data
            images, target = images.to(device), target.to(device)
            output = model(images)
            embeddings.append(output)
            labels.append(target)
    embeddings = torch.cat(embeddings)
    print(embeddings.shape)
    labels = torch.cat(labels)
    print(labels.shape)
    return embeddings, labels

train_embeddings, train_labels = get_embeddings(trainloader)
test_embeddings, test_labels = get_embeddings(testloader)

# 将 PyTorch 张量转换为 NumPy 数组
train_embeddings_np = train_embeddings.cpu().numpy()
train_labels_np = train_labels.cpu().numpy()

# 使用 t-SNE 降维到2D
tsne = TSNE(n_components=2, random_state=42)
train_embeddings_2d = tsne.fit_transform(train_embeddings_np)

# 创建散点图
plt.figure(figsize=(10, 10))
scatter = plt.scatter(train_embeddings_2d[:, 0], train_embeddings_2d[:, 1], c=train_labels_np, cmap='tab10', s=10)
plt.colorbar(scatter, ticks=range(10))
plt.title('CIFAR-10 Embeddings')
plt.savefig('cifar10_embeddings.png')
plt.show()
