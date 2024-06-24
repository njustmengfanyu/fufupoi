import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import random
import numpy as np
import matplotlib.pyplot as plt

# 指定使用 GPU 0 卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

# 随机化 150 张图片的标签
num_random_labels = 10000
num_classes = 10

random_indices = random.sample(range(len(trainset)), num_random_labels)
random_labels = [random.randint(0, num_classes - 1) for _ in range(num_random_labels)]

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for idx, new_label in zip(random_indices, random_labels):
    trainset.targets[idx] = new_label

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

# 加载测试集
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# 定义 ResNet-18 模型
model = resnet18(num_classes=10)
model = model.to(device)  # 将模型移动到 GPU

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# 训练模型
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到 GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:  # 每 100 个批次输出一次
            print(f'Epoch [{epoch+1}], Step [{batch_idx+1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
            train_losses.append(running_loss / 100)
            running_loss = 0.0

# 测试模型
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到 GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            test_accuracies.append(100 * correct / total)

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# 训练和测试循环
num_epochs = 85
for epoch in range(num_epochs):
    train(epoch)
    test()
    
# 保存整个模型
torch.save(model.state_dict(), f'{num_random_labels}label_complete_model.pth')

# 绘制训练和测试损失图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Loss over epochs')
plt.xlabel('step')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和测试准确率图
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='blue')
plt.title('Accuracy over epochs')
plt.xlabel('iteration')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.tight_layout()
plt.savefig(f'{num_random_labels}label.png')
